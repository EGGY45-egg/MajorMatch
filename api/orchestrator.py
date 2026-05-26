from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app_logic import ProfileInput, coerce_profile_values, profile_to_text, recommend_track, suggest_career_context, suggest_courses
from course_index import project_courses_with_query
from api.ollama import chat_completion, resolve_chat_model


ToolCallable = Callable[[List[Dict[str, str]], Optional[str]], Dict[str, Any]]


def _clean_assistant_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    markers = [
        r"<\|start_header_id\|>assistant<\|end_header_id\|>",
        r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*",
        r"<\|start_header_id\|>",
        r"<\|end_header_id\|>",
    ]
    for marker in markers:
        cleaned = re.sub(marker, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"^assistant\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _is_normal_chat_question(user_message: str) -> bool:
    lowered = (user_message or "").strip().lower()
    if not lowered:
        return True

    greeting_patterns = [
        r"^(hi|hello|hey|yo|sup)[\W_]*$",
        r"^(hi|hello|hey|yo|sup)\b",
        r"what are you\??$",
        r"who are you\??$",
        r"tell me about yourself\??$",
        r"introduce yourself\??$",
    ]
    for pattern in greeting_patterns:
        if re.search(pattern, lowered):
            return True
    return False


def _friendly_identity_reply() -> str:
    return (
        "I am MajorMatch, an AI assistant that helps you choose courses and career paths. "
        "I can answer questions directly, and I’ll use tools only when they add value."
    )


@dataclass(frozen=True)
class ToolTrace:
    name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]


@dataclass(frozen=True)
class OrchestratorResult:
    reply: str
    profile: Dict[str, int]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    tool_trace: List[ToolTrace] = field(default_factory=list)
    raw: str = ""


def build_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "predict_track",
                "description": "Predict the most likely career track based on skill. If called without numeric inputs, the front-end should open the prediction UI to collect user inputs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coding": {"type": "integer", "minimum": 0, "maximum": 10},
                        "math": {"type": "integer", "minimum": 0, "maximum": 10},
                        "design": {"type": "integer", "minimum": 0, "maximum": 10},
                        "open_ui": {"type": "boolean", "description": "If true, request the front-end to open the predict UI for interactive input."},
                    }
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_career_context",
                "description": "Fetch live job-market context for a recommended career track.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "track": {"type": "string"},
                        "location": {"type": "string"},
                    },
                    "required": ["track"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_semantic_search",
                "description": "Search the course corpus semantically and generate a projection for visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {"type": "string", "description": "The user's course exploration or learning question."},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                        "projection_method": {
                            "type": "string",
                            "enum": ["pca", "umap", "tsne"],
                            "default": "pca",
                        },
                    },
                    "required": ["user_query"],
                },
            },
        },
    ]


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    # Normalize dict-like inputs and try to recover from nested/serialized payloads
    if isinstance(arguments, dict):
        parsed: Dict[str, Any] = dict(arguments)
    elif not arguments:
        return {}
    elif isinstance(arguments, str):
        try:
            loaded = json.loads(arguments)
            parsed = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    else:
        return {}

    # Some models/agents return parameters nested inside arrays or as JSON-encoded strings.
    # Attempt to unwrap common wrappers so callers can find keys like 'track', 'user_query', etc.
    # Example malformed shapes handled:
    # - {'object': "[{'track': 'Healthcare', 'location': 'United States'}]"}
    # - {'parameters': [{'object': "{...}"}]}
    # - {'0': {'track': 'X'}}

    # If single key 'object' contains a JSON string or list, try to decode it.
    if "object" in parsed and isinstance(parsed["object"], str):
        try:
            inner = json.loads(parsed["object"])
            if isinstance(inner, list) and inner:
                # if list of dicts, merge first dict
                if isinstance(inner[0], dict):
                    parsed.update(inner[0])
            elif isinstance(inner, dict):
                parsed.update(inner)
        except json.JSONDecodeError:
            pass

    # If any values are JSON strings, try to decode them too.
    for k, v in list(parsed.items()):
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                decoded = json.loads(v)
                if isinstance(decoded, dict):
                    parsed[k] = decoded
                    parsed.update(decoded)
            except json.JSONDecodeError:
                pass
        if isinstance(v, list) and v and isinstance(v[0], dict):
            # flatten single-item parameter lists
            parsed.update(v[0])

    # Also handle 'parameters' wrappers
    if "parameters" in parsed and isinstance(parsed["parameters"], list) and parsed["parameters"]:
        first = parsed["parameters"][0]
        if isinstance(first, dict):
            parsed.update(first)

    return {k: v for k, v in parsed.items()}


def _extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = call.get("name") or function.get("name")
        if not name:
            continue
        normalized.append(
            {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "name": name,
                "arguments": _parse_tool_arguments(call.get("arguments") or function.get("arguments")),
            }
        )
    return normalized


def _execute_tool(name: str, arguments: Dict[str, Any], profile: Dict[str, int], location: str) -> Dict[str, Any]:
    if name == "predict_track":
        merged_profile = coerce_profile_values({**profile, **arguments})
        # If the tool is called without explicit numeric inputs (or with open_ui=True),
        # instruct the front-end to open the interactive prediction UI instead of
        # attempting to parse values from the model output.
        provided_numeric = any(k in arguments for k in ("coding", "math", "design"))
        if not provided_numeric or arguments.get("open_ui") is True:
            return {
                "profile": merged_profile,
                "action": "open_ui",
                "message": "open_predict_ui",
            }
        recommendation = recommend_track(ProfileInput(**merged_profile))
        return {
            "profile": merged_profile,
            "prediction": recommendation,
        }

    if name == "get_career_context":
        track = str(arguments.get("track") or "").strip()
        effective_location = str(arguments.get("location") or location or "United States")
        context = suggest_career_context(track, location=effective_location)
        return context.to_dict()

    if name == "execute_semantic_search":
        query = str(arguments.get("user_query") or "").strip()
        top_k = max(1, min(int(arguments.get("top_k") or 5), 5))
        projection_method = str(arguments.get("projection_method") or "pca").strip().lower() or "pca"
        if projection_method == "tnse":
            projection_method = "tsne"
        results = suggest_courses(query, top_k=top_k)

        projection_methods: Dict[str, Dict[str, Any]] = {}
        for method in ("pca", "umap", "tsne"):
            try:
                course_points, query_point = project_courses_with_query(query, method=method)
                projection_methods[method] = {
                    "available": True,
                    "method": method,
                    "courses": [
                        {
                            "id": point.id,
                            "title": point.title,
                            "description": point.description,
                            "x": point.x,
                            "y": point.y,
                        }
                        for point in course_points
                    ],
                    "query_point": None
                    if query_point is None
                    else {
                        "id": query_point.id,
                        "title": query_point.title,
                        "description": query_point.description,
                        "x": query_point.x,
                        "y": query_point.y,
                    },
                }
            except Exception as error:
                projection_methods[method] = {
                    "available": False,
                    "method": method,
                    "error": str(error),
                }

        selected_projection = projection_methods.get(projection_method) or projection_methods.get("pca", {})
        if selected_projection.get("available") is not True:
            selected_projection = next(
                (value for value in projection_methods.values() if value.get("available")),
                selected_projection,
            )

        return {
            "query": query,
            "top_k": top_k,
            "results": results,
            "projection": {
                **selected_projection,
                "methods": projection_methods,
            },
        }

    raise ValueError(f"Unsupported tool: {name}")


def _build_final_response_prompt(tool_trace: List[ToolTrace], artifacts: Dict[str, Any]) -> str:
    # Global instruction: short one-line summary + up to 2 concise bullets.
    sections: List[str] = [
        "Write one short, user-facing reply grounded in the tool results below.",
        "Format: 1) single-sentence summary, 2) up to two short bullet points (concise).",
        "Do not include suggested next steps or follow-ups; keep the reply strictly descriptive.",
    ]

    for trace in tool_trace:
        if trace.name == "predict_track":
            prediction = artifacts.get("prediction") or {}
            sections.append(
                "Career recommendation result: "
                f"track={prediction.get('track', 'unknown')}, "
                f"confidence={float(prediction.get('confidence', 0.0)):.2f}. "
                "Tone: concise, prescriptive — state the top recommendation in one sentence and two bullets of supporting signals."
            )
        elif trace.name == "get_career_context":
            career_context = artifacts.get("career_context") or {}
            sections.append(
                "Career market context result: "
                f"available={career_context.get('available', False)}, "
                f"track={career_context.get('track', 'unknown')}, "
                f"job_count={career_context.get('job_count')}, "
                f"salary_min={career_context.get('salary_min')}, "
                f"salary_max={career_context.get('salary_max')}. "
                "Use a strict database-summary style: start with 'Based on the career context tool, the results are:' and then report only the exact job_count, salary_min, salary_max, top_job_titles, and top_companies from the tool output. Do not invent example jobs, do not change the role names, and do not add interpretation beyond a brief note that these are tool results."
            )
        elif trace.name == "execute_semantic_search":
            semantic_search = artifacts.get("semantic_search") or {}
            results = semantic_search.get("results") or []
            top_titles = [item.get("title", "Untitled") for item in results[:5]]
            projection = semantic_search.get("projection") or {}
            sections.append(
                "Semantic search result: "
                f"query='{semantic_search.get('query', '')}', "
                f"top_matches={top_titles}, "
                f"projection_available={projection.get('available', False)}. "
                "Use a strict database-summary style: start with 'Based on what is in my database using the semantic search tool, the results are:' and then list the top matches exactly as returned. Do not add interpretation, ranking claims, or extra recommendations. If a map/projection is available, mention only that the visualization is shown below."
            )

    return "\n\n".join(sections)


def run_orchestrated_assistant(
    user_message: str,
    current_profile: Dict[str, int],
    *,
    location: str = "United States",
    model: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    max_steps: int = 4,
    chat_fn: Callable[..., Dict[str, Any]] = chat_completion,
) -> OrchestratorResult:
    profile = coerce_profile_values(current_profile)
    resolved_model = resolve_chat_model(model)

    if _is_normal_chat_question(user_message):
        return OrchestratorResult(
            reply=_friendly_identity_reply() if re.search(r"\b(what are you|who are you|tell me about yourself|introduce yourself)\b", user_message.lower()) else "Hello. I am MajorMatch, an AI assistant that helps with courses and careers.",
            profile=profile,
            artifacts={},
            tool_trace=[],
            raw="",
        )

    system_prompt = (
        "You are MajorMatch's orchestrator. Use tools automatically whenever they are relevant. "
        "Do not call tools for greetings, introductions, identity questions, or other normal chat. "
        "If the user asks what you are or says hello, answer directly with a friendly plain-language introduction. "
        "Call predict_track when the user's profile or skill fit needs a recommendation. "
        "Call get_career_context when the user asks about market demand, salary, or career outlook. "
        "Call execute_semantic_search when the user asks for courses, learning resources, course planning, or a visual map of relevant options. "
        "When no tool is needed, respond naturally and user-friendly without mentioning tools. "
        "Keep replies concise, grounded in tool results when used, and user-facing."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": f"Current profile: {profile_to_text(profile)}. Preferred location: {location}.",
        },
    ]
    if conversation_history:
        messages.extend(conversation_history)
    else:
        messages.append({"role": "user", "content": user_message})

    tool_schemas = build_tool_schemas()
    trace: List[ToolTrace] = []
    artifacts: Dict[str, Any] = {}
    last_content = ""
    raw = ""

    for _ in range(max_steps):
        response = chat_fn(messages, model=resolved_model, tools=tool_schemas, options={"temperature": 0.2})
        raw = json.dumps(response)
        message = response.get("message", {}) if isinstance(response, dict) else {}
        last_content = str(message.get("content", "") or "").strip()
        tool_calls = _extract_tool_calls(message)

        if not tool_calls:
            if trace:
                final_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": _build_final_response_prompt(trace, artifacts),
                    },
                ]
                final_response = chat_fn(
                    final_messages,
                    model=resolved_model,
                    options={"temperature": 0.2},
                )
                final_message = final_response.get("message", {}) if isinstance(final_response, dict) else {}
                final_content = _clean_assistant_text(str(final_message.get("content", "") or ""))
                if final_content:
                    raw = json.dumps(final_response)
                    return OrchestratorResult(
                        reply=final_content,
                        profile=profile,
                        artifacts=artifacts,
                        tool_trace=trace,
                        raw=raw,
                    )
            return OrchestratorResult(
                reply=_clean_assistant_text(last_content) or "I could not generate a response.",
                profile=profile,
                artifacts=artifacts,
                tool_trace=trace,
                raw=raw,
            )

        assistant_message: Dict[str, Any] = {"role": "assistant", "content": last_content, "tool_calls": tool_calls}
        messages.append(assistant_message)

        for call in tool_calls:
            name = str(call.get("name") or "")
            arguments = call.get("arguments") or {}
            result = _execute_tool(name, arguments, profile, location)
            trace.append(ToolTrace(name=name, arguments=arguments, result=result))

            if name == "predict_track":
                profile = coerce_profile_values(result.get("profile") or profile)
                artifacts["prediction"] = result.get("prediction")
                artifacts["profile"] = profile
            elif name == "get_career_context":
                artifacts["career_context"] = result
            elif name == "execute_semantic_search":
                artifacts["semantic_search"] = result

            tool_message: Dict[str, Any] = {
                "role": "tool",
                "content": json.dumps(result),
                "name": name,
            }
            if call.get("id"):
                tool_message["tool_call_id"] = call["id"]
            messages.append(tool_message)

    return OrchestratorResult(
        reply=last_content or "I reached the tool limit before producing a final response.",
        profile=profile,
        artifacts=artifacts,
        tool_trace=trace,
        raw=raw,
    )
