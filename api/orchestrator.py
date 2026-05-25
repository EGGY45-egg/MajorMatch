from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app_logic import ProfileInput, coerce_profile_values, profile_to_text, recommend_track, suggest_career_context, suggest_courses
from course_index import project_courses_with_query
from api.ollama import chat_completion, resolve_chat_model


ToolCallable = Callable[[List[Dict[str, str]], Optional[str]], Dict[str, Any]]


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
                "description": "Predict the most likely career track from coding, math, and design skill ratings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coding": {"type": "integer", "minimum": 0, "maximum": 10},
                        "math": {"type": "integer", "minimum": 0, "maximum": 10},
                        "design": {"type": "integer", "minimum": 0, "maximum": 10},
                    },
                    "required": ["coding", "math", "design"],
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
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


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
        top_k = int(arguments.get("top_k") or 5)
        projection_method = str(arguments.get("projection_method") or "pca").strip().lower() or "pca"
        results = suggest_courses(query, top_k=top_k)

        projection: Dict[str, Any]
        try:
            course_points, query_point = project_courses_with_query(query, method=projection_method)
            projection = {
                "available": True,
                "method": projection_method,
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
            projection = {
                "available": False,
                "method": projection_method,
                "error": str(error),
            }

        return {
            "query": query,
            "top_k": top_k,
            "results": results,
            "projection": projection,
        }

    raise ValueError(f"Unsupported tool: {name}")


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

    system_prompt = (
        "You are MajorMatch's orchestrator. Use tools automatically whenever they are relevant. "
        "Call predict_track when the user's profile or skill fit needs a recommendation. "
        "Call get_career_context when the user asks about market demand, salary, or career outlook. "
        "Call execute_semantic_search when the user asks for courses, learning resources, course planning, or a visual map of relevant options. "
        "Keep replies concise, grounded in tool results, and user-facing."
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
            return OrchestratorResult(
                reply=last_content or "I could not generate a response.",
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
