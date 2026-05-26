import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# profile-related helpers were removed from `app_logic.py` when the
# structured profile concept was removed; this module no longer imports them.


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:latest")
OLLAMA_REQUEST_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "120"))
PREFERRED_CHAT_MODELS = (
    "llama2:latest",
    "llama3.2:1b",
    "llama3.2",
    "llama3.1",
    "llama2",
)


@dataclass(frozen=True)
class OllamaInterviewResult:
    reply: str
    complete: bool
    raw: str = ""
    used_fallback: bool = False


def ollama_is_available(base_url: str = OLLAMA_BASE_URL) -> bool:
    try:
        request = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def list_local_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    try:
        request = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models", [])
        names = []
        for model in models:
            name = model.get("name")
            if name:
                names.append(str(name))
        return names
    except Exception:
        return []


def resolve_chat_model(requested_model: Optional[str] = None, base_url: str = OLLAMA_BASE_URL) -> str:
    if requested_model:
        return requested_model

    # Honor an explicit environment-configured `OLLAMA_MODEL` as the primary choice.
    # This forces usage of the configured model (e.g., 'llama2:latest') unless a
    # `requested_model` is explicitly passed by the caller.
    if OLLAMA_MODEL:
        return OLLAMA_MODEL

    available = list_local_models(base_url)
    for preferred in PREFERRED_CHAT_MODELS:
        if preferred in available:
            return preferred
    if available:
        return available[0]
    return OLLAMA_MODEL


def _post_json(url: str, payload: Dict[str, object]) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as he:
        # Try to read the response body to surface server-side validation errors
        try:
            body = he.read().decode("utf-8")
        except Exception:
            body = "<unable to read response body>"
        raise RuntimeError(f"HTTP {he.code} {he.reason}: {body}") from he


def chat_completion_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    options: Optional[Dict[str, Any]] = None,
    base_url: str = OLLAMA_BASE_URL,
):
    """Create a streaming chat completion generator that yields text chunks.

    This performs a POST with `stream=True` and yields bytes as decoded
    UTF-8 fragments. Consumers should provide a callback to consume chunks
    and assemble the final assistant reply.
    """
    resolved_model = resolve_chat_model(model, base_url)
    payload: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS) as response:
            # Ollama streams newline-delimited JSON objects. Yield only the
            # assistant text fragments so callers do not render raw payloads.
            while True:
                line = response.readline()
                if not line:
                    break
                try:
                    text = line.decode("utf-8")
                except Exception:
                    text = line.decode("utf-8", errors="ignore")
                cleaned = text.strip()
                if not cleaned:
                    continue
                try:
                    payload = json.loads(cleaned)
                except json.JSONDecodeError:
                    yield cleaned
                    continue
                message = payload.get("message", {}) if isinstance(payload, dict) else {}
                content = str(message.get("content", "") or "")
                if content:
                    yield content
    except urllib.error.HTTPError as he:
        try:
            body = he.read().decode("utf-8")
        except Exception:
            body = "<unable to read response body>"
        raise RuntimeError(f"HTTP {he.code} {he.reason}: {body}") from he


def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    options: Optional[Dict[str, Any]] = None,
    base_url: str = OLLAMA_BASE_URL,
) -> Dict[str, Any]:
    """Send a chat completion request to Ollama.

    This shared helper lets the interview flow and the agent orchestrator use
    the same transport while optionally enabling tool calling.
    """
    resolved_model = resolve_chat_model(model, base_url)
    # Some local Ollama models (for example: llama2:latest) do not support the
    # `tools` parameter. If a tools list is provided and the resolved model is
    # known to not support tools, try to fall back to another available model
    # that may support function/tool calling.
    if tools and resolved_model and "llama2" in resolved_model:
        try:
            available = list_local_models(base_url)
            for pref in PREFERRED_CHAT_MODELS:
                if pref != resolved_model and pref in available:
                    resolved_model = pref
                    break
        except Exception:
            # If listing models fails, continue with the originally resolved model.
            pass
    payload: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if options:
        payload["options"] = options
    return _post_json(f"{base_url}/api/chat", payload)


def _extract_json_object(text: str) -> Optional[Dict[str, object]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _latest_user_message(messages: List[Dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _infer_score_from_text(text: str, field: str, current_score: int) -> int:
    lowered = text.lower()
    field_terms = {
        "coding": ["coding", "code", "programming", "developer", "software", "apps", "debug", "logic"],
        "math": ["math", "maths", "algebra", "statistics", "stats", "data", "numbers", "equations"],
        "design": ["design", "drawing", "drawing", "ux", "ui", "visual", "art", "creative", "graphics"],
    }

    strong_terms = ["good at", "strong at", "best at", "love", "really good", "very good", "enjoy"]
    weak_terms = ["not good at", "not so good", "bad at", "weak at", "struggle with", "hate", "poor at"]
    neutral_terms = ["okay with", "fine with", "comfortable with", "average at", "somewhat"]

    score = current_score
    if any(term in lowered for term in field_terms.get(field, [])):
        score += 2

    if any(term in lowered for term in strong_terms):
        score += 2
    if any(term in lowered for term in weak_terms):
        score -= 2
    if any(term in lowered for term in neutral_terms):
        score += 1

    if field == "coding" and any(term in lowered for term in ["coding", "code", "programming", "developer", "software", "apps", "debug", "logic"]):
        if any(term in lowered for term in strong_terms):
            score += 3
        elif any(term in lowered for term in weak_terms):
            score -= 3
    if field == "math" and any(term in lowered for term in ["math", "statistics", "stats", "data", "numbers", "equations"]):
        if any(term in lowered for term in strong_terms):
            score += 3
        elif any(term in lowered for term in weak_terms):
            score -= 3
    if field == "design" and any(term in lowered for term in ["design", "drawing", "ux", "ui", "visual", "art", "creative", "graphics"]):
        if any(term in lowered for term in strong_terms):
            score += 3
        elif any(term in lowered for term in weak_terms):
            score -= 3

    return max(0, min(10, score))


def _infer_profile_from_text(text: str, current_profile: Dict[str, int]) -> Dict[str, int]:
    updated = coerce_profile_values(current_profile)
    if not text.strip():
        return updated

    for field in ("coding", "math", "design"):
        updated[field] = _infer_score_from_text(text, field, updated[field])

    if "not good at drawing" in text.lower() or "not so good at drawing" in text.lower():
        updated["design"] = min(updated["design"], 2)
    if "good at coding" in text.lower() or "best at coding" in text.lower():
        updated["coding"] = max(updated["coding"], 8)
    if "good at math" in text.lower() or "also good at math" in text.lower():
        updated["math"] = max(updated["math"], 8)

    return updated


def interview_profile(
    messages: List[Dict[str, str]],
    model: str = OLLAMA_MODEL,
) -> OllamaInterviewResult:
    if not ollama_is_available():
        raise ConnectionError(f"Ollama is not reachable at {OLLAMA_BASE_URL}")

    system_prompt = (
        "You are MajorMatch, a concise interview assistant for students. "
        "Have a natural conversation with the user and ask one short follow-up question at a time. "
        "Use the conversation to estimate or update a profile with coding, math, and design scores from 0 to 10. "
        "Reply in plain text. Do not emit code fences or JSON unless the user explicitly asks for it. "
        "If the user already provided enough information, briefly acknowledge that and say you can make a recommendation now."
    )

    payload_messages = [{"role": "system", "content": system_prompt}]
    payload_messages.extend(messages)
    # Do not include the user's structured profile in the system prompt to
    # avoid the assistant making assumptions or referencing profile values
    # inferred from previous messages. The interview should rely on explicit
    # structured `profile` updates returned by the model or user inputs.

    try:
        response = chat_completion(payload_messages, model=model, options={"temperature": 0.2})
        content = response.get("message", {}).get("content", "")
        parsed = _extract_json_object(content)

        reply = content.strip()
        if parsed and isinstance(parsed.get("reply"), str):
            reply = str(parsed.get("reply", reply)).strip()

        if not reply or reply in {"{", "}", "{}"}:
            raise ValueError(f"Ollama returned unusable reply: {content!r}")

        if parsed and "complete" in parsed:
            complete = bool(parsed.get("complete", False))
        else:
            complete = False
        return OllamaInterviewResult(
            reply=reply,
            complete=complete,
            raw=content,
            used_fallback=False,
        )
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        raise
