import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


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


