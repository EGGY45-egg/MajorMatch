import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from app_logic import (
    coerce_profile_values,
    default_profile_values,
    missing_profile_fields,
    merge_profile_values,
    profile_to_text,
)


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
PREFERRED_CHAT_MODELS = (
    "llama3.2:1b",
    "llama3.2",
    "llama3.1",
    "llama2",
)


@dataclass(frozen=True)
class OllamaInterviewResult:
    reply: str
    profile: Dict[str, int]
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
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


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


def _fallback_reply(profile: Dict[str, int]) -> OllamaInterviewResult:
    missing = missing_profile_fields(profile)
    if missing:
        field = missing[0]
        prompts = {
            "coding": "What kind of coding tasks do you feel strongest in: building apps, solving logic problems, or scripting?",
            "math": "How comfortable are you with math and data work: basic, moderate, or strong?",
            "design": "How would you rate your interest in design, visuals, or user experience work?",
        }
        return OllamaInterviewResult(
            reply=prompts[field],
            profile=coerce_profile_values(profile),
            complete=False,
            used_fallback=True,
        )
    return OllamaInterviewResult(
        reply="I have enough to make a recommendation now.",
        profile=coerce_profile_values(profile),
        complete=True,
        used_fallback=True,
    )


def _question_for_missing_fields(profile: Dict[str, int]) -> str:
    missing = missing_profile_fields(profile)
    if not missing:
        return "I have enough to make a recommendation now."

    field = missing[0]
    prompts = {
        "coding": "What kind of coding tasks do you feel strongest in: building apps, solving logic problems, or scripting?",
        "math": "How comfortable are you with math and data work: basic, moderate, or strong?",
        "design": "How would you rate your interest in design, visuals, or user experience work?",
    }
    return prompts[field]


def interview_profile(
    messages: List[Dict[str, str]],
    current_profile: Dict[str, int],
    model: str = OLLAMA_MODEL,
) -> OllamaInterviewResult:
    current_profile = coerce_profile_values(current_profile)
    latest_user_text = _latest_user_message(messages)
    inferred_profile = _infer_profile_from_text(latest_user_text, current_profile)
    if not ollama_is_available():
        result = _fallback_reply(inferred_profile)
        return OllamaInterviewResult(
            reply=_question_for_missing_fields(result.profile),
            profile=result.profile,
            complete=result.complete,
            raw=result.raw,
            used_fallback=True,
        )

    resolved_model = resolve_chat_model(model)

    system_prompt = (
        "You are MajorMatch, a concise interview assistant for students. "
        "Have a natural conversation with the user and ask one short follow-up question at a time. "
        "Use the conversation to estimate or update a profile with coding, math, and design scores from 0 to 10. "
        "Reply in plain text. Do not emit code fences or JSON unless the user explicitly asks for it. "
        "If the user already provided enough information, briefly acknowledge that and say you can make a recommendation now."
    )

    payload_messages = [{"role": "system", "content": system_prompt}]
    payload_messages.extend(messages)
    payload_messages.append(
        {
            "role": "system",
            "content": (
                f"Current structured profile: {profile_to_text(current_profile)}. "
                f"Missing fields: {', '.join(missing_profile_fields(current_profile)) or 'none'}."
            ),
        }
    )

    payload = {
        "model": resolved_model,
        "messages": payload_messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        response = _post_json(f"{OLLAMA_BASE_URL}/api/chat", payload)
        content = response.get("message", {}).get("content", "")
        parsed = _extract_json_object(content)
        profile_updates = parsed.get("profile", {}) if parsed else {}
        profile = merge_profile_values(inferred_profile, profile_updates)

        reply = content.strip()
        if parsed and isinstance(parsed.get("reply"), str):
            reply = str(parsed.get("reply", reply)).strip()

        if not reply or reply in {"{", "}", "{}"}:
            reply = _question_for_missing_fields(profile)

        if parsed and "complete" in parsed:
            complete = bool(parsed.get("complete", False))
        else:
            complete = not missing_profile_fields(profile)
        return OllamaInterviewResult(
            reply=reply,
            profile=profile,
            complete=complete,
            raw=content,
            used_fallback=False,
        )
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return OllamaInterviewResult(
            reply=_question_for_missing_fields(inferred_profile),
            profile=inferred_profile,
            complete=not missing_profile_fields(inferred_profile),
            used_fallback=True,
        )
