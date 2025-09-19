import os, json, re, time
from typing import Any, Dict, List, Tuple
import google.generativeai as genai

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def build_prompt(question: str, contexts: List[dict], locale: str | None = "en") -> Tuple[str, str]:
    system_text = (
        "You are MedBot, a helpful assistant for laypeople. "
        "Always be clear, neutral, and concise. "
        "You are NOT a medical professional; do not provide diagnosis. "
        "Provide educational information and when appropriate recommend seeking professional care. "
        "If the question is unrelated to health, answer briefly or say you don't know.\n\n"
        "STRUCTURED JSON OUTPUT ONLY (no extra text, no markdown fences). "
        "Respond with a JSON object:\n"
        "{\n"
        '  "answer": string,\n'
        '  "citations": string[],\n'
        '  "safety_notice": string,\n'
        '  "follow_up_questions": string[]\n'
        "}\n"
    )
    ctx_lines = []
    for i, c in enumerate(contexts, 1):
        cid = c.get("id") or c.get("_id") or f"ctx-{i}"
        snip = (c.get("snippet") or c.get("text") or "")[:600]
        score = c.get("score", 0)
        ctx_lines.append(f"- id: {cid} | score: {score:.3f} | text: {snip}")
    user_text = (
        f"Locale: {locale}\n"
        f"Question: {question}\n"
        "Top contexts:\n" + "\n".join(ctx_lines)
    )
    return system_text, user_text

_CODEFENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)

def _clean_code_fences(text: str) -> str:
    return _CODEFENCE_RE.sub("", text).strip()

def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        cleaned = _clean_code_fences(text)
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        pass
    return {}

def call_gemini_json(system_text: str, user_text: str, max_retries: int = 2):
    model = genai.GenerativeModel(GEMINI_MODEL)
    retries = 0
    raw, data, usage = None, None, {}

    while retries <= max_retries:
        try:
            resp = model.generate_content(
                [system_text, user_text],
                generation_config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
            )
            raw = (resp.text or "").strip()
            try:
                usage_obj = getattr(resp, "usage_metadata", {}) or {}
                usage = json.loads(json.dumps(usage_obj, default=str))
            except Exception:
                usage = {}
            parsed = _extract_json(raw)
            if parsed and isinstance(parsed, dict) and "answer" in parsed:
                data = parsed
                break
        except Exception:
            time.sleep(0.8)
        retries += 1
    return raw, data, usage, retries
