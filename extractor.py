"""
LLM extraction interface with constrained output + repair loop.
Supports two backends:
  - "ollama": fully local, calls a running Ollama server
  - "mistral_api": Mistral's cloud API, requires an API key

Both paths go through the same schema validation + repair loop.

Note on Qwen3 and other hybrid "thinking" models: by default they emit a
long internal reasoning block wrapped in <think>...</think> before the
real answer. This is slow and can break JSON parsing if the reasoning
text leaks into the output. We disable thinking mode via Ollama's
"think": false option and force strict JSON output via "format": "json",
plus strip any stray <think> block as a safety net.
"""
import json
import re
import requests
from pydantic import ValidationError
from schema import InvoiceExtraction

MAX_REPAIR_ATTEMPTS = 3
OLLAMA_TIMEOUT_SECONDS = 120
MISTRAL_TIMEOUT_SECONDS = 60

OLLAMA_URL = "http://localhost:11434/api/chat"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

SYSTEM_PROMPT = """You are an invoice data extraction assistant. Extract structured data from raw OCR text.
Respond with ONLY a JSON object matching this exact schema, no other text, no markdown fences, no explanation:
{
  "vendor_name": string,
  "invoice_date": string or null (YYYY-MM-DD format),
  "total_amount": number,
  "tax_amount": number,
  "category": one of ["Office", "IT", "Travel", "Food", "Utilities", "Other"],
  "description": string,
  "confidence": number between 0 and 1 representing your certainty
}
If a field cannot be determined, use null (or 0.0 for tax_amount if genuinely absent). Do not invent data."""


class ExtractionError(Exception):
    pass


def _call_ollama(messages: list, model: str = "mistral") -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",       # forces strict JSON output, no markdown/prose wrapping
            "think": False,         # disables reasoning mode on Qwen3 and similar hybrid models
            "options": {"temperature": 0.1, "num_predict": 512},
        },
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        raise ExtractionError(f"Ollama request failed ({resp.status_code}): {resp.text}")
    return resp.json()["message"]["content"]


def _call_mistral_api(messages: list, api_key: str, model: str = "mistral-small-latest") -> str:
    resp = requests.post(
        MISTRAL_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.1, "response_format": {"type": "json_object"}},
        timeout=MISTRAL_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        raise ExtractionError(f"Mistral API request failed ({resp.status_code}): {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


def _extract_json_block(text: str) -> str:
    """Strip any leaked <think> block and isolate the JSON object.
    This is formatting cleanup only, not field-level extraction."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def extract_fields(
    raw_text: str,
    backend: str = "ollama",
    ollama_model: str = "mistral",
    mistral_api_key: str = None,
    mistral_model: str = "mistral-small-latest",
) -> tuple[InvoiceExtraction | None, list[dict]]:
    """
    Returns (validated_extraction_or_None, attempt_log).
    attempt_log feeds directly into your report's failure taxonomy.
    """
    attempt_log = []
    repair_context = ""

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        user_content = f"OCR text:\n{raw_text}"
        if repair_context:
            user_content += f"\n\nYour previous output failed validation: {repair_context}\nFix it and return only valid JSON."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            if backend == "ollama":
                raw_output = _call_ollama(messages, model=ollama_model)
            elif backend == "mistral_api":
                if not mistral_api_key:
                    raise ExtractionError("Mistral API key not provided")
                raw_output = _call_mistral_api(messages, api_key=mistral_api_key, model=mistral_model)
            else:
                raise ExtractionError(f"Unknown backend: {backend}")
        except requests.exceptions.Timeout:
            attempt_log.append({"attempt": attempt, "status": "timeout",
                                 "error": f"Request exceeded {OLLAMA_TIMEOUT_SECONDS if backend == 'ollama' else MISTRAL_TIMEOUT_SECONDS}s"})
            return None, attempt_log
        except requests.exceptions.ConnectionError:
            attempt_log.append({"attempt": attempt, "status": "connection_error",
                                 "error": "Could not reach the backend. Is Ollama running (ollama serve)?"})
            return None, attempt_log
        except ExtractionError as e:
            attempt_log.append({"attempt": attempt, "status": "backend_error", "error": str(e)})
            return None, attempt_log

        json_str = _extract_json_block(raw_output)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            attempt_log.append({"attempt": attempt, "status": "json_decode_error", "error": str(e)})
            repair_context = f"Output was not valid JSON: {e}"
            continue

        try:
            validated = InvoiceExtraction(**data)
            attempt_log.append({"attempt": attempt, "status": "success"})
            return validated, attempt_log
        except ValidationError as e:
            attempt_log.append({"attempt": attempt, "status": "schema_invalid", "error": str(e)})
            repair_context = str(e)
            continue

    return None, attempt_log


def check_ollama_available() -> tuple[bool, list[str]]:
    """Ping local Ollama server and return (is_up, list_of_models)."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return True, models
    except requests.exceptions.RequestException:
        pass
    return False, []