from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request

class LLMError(RuntimeError):
    pass

@dataclass
class GeminiLLM:
    api_key: str
    model: str = "gemini-2.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_seconds: int = 45

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        url = (
            f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
            },
        )

        with request.urlopen(req, timeout=self.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")

        parsed = json.loads(raw)
        candidates = parsed.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise LLMError("Invalid Gemini response: missing candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not isinstance(parts, list) or not parts:
            raise LLMError("Invalid Gemini response: missing content parts.")

        content = parts[0].get("text")
        if not isinstance(content, str):
            raise LLMError("Invalid Gemini response: missing text content.")

        return json.loads(content)


def build_default_llm() -> GeminiLLM:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is required.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    return GeminiLLM(api_key=api_key, model=model, base_url=base_url)
