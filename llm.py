from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request

class LLMError(RuntimeError):
    pass

@dataclass
class OpenAICompatLLM:
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 45

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        with request.urlopen(req, timeout=self.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")

        parsed = json.loads(raw)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("Invalid LLM response: missing choices.")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError("Invalid LLM response: missing message content.")

        return json.loads(content)


def build_default_llm() -> OpenAICompatLLM:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is required.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAICompatLLM(api_key=api_key, model=model, base_url=base_url)
