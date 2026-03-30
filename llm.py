from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any
from urllib import error, request

class LLMError(RuntimeError):
    pass

@dataclass
class GeminiLLM:
    api_key: str
    model: str = "gemini-2.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_seconds: int = 45
    output_log_path: str | None = "llm_output.txt"

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
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
                "temperature": temperature,
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

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise LLMError(self._build_http_error_message(exc.code, raw)) from exc

        parsed = json.loads(raw)
        candidates = parsed.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise LLMError(self._build_invalid_response_message(parsed, "missing candidates"))

        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        content = self._extract_text_from_candidate(candidate)
        if content is None:
            raise LLMError(self._build_invalid_response_message(parsed, "missing text content"))

        self._append_output_log(system_prompt, user_prompt, content)
        return self._parse_json_object(content)

    def _append_output_log(self, system_prompt: str, user_prompt: str, content: str) -> None:
        if not self.output_log_path:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        block = (
            f"[{timestamp}] model={self.model}\n"
            "=== SYSTEM PROMPT ===\n"
            f"{system_prompt}\n"
            "=== USER PROMPT ===\n"
            f"{user_prompt}\n"
            "=== LLM OUTPUT ===\n"
            f"{content}\n"
            "=== END ===\n\n"
        )
        try:
            with open(self.output_log_path, "a", encoding="utf-8") as handle:
                handle.write(block)
        except OSError:
            pass

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            repaired = GeminiLLM._repair_json_text(content)
            if repaired != content:
                try:
                    parsed = json.loads(repaired)
                except json.JSONDecodeError:
                    raise LLMError(f"Gemini returned non-JSON content: {content[:300]}") from exc
            else:
                raise LLMError(f"Gemini returned non-JSON content: {content[:300]}") from exc

        if not isinstance(parsed, dict):
            raise LLMError(f"Gemini returned JSON that was not an object: {type(parsed).__name__}")
        return parsed

    @staticmethod
    def _repair_json_text(content: str) -> str:
        repaired = content.strip()
        repaired = re.sub(r"^```(?:json)?\s*", "", repaired)
        repaired = re.sub(r"\s*```$", "", repaired)
        repaired = re.sub(r'([,{]\s*)""([A-Za-z0-9_]+)"\s*:', r'\1"\2":', repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(
            r'("world_updates"\s*:\s*\{.*?)(,\s*"(?:confidence|feedback|reason|revise|advances_goal|goal_reached)"\s*:)',
            r"\1}\2",
            repaired,
            count=1,
            flags=re.DOTALL,
        )
        repaired = GeminiLLM._balance_json_delimiters(repaired)
        return repaired

    @staticmethod
    def _balance_json_delimiters(content: str) -> str:
        stack: list[str] = []
        in_string = False
        escape = False

        for char in content:
            if in_string:
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif char == "]" and stack and stack[-1] == "[":
                stack.pop()

        closing = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        return content + closing

    @staticmethod
    def _extract_text_from_candidate(candidate: dict[str, Any]) -> str | None:
        content_obj = candidate.get("content", {})
        parts = content_obj.get("parts", []) if isinstance(content_obj, dict) else []
        if not isinstance(parts, list):
            return None

        text_parts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
        if not text_parts:
            return None
        return "".join(text_parts)

    @staticmethod
    def _build_http_error_message(status_code: int, raw: str) -> str:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            snippet = raw[:300].strip()
            return f"Gemini HTTP {status_code}: {snippet or 'empty response body'}"

        return GeminiLLM._build_invalid_response_message(parsed, f"HTTP {status_code}")

    @staticmethod
    def _build_invalid_response_message(parsed: dict[str, Any], prefix: str) -> str:
        prompt_feedback = parsed.get("promptFeedback")
        candidates = parsed.get("candidates")
        candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
        finish_reason = candidate.get("finishReason") if isinstance(candidate, dict) else None
        safety_ratings = candidate.get("safetyRatings") if isinstance(candidate, dict) else None

        details: list[str] = []
        if finish_reason:
            details.append(f"finishReason={finish_reason}")
        if prompt_feedback:
            details.append(f"promptFeedback={json.dumps(prompt_feedback, ensure_ascii=True)}")
        if safety_ratings:
            details.append(f"safetyRatings={json.dumps(safety_ratings, ensure_ascii=True)}")

        detail_suffix = f" ({'; '.join(details)})" if details else ""
        return f"Invalid Gemini response: {prefix}{detail_suffix}."


def build_default_llm() -> GeminiLLM:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is required.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    output_log_path = os.getenv("GEMINI_OUTPUT_LOG_PATH", "llm_output.txt")
    return GeminiLLM(
        api_key=api_key,
        model=model,
        base_url=base_url,
        output_log_path=output_log_path,
    )
