from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

class LLMError(RuntimeError):
    pass


def _parse_json_object_with_repair(content: str, provider_name: str) -> dict[str, Any]:
    parse_error: json.JSONDecodeError | None = None
    candidates = [content]
    extracted = _extract_first_json_object(content)
    if extracted is not None and extracted != content:
        candidates.append(extracted)

    for candidate in candidates:
        for parse_candidate in (candidate, GeminiLLM._repair_json_text(candidate)):
            try:
                parsed = json.loads(parse_candidate)
                break
            except json.JSONDecodeError as exc:
                parse_error = exc
        else:
            continue
        break
    else:
        raise LLMError(f"{provider_name} returned non-JSON content: {content[:300]}") from parse_error

    if not isinstance(parsed, dict):
        raise LLMError(f"{provider_name} returned JSON that was not an object: {type(parsed).__name__}")
    return parsed


def _extract_first_json_object(content: str) -> str | None:
    start = content.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]

    return None

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
        return _parse_json_object_with_repair(content, "Gemini")

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


@dataclass
class AnthropicLLM:
    api_key: str
    model: str = "claude-opus-4-1"
    base_url: str = "https://api.anthropic.com/v1"
    timeout_seconds: int = 45
    output_log_path: str | None = "llm_output.txt"
    max_output_tokens: int = 2048
    anthropic_version: str = "2023-06-01"

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/messages"
        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise LLMError(self._build_http_error_message(exc.code, raw)) from exc

        parsed = json.loads(raw)
        content = self._extract_text(parsed)
        if content is None:
            raise LLMError(self._build_invalid_response_message(parsed, "missing text content"))

        self._append_output_log(system_prompt, user_prompt, content)
        return _parse_json_object_with_repair(content, "Anthropic")

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
    def _extract_text(parsed: dict[str, Any]) -> str | None:
        content_list = parsed.get("content")
        if not isinstance(content_list, list):
            return None
        text_parts: list[str] = []
        for part in content_list:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "text":
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
            return f"Anthropic HTTP {status_code}: {snippet or 'empty response body'}"
        return AnthropicLLM._build_invalid_response_message(parsed, f"HTTP {status_code}")

    @staticmethod
    def _build_invalid_response_message(parsed: dict[str, Any], prefix: str) -> str:
        details: list[str] = []
        stop_reason = parsed.get("stop_reason")
        stop_type = parsed.get("type")
        error_obj = parsed.get("error")
        if stop_reason:
            details.append(f"stop_reason={stop_reason}")
        if stop_type:
            details.append(f"type={stop_type}")
        if error_obj:
            details.append(f"error={json.dumps(error_obj, ensure_ascii=True)}")
        detail_suffix = f" ({'; '.join(details)})" if details else ""
        return f"Invalid Anthropic response: {prefix}{detail_suffix}."


@dataclass
class LocalTransformersLLM:
    model_name_or_path: str
    max_new_tokens: int = 1024
    temperature: float = 0.1
    load_in_4bit: bool = False
    output_log_path: str | None = "llm_output.txt"

    def __post_init__(self) -> None:
        self._model, self._tokenizer = self._load_model()

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        prompt_text = self._build_prompt_text(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt + "\nReturn a single compact JSON object only.",
                },
            ]
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generation_kwargs = self._build_generation_kwargs(temperature=temperature)
        output_ids = self._model.generate(**inputs, **generation_kwargs)
        prompt_length = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][prompt_length:]
        content = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        self._append_output_log(system_prompt, user_prompt, content)
        return _parse_json_object_with_repair(content, "Local transformers")

    def _load_model(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise LLMError(
                "Local transformers inference requires `torch` and `transformers`."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        if not torch.cuda.is_available():
            model = model.to("cpu")
        model.eval()
        return model, tokenizer

    def _build_generation_kwargs(self, temperature: float) -> dict[str, Any]:
        resolved_temperature = self.temperature if self.temperature is not None else temperature
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if resolved_temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = resolved_temperature
            generation_kwargs["top_p"] = 0.9
        else:
            generation_kwargs["do_sample"] = False
        return generation_kwargs

    def _append_output_log(self, system_prompt: str, user_prompt: str, content: str) -> None:
        if not self.output_log_path:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        block = (
            f"[{timestamp}] model=local:{self.model_name_or_path}\n"
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

    def _build_prompt_text(self, messages: list[dict[str, str]]) -> str:
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return ActionAdapterLLM._normalize_messages_for_template(messages)


def _normalize_provider_name(raw_provider: str | None) -> str:
    provider = (raw_provider or "").strip().lower()
    if provider in {"", "gemini", "google"}:
        return "gemini"
    if provider in {"anthropic", "claude", "opus"}:
        return "anthropic"
    if provider in {"local", "transformers", "huggingface", "hf"}:
        return "local"
    raise LLMError(f"Unsupported LLM provider: {raw_provider}")


def _build_hosted_llm(
    provider: str,
    model: str | None = None,
    output_log_path: str | None = None,
) -> GeminiLLM | AnthropicLLM | LocalTransformersLLM:
    normalized_provider = _normalize_provider_name(provider)

    if normalized_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY is required for Gemini provider.")
        resolved_model = model or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        resolved_output_log_path = output_log_path or os.getenv("GEMINI_OUTPUT_LOG_PATH", "llm_output.txt")
        return GeminiLLM(
            api_key=api_key,
            model=resolved_model,
            base_url=base_url,
            output_log_path=resolved_output_log_path,
        )

    if normalized_provider == "local":
        resolved_model = model or os.getenv("LLM_MODEL") or os.getenv("LOCAL_LLM_MODEL")
        if not resolved_model:
            raise LLMError("LOCAL_LLM_MODEL or a role-specific local model is required.")
        max_new_tokens = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "1024"))
        temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1"))
        load_in_4bit = os.getenv("LOCAL_LLM_LOAD_IN_4BIT", "0").strip().lower() in {"1", "true", "yes"}
        resolved_output_log_path = output_log_path or os.getenv("GEMINI_OUTPUT_LOG_PATH", "llm_output.txt")
        return LocalTransformersLLM(
            model_name_or_path=resolved_model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            load_in_4bit=load_in_4bit,
            output_log_path=resolved_output_log_path,
        )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY is required for Anthropic provider.")
    resolved_model = model or os.getenv("LLM_MODEL") or os.getenv("ANTHROPIC_MODEL", "claude-opus-4-1")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
    resolved_output_log_path = output_log_path or os.getenv(
        "ANTHROPIC_OUTPUT_LOG_PATH",
        os.getenv("GEMINI_OUTPUT_LOG_PATH", "llm_output.txt"),
    )
    max_output_tokens = int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "2048"))
    anthropic_version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
    return AnthropicLLM(
        api_key=api_key,
        model=resolved_model,
        base_url=base_url,
        output_log_path=resolved_output_log_path,
        max_output_tokens=max_output_tokens,
        anthropic_version=anthropic_version,
    )


def build_default_llm() -> GeminiLLM | AnthropicLLM | LocalTransformersLLM:
    provider = os.getenv("LLM_PROVIDER", "gemini")
    return _build_hosted_llm(provider=provider)


def build_critic_llm(default_llm: object | None = None) -> object:
    provider = _normalize_provider_name(os.getenv("CRITIC_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "gemini")))
    model = os.getenv("CRITIC_LLM_MODEL") or None
    output_log_path = os.getenv(
        "CRITIC_MODEL_OUTPUT_LOG_PATH",
        os.getenv("GEMINI_OUTPUT_LOG_PATH", "llm_output.txt"),
    )
    if (
        default_llm is not None
        and provider == _normalize_provider_name(os.getenv("LLM_PROVIDER", "gemini"))
        and not model
    ):
        return default_llm
    return _build_hosted_llm(
        provider=provider,
        model=model,
        output_log_path=output_log_path,
    )


def build_narrator_llm(default_llm: object | None = None) -> object:
    provider = _normalize_provider_name(os.getenv("NARRATOR_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "gemini")))
    model = os.getenv("NARRATOR_LLM_MODEL") or None
    output_log_path = os.getenv(
        "NARRATOR_MODEL_OUTPUT_LOG_PATH",
        os.getenv("GEMINI_OUTPUT_LOG_PATH", "llm_output.txt"),
    )
    if (
        default_llm is not None
        and provider == _normalize_provider_name(os.getenv("LLM_PROVIDER", "gemini"))
        and not model
    ):
        return default_llm
    return _build_hosted_llm(
        provider=provider,
        model=model,
        output_log_path=output_log_path,
    )


@dataclass
class ActionAdapterLLM:
    adapter_path: str = "artifacts/gemma-action-dpo"
    base_model: str | None = None
    max_new_tokens: int = 96
    temperature: float = 0.0
    load_in_4bit: bool = False
    output_log_path: str | None = "llm_output.txt"

    def __post_init__(self) -> None:
        self._model, self._tokenizer, self._resolved_model_name = self._load_model()

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._build_prompt_text(messages)
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        strict_prompt_text = self._build_prompt_text(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt
                    + "\nReturn a single compact JSON object only. "
                    + "Keep action under 30 words and rationale under 12 words.",
                },
            ]
        )
        strict_inputs = self._tokenizer(strict_prompt_text, return_tensors="pt")
        if model_device is not None:
            strict_inputs = {key: value.to(model_device) for key, value in strict_inputs.items()}

        generation_kwargs = self._build_generation_kwargs(
            temperature=self.temperature,
            max_new_tokens=min(self.max_new_tokens, 72),
        )
        output_ids = self._model.generate(**strict_inputs, **generation_kwargs)
        prompt_length = int(strict_inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][prompt_length:]
        content = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        content = self._sanitize_generated_content(content)
        self._append_output_log(system_prompt, user_prompt, content)

        try:
            payload = self._parse_json_object(content)
            return self._normalize_action_payload(payload)
        except LLMError:
            pass

        recovered = self._recover_action_payload(content)
        if recovered is not None:
            return self._normalize_action_payload(recovered)

        retry_content = self._generate_action_content(
            system_prompt=system_prompt,
            user_prompt=(
                user_prompt
                + "\nThe previous answer was invalid because it included prose instead of JSON. "
                + "Return exactly one JSON object with keys action, confidence, and rationale. "
                + "Do not include Thought, markdown, explanation, or any text outside the JSON object."
            ),
            temperature=0.0,
            max_new_tokens=min(self.max_new_tokens, 48),
        )
        retry_content = self._sanitize_generated_content(retry_content)
        self._append_output_log(system_prompt, user_prompt, retry_content)
        try:
            payload = self._parse_json_object(retry_content)
            return self._normalize_action_payload(payload)
        except LLMError:
            pass

        recovered = self._recover_action_payload(retry_content)
        if recovered is not None:
            return self._normalize_action_payload(recovered)
        return self._normalize_action_payload(
            {
                "action": "look around for a concrete next step",
                "confidence": 0.1,
                "rationale": "adapter emitted non-json",
            }
        )

    def _generate_action_content(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        strict_prompt_text = self._build_prompt_text(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt
                    + "\nReturn a single compact JSON object only. "
                    + "Keep action under 30 words and rationale under 12 words.",
                },
            ]
        )
        strict_inputs = self._tokenizer(strict_prompt_text, return_tensors="pt")
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            strict_inputs = {key: value.to(model_device) for key, value in strict_inputs.items()}

        generation_kwargs = self._build_generation_kwargs(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        output_ids = self._model.generate(**strict_inputs, **generation_kwargs)
        prompt_length = int(strict_inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][prompt_length:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _append_output_log(self, system_prompt: str, user_prompt: str, content: str) -> None:
        if not self.output_log_path:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        block = (
            f"[{timestamp}] model={self._resolved_model_name}\n"
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

    def _load_model(self) -> tuple[Any, Any, str]:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise LLMError(
                "Action adapter inference requires `torch`, `transformers`, and `peft`."
            ) from exc

        adapter_path = self._resolve_adapter_path(Path(self.adapter_path))
        if not adapter_path.exists():
            raise LLMError(f"Action adapter path does not exist: {adapter_path}")

        resolved_base_model = self.base_model or self._resolve_base_model(adapter_path)
        tokenizer_source = resolved_base_model or str(adapter_path)
        tokenizer_kwargs: dict[str, Any] = {}
        if resolved_base_model and "mistral" in resolved_base_model.lower():
            tokenizer_kwargs["fix_mistral_regex"] = True
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        except AttributeError:
            tokenizer_kwargs.pop("fix_mistral_regex", None)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}
        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if not resolved_base_model:
            raise LLMError(
                f"Could not determine the base model for action adapter at {adapter_path}."
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_base_model,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
        )

        if not torch.cuda.is_available():
            model = model.to("cpu")

        model.eval()
        resolved_model_name = f"action-dpo:{adapter_path.as_posix()}"
        return model, tokenizer, resolved_model_name

    def _build_generation_kwargs(self, temperature: float, max_new_tokens: int) -> dict[str, Any]:
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
        }
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.9
        else:
            generation_kwargs["do_sample"] = False
        return generation_kwargs

    @staticmethod
    def _resolve_base_model(adapter_path: Path) -> str | None:
        config_path = adapter_path / "adapter_config.json"
        if not config_path.exists():
            return None
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        base_model = parsed.get("base_model_name_or_path")
        return str(base_model) if isinstance(base_model, str) and base_model.strip() else None

    @staticmethod
    def _resolve_adapter_path(adapter_path: Path) -> Path:
        if (adapter_path / "adapter_config.json").exists():
            return adapter_path

        candidates = [
            child.parent
            for child in adapter_path.rglob("adapter_config.json")
            if not any(part.lower().startswith("checkpoint-") for part in child.parts)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return adapter_path

    def _build_prompt_text(self, messages: list[dict[str, str]]) -> str:
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return self._normalize_messages_for_template(messages)

    @staticmethod
    def _normalize_messages_for_template(messages: list[dict[str, str]]) -> str:
        system_parts = [m["content"].strip() for m in messages if m["role"] == "system" and m["content"].strip()]
        user_parts = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
        prompt_parts: list[str] = []
        if system_parts:
            prompt_parts.append("System instructions:\n" + "\n\n".join(system_parts))
        if user_parts:
            prompt_parts.append("User request:\n" + "\n\n".join(user_parts))
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        parsed = ActionAdapterLLM._try_parse_json(content)
        if parsed is not None:
            return parsed

        extracted = ActionAdapterLLM._extract_json_object(content)
        if extracted is not None:
            parsed = ActionAdapterLLM._try_parse_json(extracted)
            if parsed is not None:
                return parsed

        recovered = ActionAdapterLLM._recover_action_payload(content)
        if recovered is not None:
            return recovered

        raise LLMError(f"Action adapter returned non-JSON content: {content[:300]}")

    @staticmethod
    def _sanitize_generated_content(content: str) -> str:
        cleaned = content.strip()
        if not cleaned:
            return cleaned
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        repeated_token_pattern = re.compile(r"\b([A-Za-z]{3,})\b(?:\s+\1\b){5,}", flags=re.IGNORECASE)
        cleaned = repeated_token_pattern.sub(r"\1", cleaned)

        json_object = ActionAdapterLLM._extract_json_object(cleaned)
        if json_object is not None:
            cleaned = json_object
        return cleaned.strip()

    @staticmethod
    def _normalize_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        confidence = payload.get("confidence", 0.5)

        action = ActionAdapterLLM._collapse_repetition(action)
        rationale = ActionAdapterLLM._collapse_repetition(rationale)
        action = ActionAdapterLLM._truncate_words(action, 30) or "look around"
        rationale = ActionAdapterLLM._truncate_words(rationale, 12)
        confidence_value = ActionAdapterLLM._coerce_confidence(confidence)

        return {
            "action": action,
            "confidence": confidence_value,
            "rationale": rationale,
        }

    @staticmethod
    def _collapse_repetition(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(
            r"\b([A-Za-z]{3,})\b(?:\s+\1\b){2,}",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip(" ,")

    @staticmethod
    def _truncate_words(text: str, limit: int) -> str:
        words = text.split()
        if len(words) <= limit:
            return text.strip()
        return " ".join(words[:limit]).rstrip(" ,.;:") + "."

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.5
        return max(0.0, min(1.0, parsed))

    @staticmethod
    def _try_parse_json(content: str) -> dict[str, Any] | None:
        repaired = GeminiLLM._repair_json_text(content)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _extract_json_object(content: str) -> str | None:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return content[start : end + 1]

    @staticmethod
    def _recover_action_payload(content: str) -> dict[str, Any] | None:
        action = ActionAdapterLLM._extract_string_field(content, "action")
        confidence = ActionAdapterLLM._extract_numeric_field(content, "confidence")
        rationale = ActionAdapterLLM._extract_string_field(content, "rationale")

        if action is None and rationale is None:
            return None

        payload: dict[str, Any] = {}
        if action is not None:
            payload["action"] = action
        if confidence is not None:
            payload["confidence"] = confidence
        if rationale is not None:
            payload["rationale"] = rationale
        return payload or None

    @staticmethod
    def _extract_string_field(content: str, field_name: str) -> str | None:
        field_pattern = re.compile(rf'"{re.escape(field_name)}"\s*:\s*"', flags=re.DOTALL)
        match = field_pattern.search(content)
        if not match:
            return None

        start = match.end()
        terminator_patterns = [
            re.compile(r'"\s*,\s*"confidence"\s*:', flags=re.DOTALL),
            re.compile(r'"\s*,\s*"rationale"\s*:', flags=re.DOTALL),
            re.compile(r'"\s*}\s*(?:\n|$)', flags=re.DOTALL),
            re.compile(r'"\s*\n\s*}\s*(?:\n|$)', flags=re.DOTALL),
        ]

        end = -1
        for pattern in terminator_patterns:
            terminator_match = pattern.search(content, start)
            if terminator_match is None:
                continue
            candidate_end = terminator_match.start()
            if end == -1 or candidate_end < end:
                end = candidate_end

        value = content[start:end].strip() if end != -1 else content[start:].strip()
        value = re.sub(r"[}\]]+\s*$", "", value).strip()
        value = value.replace('\\"', '"')
        value = re.sub(r"\s+", " ", value).strip()
        return value or None

    @staticmethod
    def _extract_numeric_field(content: str, field_name: str) -> float | None:
        match = re.search(
            rf'"{re.escape(field_name)}"\s*:\s*(-?\d+(?:\.\d+)?)',
            content,
        )
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None


def build_action_llm(default_llm: object | None = None) -> object:
    adapter_path = os.getenv("ACTION_MODEL_PATH", "").strip()
    if not adapter_path:
        provider = _normalize_provider_name(os.getenv("ACTION_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "gemini")))
        model = os.getenv("ACTION_LLM_MODEL") or None
        output_log_path = os.getenv(
            "ACTION_MODEL_OUTPUT_LOG_PATH",
            os.getenv("GEMINI_OUTPUT_LOG_PATH", "llm_output.txt"),
        )
        if default_llm is not None and provider == _normalize_provider_name(os.getenv("LLM_PROVIDER", "gemini")) and not model:
            return default_llm
        return _build_hosted_llm(
            provider=provider,
            model=model,
            output_log_path=output_log_path,
        )

    base_model = os.getenv("ACTION_MODEL_BASE")
    max_new_tokens = int(os.getenv("ACTION_MODEL_MAX_NEW_TOKENS", "96"))
    temperature = float(os.getenv("ACTION_MODEL_TEMPERATURE", "0.3"))
    load_in_4bit = os.getenv("ACTION_MODEL_LOAD_IN_4BIT", "0").strip().lower() in {"1", "true", "yes"}
    output_log_path = os.getenv(
        "ACTION_MODEL_OUTPUT_LOG_PATH",
        os.getenv("GEMINI_OUTPUT_LOG_PATH", "llm_output.txt"),
    )
    return ActionAdapterLLM(
        adapter_path=adapter_path,
        base_model=base_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        load_in_4bit=load_in_4bit,
        output_log_path=output_log_path,
    )
