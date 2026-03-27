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


def _append_output_log(
    output_log_path: str | None,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    content: str,
) -> None:
    if not output_log_path:
        return

    timestamp = datetime.now(timezone.utc).isoformat()
    block = (
        f"[{timestamp}] model={model_name}\n"
        "=== SYSTEM PROMPT ===\n"
        f"{system_prompt}\n"
        "=== USER PROMPT ===\n"
        f"{user_prompt}\n"
        "=== LLM OUTPUT ===\n"
        f"{content}\n"
        "=== END ===\n\n"
    )
    try:
        with open(output_log_path, "a", encoding="utf-8") as handle:
            handle.write(block)
    except OSError:
        pass

@dataclass
class GeminiLLM:
    api_key: str
    model: str = "gemini-2.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_seconds: int = 45
    output_log_path: str | None = "llm_output.txt"

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
        _append_output_log(self.output_log_path, self.model, system_prompt, user_prompt, content)

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
        return repaired

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
class ActionSFTLLM:
    adapter_path: str = "artifacts/gemma-action-sft"
    base_model: str | None = None
    max_new_tokens: int = 192
    temperature: float = 0.0
    output_log_path: str | None = "llm_output.txt"

    def __post_init__(self) -> None:
        self._model, self._tokenizer, self._resolved_model_name = self._load_model()

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            self._normalize_messages_for_template(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature
        else:
            generation_kwargs["do_sample"] = False

        output_ids = self._model.generate(**inputs, **generation_kwargs)
        prompt_length = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][prompt_length:]
        content = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        _append_output_log(
            self.output_log_path,
            self._resolved_model_name,
            system_prompt,
            user_prompt,
            content,
        )
        return self._parse_json_object(content)

    def _load_model(self) -> tuple[Any, Any, str]:
        try:
            import torch
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise LLMError(
                "Action SFT inference requires `torch`, `transformers`, and `peft`. "
                "Install the updated requirements before running the pipeline."
            ) from exc

        adapter_path = Path(self.adapter_path)
        if not adapter_path.exists():
            raise LLMError(f"Action SFT adapter path does not exist: {adapter_path}")

        resolved_base_model = self.base_model or self._resolve_base_model(adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                str(adapter_path),
                base_model_name_or_path=resolved_base_model,
                **model_kwargs,
            )
        except TypeError:
            model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_path), **model_kwargs)

        if not torch.cuda.is_available():
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                model = model.to("mps")
            else:
                model = model.to("cpu")

        model.eval()
        resolved_model_name = f"action-sft:{adapter_path.as_posix()}"
        return model, tokenizer, resolved_model_name

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
    def _normalize_messages_for_template(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        pending_system_parts: list[str] = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                pending_system_parts.append(content.strip())
                continue

            if role == "user" and pending_system_parts:
                system_prefix = "\n\n".join(
                    f"System instructions:\n{part}" for part in pending_system_parts if part
                ).strip()
                merged_content = f"{system_prefix}\n\n{content}".strip()
                normalized.append({"role": "user", "content": merged_content})
                pending_system_parts = []
                continue

            normalized.append({"role": role, "content": content})

        if pending_system_parts:
            normalized.insert(
                0,
                {
                    "role": "user",
                    "content": "\n\n".join(
                        f"System instructions:\n{part}" for part in pending_system_parts if part
                    ).strip(),
                },
            )

        return normalized

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        parsed = ActionSFTLLM._try_parse_json(content)
        if parsed is not None:
            return parsed

        extracted = ActionSFTLLM._extract_json_object(content)
        if extracted is not None:
            parsed = ActionSFTLLM._try_parse_json(extracted)
            if parsed is not None:
                return parsed

        recovered = ActionSFTLLM._recover_action_payload(content)
        if recovered is not None:
            return recovered

        raise LLMError(f"Action SFT model returned non-JSON content: {content[:300]}")

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
        action = ActionSFTLLM._extract_string_field(content, "action")
        confidence = ActionSFTLLM._extract_numeric_field(content, "confidence")
        rationale = ActionSFTLLM._extract_string_field(content, "rationale")

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
            re.compile(r'"\s*\n\s*\{\s*"confidence"\s*:', flags=re.DOTALL),
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


def build_action_llm() -> ActionSFTLLM:
    adapter_path = os.getenv("ACTION_SFT_MODEL_PATH", "artifacts/gemma-action-sft")
    base_model = os.getenv("ACTION_SFT_BASE_MODEL")
    max_new_tokens = int(os.getenv("ACTION_SFT_MAX_NEW_TOKENS", "192"))
    temperature = float(os.getenv("ACTION_SFT_TEMPERATURE", "0"))
    output_log_path = os.getenv("ACTION_SFT_OUTPUT_LOG_PATH", os.getenv("GEMINI_OUTPUT_LOG_PATH", "llm_output.txt"))
    return ActionSFTLLM(
        adapter_path=adapter_path,
        base_model=base_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_log_path=output_log_path,
    )
