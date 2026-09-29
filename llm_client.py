"""
llm_client.py
-------------
One common interface for four LLM providers:

  * OpenAI     -> Responses API        (client.responses.create)
  * Gemini     -> google-genai SDK     (client.models.generate_content)
  * Anthropic  -> Messages API         (client.messages.create)
  * Ollama     -> local REST API       (POST {OLLAMA_BASE_URL}/api/generate)

Every provider returns the same `LLMResult`, so prompts can be compared fairly.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------- #
# Common result object
# --------------------------------------------------------------------------- #
@dataclass
class LLMResult:
    provider: str
    model: str
    text: str
    input_tokens: int = 0
    output_tokens: int = 0       # all billed output tokens (includes reasoning)
    reasoning_tokens: int = 0    # subset of output_tokens spent on "thinking"
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated: bool = False

    def cost(self, input_price_per_1m: Optional[float],
             output_price_per_1m: Optional[float]) -> Optional[float]:
        """Estimated USD cost. Returns None if prices are not configured."""
        if input_price_per_1m is None or output_price_per_1m is None:
            return None
        return (self.input_tokens * input_price_per_1m
                + self.output_tokens * output_price_per_1m) / 1_000_000


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BaseLLM(ABC):
    name = "base"

    def __init__(self, model: str, api_key: Optional[str] = None,
                 max_output_tokens: int = 1024,
                 temperature: Optional[float] = None):
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResult:
        """Call the model and measure wall-clock latency."""
        start = time.perf_counter()
        result = self._call(prompt, system)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    @abstractmethod
    def _call(self, prompt: str, system: Optional[str]) -> LLMResult:
        ...


# --------------------------------------------------------------------------- #
# OpenAI - Responses API
# --------------------------------------------------------------------------- #
class OpenAILLM(BaseLLM):
    name = "openai"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        # Optional, only for reasoning models (e.g. "minimal", "low", "medium")
        self.reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip() or None

    def _call(self, prompt, system):
        params = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
        }
        if system:
            params["instructions"] = system
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}

        resp = self.client.responses.create(**params)
        usage = resp.usage
        details = getattr(usage, "output_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0

        return LLMResult(
            provider=self.name,
            model=self.model,
            text=(resp.output_text or "").strip(),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=reasoning,
            total_tokens=usage.total_tokens,
        )


# --------------------------------------------------------------------------- #
# Google Gemini - google-genai SDK
# --------------------------------------------------------------------------- #
class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=self.api_key)

    def _call(self, prompt, system):
        config = self._types.GenerateContentConfig(
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            system_instruction=system,
        )
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        meta = resp.usage_metadata
        prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
        answer_tokens = getattr(meta, "candidates_token_count", 0) or 0
        thought_tokens = getattr(meta, "thoughts_token_count", 0) or 0

        return LLMResult(
            provider=self.name,
            model=self.model,
            text=(resp.text or "").strip(),
            input_tokens=prompt_tokens,
            # Gemini reports thinking tokens separately; both are billed as output
            output_tokens=answer_tokens + thought_tokens,
            reasoning_tokens=thought_tokens,
            total_tokens=getattr(meta, "total_token_count", 0)
            or prompt_tokens + answer_tokens + thought_tokens,
        )


# --------------------------------------------------------------------------- #
# Anthropic - Messages API
# --------------------------------------------------------------------------- #
class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _call(self, prompt, system):
        params = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            params["system"] = system
        if self.temperature is not None:
            params["temperature"] = self.temperature

        resp = self.client.messages.create(**params)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        usage = resp.usage

        return LLMResult(
            provider=self.name,
            model=self.model,
            text=text.strip(),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )


# --------------------------------------------------------------------------- #
# Ollama - local REST API (no API key; server runs on your own machine)
# --------------------------------------------------------------------------- #
class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(**kwargs)
        import requests
        self._requests = requests
        self.base_url = base_url.rstrip("/")

    def _call(self, prompt, system):
        options = {"num_predict": self.max_output_tokens}
        if self.temperature is not None:
            options["temperature"] = self.temperature

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        try:
            resp = self._requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=120
            )
            resp.raise_for_status()
        except self._requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self.base_url}. "
                f"Is 'ollama serve' running and is OLLAMA_BASE_URL correct? ({e})"
            ) from e

        data = resp.json()
        in_tok = data.get("prompt_eval_count", 0) or 0
        out_tok = data.get("eval_count", 0) or 0

        return LLMResult(
            provider=self.name,
            model=self.model,
            text=(data.get("response") or "").strip(),
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
PROVIDERS = {
    "openai": OpenAILLM,
    "gemini": GeminiLLM,
    "anthropic": AnthropicLLM,
    "ollama": OllamaLLM,
}


def create_llm(provider: str) -> BaseLLM:
    """Build a client from .env settings, e.g. OPENAI_MODEL / OPENAI_API_KEY."""
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDERS)}")

    prefix = provider.upper()
    api_key = os.getenv(f"{prefix}_API_KEY", "").strip() or None
    model = os.getenv(f"{prefix}_MODEL", "").strip()

    if not model:
        raise ValueError(f"{prefix}_MODEL is missing in .env")
    if provider != "ollama" and not api_key:
        raise ValueError(f"{prefix}_API_KEY is missing in .env")

    temp_raw = os.getenv("TEMPERATURE", "").strip()
    kwargs = dict(
        model=model,
        api_key=api_key,
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "1024")),
        temperature=float(temp_raw) if temp_raw else None,
    )
    if provider == "ollama":
        kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL", "").strip() or "http://localhost:11434"

    return PROVIDERS[provider](**kwargs)
