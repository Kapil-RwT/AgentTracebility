"""
LLM client abstraction supporting two backends:
  - AnthropicLLM  : Claude via Anthropic SDK (production)
  - OllamaLLM     : Any OpenAI-compatible endpoint (Ollama / vLLM / LM Studio for local testing)

Select backend via LLM_BACKEND env var: "anthropic" (default) or "ollama".
"""

import json
import os
from typing import Any

import httpx


class LLMResponse:
    def __init__(
        self,
        content: str,
        tool_calls: list[dict] | None = None,
        stop_reason: str = "end_turn",
        _raw: Any = None,
    ):
        self.content = content
        self.tool_calls: list[dict] = tool_calls or []
        self.stop_reason = stop_reason
        # Raw vendor response — needed by AnthropicLLM for multi-turn message history
        self._raw = _raw

    @property
    def wants_tool_call(self) -> bool:
        return bool(self.tool_calls)


class AnthropicLLM:
    """
    Claude via the Anthropic SDK.

    Conversation history format uses raw Anthropic content block objects so that
    multi-turn tool-use works correctly (the SDK requires the original block list
    as the assistant message when continuing a tool_use conversation).
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        import anthropic  # deferred import so the module loads even without the package
        self._sdk = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, messages: list[dict], system: str, tools: list[dict]) -> LLMResponse:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                for t in tools
            ]
        raw = self.client.messages.create(**kwargs)

        tool_calls: list[dict] = []
        text_parts: list[str] = []
        for block in raw.content:
            if block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
            elif block.type == "text":
                text_parts.append(block.text)

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=raw.stop_reason,
            _raw=raw,
        )

    def append_assistant_turn(self, messages: list[dict], response: LLMResponse) -> None:
        """Append assistant message using raw Anthropic content blocks."""
        messages.append({"role": "assistant", "content": response._raw.content})

    def append_tool_results(
        self, messages: list[dict], tool_calls: list[dict], results: list[Any]
    ) -> None:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": json.dumps(result, default=str),
                }
                for tc, result in zip(tool_calls, results)
            ],
        })


class OllamaLLM:
    """
    OpenAI-compatible chat completions (Ollama, vLLM, LM Studio).

    Recommended models with good tool-use support:
      - llama3.1:8b   (fast, good quality)
      - mistral-nemo  (strong at structured output)
      - qwen2.5:7b    (excellent instruction following)

    Start Ollama: `ollama serve` then `ollama pull llama3.1:8b`
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.1:8b",
        api_key: str = "ollama",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _to_openai_tools(self, tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        json_mode: bool = False,
    ) -> LLMResponse:
        all_messages = [{"role": "system", "content": system}, *messages]
        payload: dict = {
            "model": self.model,
            "messages": all_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = self._to_openai_tools(tools)
            payload["tool_choice"] = "required"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        timeout = int(os.getenv("OLLAMA_TIMEOUT_S", "120"))
        with httpx.Client(timeout=timeout) as client:
            r = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()

        data = r.json()
        choice = data["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        tool_calls: list[dict] = []
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                raw_args = fn["arguments"]
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                tool_calls.append({"id": tc["id"], "name": fn["name"], "input": args})

        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else finish_reason,
            _raw=msg,
        )

    def append_assistant_turn(self, messages: list[dict], response: LLMResponse) -> None:
        msg: dict = {"role": "assistant"}
        if response.content:
            msg["content"] = response.content
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])},
                }
                for tc in response.tool_calls
            ]
        # Skip appending if the message has neither content nor tool_calls —
        # Ollama rejects bare {"role": "assistant"} messages with a 400.
        if msg.get("content") or msg.get("tool_calls"):
            messages.append(msg)

    def append_tool_results(
        self, messages: list[dict], tool_calls: list[dict], results: list[Any]
    ) -> None:
        for tc, result in zip(tool_calls, results):
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, default=str),
            })


def create_llm(backend: str | None = None) -> AnthropicLLM | OllamaLLM:
    """
    Factory function. Reads LLM_BACKEND, ANTHROPIC_MODEL, OLLAMA_* env vars.

    LLM_BACKEND=anthropic  → Claude Haiku (fast, cheap, good at tool use)
    LLM_BACKEND=ollama     → local Ollama (free, requires GPU/CPU RAM)
    """
    b = backend or os.getenv("LLM_BACKEND", "anthropic")

    if b == "anthropic":
        return AnthropicLLM(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        )
    if b == "ollama":
        return OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        )
    raise ValueError(f"Unknown LLM_BACKEND '{b}'. Set to 'anthropic' or 'ollama'.")
