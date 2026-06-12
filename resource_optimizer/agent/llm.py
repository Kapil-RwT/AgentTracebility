"""
Custom LangChain BaseChatModel that wraps the internal Gemini 2.5 Flash
endpoint hosted at genvoy.flipkart.net.

Usage
-----
    from agent.llm import gemini_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = gemini_llm.invoke([
        SystemMessage(content="You are a helpful SRE assistant."),
        HumanMessage(content="How should I set CPU limits?"),
    ])
    print(resp.content)
"""

from __future__ import annotations

from typing import Any

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

# ── Endpoint config ──────────────────────────────────────────────────────────
GEMINI_ENDPOINT = "https://genvoy.flipkart.net/gemini-2.5-flash/:generateContent"
GEMINI_API_KEY = "abebd9ad52dd43d4b6b97c2cc8860e20"

_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


class InternalGeminiFlash(BaseChatModel):
    """LangChain chat model backed by the internal Gemini 2.5 Flash endpoint."""

    endpoint: str = Field(default=GEMINI_ENDPOINT)
    api_key: str = Field(default=GEMINI_API_KEY)
    temperature: float = Field(default=1.0)
    max_output_tokens: int = Field(default=8192)
    thinking_budget: int = Field(default=100)

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "internal-gemini-flash-2.5"

    # ── message conversion ───────────────────────────────────────────────────

    def _to_gemini_contents(self, messages: list[BaseMessage]) -> list:
        """
        Convert LangChain messages to Gemini `contents` format.
        SystemMessage has no direct Gemini role; we inject it as the
        opening user turn so the model sees it as ground-truth context.
        """
        contents: list = []
        system_parts: list = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Collect system text; will be prepended to the first user turn
                system_parts.append(msg.content)
            elif isinstance(msg, HumanMessage):
                text = msg.content
                if system_parts:
                    text = "\n\n".join(system_parts) + "\n\n" + text
                    system_parts = []
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif isinstance(msg, AIMessage):
                contents.append({"role": "model", "parts": [{"text": msg.content}]})

        # Edge-case: only system messages with no human turn
        if system_parts and not contents:
            contents.append(
                {"role": "user", "parts": [{"text": "\n\n".join(system_parts)}]}
            )

        return contents

    # ── core generate ────────────────────────────────────────────────────────

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        contents = self._to_gemini_contents(messages)

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "topP": 1,
                "seed": 0,
                "thinkingConfig": {"thinkingBudget": self.thinking_budget},
            },
            "safetySettings": _SAFETY_SETTINGS,
        }

        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.api_key,
        }

        try:
            resp = requests.post(
                self.endpoint, json=payload, headers=headers, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            text: str = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            text = f"[Gemini API error: {exc}]"

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )


# Singleton used throughout the agent
gemini_llm = InternalGeminiFlash()
