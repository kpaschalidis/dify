"""Keep runtime callback credentials out of model history and summaries.

These are explicit runtime credential argument names, not a keyword detector for
business data. Tool execution keeps the original arguments. Only history copies
are redacted, including old tool calls restored from a previous run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, TypeVar, cast

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext

_CREDENTIAL_FIELDS = frozenset({"runToken", "run_token", "harness_run_token"})
_REDACTED = "[REDACTED]"
_Part = TypeVar("_Part", bound=ModelRequestPart | ModelResponsePart)


class CredentialHistoryRedaction(Capability[None]):
    """Per-run secret discovery shared by pre-compaction and persistence paths."""

    def __init__(self) -> None:
        super().__init__()
        self._values: set[str] = set()

    def _discover(self, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _CREDENTIAL_FIELDS and isinstance(item, str) and item and item != _REDACTED:
                    self._values.add(item)
                else:
                    self._discover(item)
        elif isinstance(value, list):
            for item in value:
                self._discover(item)
        elif isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (ValueError, TypeError):
                return
            if isinstance(decoded, (dict, list)):
                self._discover(decoded)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _REDACTED if key in _CREDENTIAL_FIELDS else self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except ValueError:
                decoded = None
            if isinstance(decoded, (dict, list)):
                value = json.dumps(self._redact(decoded))
            for secret in sorted(self._values, key=len, reverse=True):
                value = value.replace(secret, _REDACTED)
            return value
        return value

    @staticmethod
    def _arguments(part: ToolCallPart) -> Any:
        if not isinstance(part.args, str):
            return part.args
        try:
            decoded = json.loads(part.args)
            if isinstance(decoded, dict):
                return decoded
        except ValueError:
            pass
        # Tool arguments must be JSON objects. Invalid/partial/scalar arguments
        # must not persist unparseable secrets; result/error and call IDs remain.
        return {"__redacted_invalid_arguments__": True}

    def redact(self, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        for message in messages:
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    self._discover(self._arguments(part))

        cleaned: list[ModelMessage] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                cleaned.append(replace(message, parts=[self._redact_part(part) for part in message.parts]))
            else:
                cleaned.append(replace(message, parts=[self._redact_part(part) for part in message.parts]))
        return cleaned

    def _redact_part(self, part: _Part) -> _Part:
        if isinstance(part, ToolCallPart):
            args = self._redact(self._arguments(part))
            return cast(_Part, replace(part, args=json.dumps(args) if isinstance(part.args, str) else args))
        if isinstance(
            part, (UserPromptPart, SystemPromptPart, TextPart, ThinkingPart, ToolReturnPart, RetryPromptPart)
        ):
            return cast(_Part, replace(part, content=self._redact(part.content)))
        return part

    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        request_context.messages = self.redact(request_context.messages)
        return request_context
