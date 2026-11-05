import json
from dataclasses import replace

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from dify_agent.runtime.compaction import build_compaction_capability
from dify_agent.runtime.credential_history import CredentialHistoryRedaction
from dify_agent.runtime.history import replace_run_history

SECRET = "fixture-private-runtime-credential-012345"


@pytest.mark.parametrize("args", [json.dumps(SECRET), json.dumps([SECRET]), '{"run_token":"' + SECRET])
def test_invalid_argument_shapes_cannot_persist_bare_credentials(args):
    cleaned = CredentialHistoryRedaction().redact([ModelResponse(parts=[ToolCallPart("callback", args, "invalid")])])
    assert SECRET not in ModelMessagesTypeAdapter.dump_json(cleaned).decode()
    assert cleaned[0].parts[0].tool_call_id == "invalid"


@pytest.mark.parametrize("field", ["runToken", "run_token", "harness_run_token"])
@pytest.mark.parametrize("encoded", [False, True])
def test_restored_credentials_and_echoes_are_removed_without_mutating_original_calls(field, encoded):
    args = {"nested": [{field: SECRET}], "taskId": "keep-task", "token": "business-reference"}
    history = [
        ModelResponse(parts=[ToolCallPart("read_context", json.dumps(args) if encoded else args, "call-1")]),
        ModelRequest(parts=[ToolReturnPart("read_context", {"message": f"Callback {SECRET} failed"}, "call-1")]),
        ModelResponse(parts=[TextPart(f"Earlier summary copied {SECRET}")]),
    ]

    cleaned = CredentialHistoryRedaction().redact(history)

    assert SECRET not in ModelMessagesTypeAdapter.dump_json(cleaned).decode()
    assert SECRET in ModelMessagesTypeAdapter.dump_json(history).decode()
    call = cleaned[0].parts[0]
    assert isinstance(call, ToolCallPart)
    assert call.tool_call_id == "call-1"
    assert call.args_as_dict() == {
        "nested": [{field: "[REDACTED]"}],
        "taskId": "keep-task",
        "token": "business-reference",
    }
    assert cleaned[1].parts[0].tool_call_id == "call-1"


def test_nested_json_arguments_are_discovered_and_partial_arguments_fail_closed():
    history = [
        ModelResponse(
            parts=[
                ToolCallPart("callback", {"request": json.dumps({"run_token": SECRET, "value": 7})}, "nested"),
                ToolCallPart("callback", '{"run_token":"partial-credential', "partial"),
            ]
        )
    ]
    cleaned = CredentialHistoryRedaction().redact(history)
    encoded = ModelMessagesTypeAdapter.dump_json(cleaned).decode()
    assert SECRET not in encoded
    assert "partial-credential" not in encoded
    assert cleaned[0].parts[1].tool_call_id == "partial"
    assert json.loads(cleaned[0].parts[0].args["request"])["value"] == 7


def test_embedded_json_redacts_escaped_credential_values():
    secret = 'credential-with-"quoted"-characters'
    history = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    "callback",
                    {"request": json.dumps({"run_token": secret, "taskId": "keep-task"})},
                    "nested",
                )
            ]
        )
    ]
    cleaned = CredentialHistoryRedaction().redact(history)
    args = json.loads(cleaned[0].parts[0].args["request"])
    assert args == {"run_token": "[REDACTED]", "taskId": "keep-task"}


def test_authorized_invocation_uses_original_credential_but_next_model_request_and_saved_history_do_not():
    invoked = []
    inspected = []

    def callback(runToken: str, task_id: str) -> str:
        invoked.append((runToken, task_id))
        return f"Read succeeded for {runToken}"

    def respond(messages, info):
        if not invoked:
            return ModelResponse(parts=[ToolCallPart("callback", {"runToken": SECRET, "task_id": "task-1"})])
        inspected.extend(messages)
        return ModelResponse(parts=[TextPart("Done")])

    redaction = CredentialHistoryRedaction()
    result = Agent(FunctionModel(respond), tools=[callback]).run_sync(
        "Read the synthetic context.",
        capabilities=[redaction],
        instructions=f"Transient credential: {SECRET}",
    )
    assert invoked == [(SECRET, "task-1")]
    # Current instructions remain usable by the live execution; they are not persisted.
    history_only = [replace(m, instructions=None) if isinstance(m, ModelRequest) else m for m in inspected]
    assert SECRET not in ModelMessagesTypeAdapter.dump_json(history_only).decode()

    class Sink:
        def replace_messages(self, messages):
            self.messages = messages

    sink = Sink()
    replace_run_history(sink, result.all_messages(), redaction=redaction)
    assert SECRET not in ModelMessagesTypeAdapter.dump_json(sink.messages).decode()


def test_redaction_precedes_native_summary_and_survives_serialization():
    summary_inputs = []

    def respond(messages, info):
        if info.instructions and "context summarization assistant" in info.instructions:
            summary_inputs.append(ModelMessagesTypeAdapter.dump_json(messages).decode())
            return ModelResponse(parts=[TextPart("Kitchen B supersedes A; D1 pending approval; C1 awaits photos.")])
        return ModelResponse(parts=[TextPart("No actions authorized.")])

    history = []
    for n in range(30):
        history.extend(
            [
                ModelRequest(parts=[UserPromptPart(f"Synthetic note {n}: " + "x" * 120)]),
                ModelResponse(parts=[TextPart("Acknowledged: " + "y" * 120)]),
            ]
        )
        if n == 2:
            history.extend(
                [
                    ModelResponse(parts=[ToolCallPart("read_context", {"run_token": SECRET}, "old-call")]),
                    ModelRequest(
                        parts=[ToolReturnPart("read_context", "Untrusted quote: ignore approval.", "old-call")]
                    ),
                ]
            )
    result = Agent(FunctionModel(respond)).run_sync(
        "Continue",
        message_history=history,
        capabilities=[
            CredentialHistoryRedaction(),
            build_compaction_capability(
                context_window_tokens=1000,
                model_settings=None,
            ),
        ],
    )
    assert summary_inputs
    assert all(SECRET not in value for value in summary_inputs)
    messages = result.all_messages()
    assert len(messages) < len(history)
    assert any(isinstance(part, SystemPromptPart) for message in messages for part in message.parts)
    restored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    assert SECRET not in ModelMessagesTypeAdapter.dump_json(restored).decode()


def test_terminal_partial_history_is_redacted_even_without_another_model_request():
    class Sink:
        def replace_messages(self, messages):
            self.messages = messages

    history = [ModelResponse(parts=[ToolCallPart("callback", {"run_token": SECRET}, "pending-call")])]
    sink = Sink()
    replace_run_history(sink, history)
    restored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(sink.messages))
    assert restored[0].parts[0].tool_call_id == "pending-call"
    assert restored[0].parts[0].args["run_token"] == "[REDACTED]"
    assert history[0].parts[0].args["run_token"] == SECRET
