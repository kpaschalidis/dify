import asyncio
import base64
import json

import httpx
import pytest
from pydantic_ai import Agent, BinaryContent, ImageUrl, ToolReturn
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from agenton.compositor import Compositor, LayerNode
from dify_agent.adapters.llm.model import _map_model_request_to_prompt_messages
from dify_agent.layers.dify_plugin.tool_client import DifyPluginToolInvokeMessage
from dify_agent.layers.dify_plugin.tools_layer import DifyPluginToolsLayer, _convert_tool_response

from .test_layers import _execution_context_config, _execution_context_provider, _tools_config, _tools_provider


def test_daemon_image_blobs_reach_next_model_step_and_survive_history_roundtrip() -> None:
    image_bytes = [b"first-image", b"second-image"]

    def handler(request: httpx.Request) -> httpx.Response:
        messages = [{"type": "json", "message": {"text": "Original page text"}}]
        messages.extend(
            {
                "type": "blob",
                "message": {"blob": base64.b64encode(data).decode()},
                "meta": {"mime_type": mime, "filename": f"page-{index + 1}"},
            }
            for index, (data, mime) in enumerate(zip(image_bytes, ["image/png", "image/jpeg"], strict=True))
        )
        return httpx.Response(
            200, text="\n".join("data: " + json.dumps({"code": 0, "message": "ok", "data": m}) for m in messages)
        )

    requests = []

    def model(messages, _info):
        requests.append(messages)
        if len(requests) == 1:
            return ModelResponse(parts=[ToolCallPart("web_search", {"query": "page"}, tool_call_id="image-call")])
        return ModelResponse(parts=[TextPart("Images inspected")])

    async def scenario():
        compositor = Compositor(
            [
                LayerNode("execution_context", _execution_context_provider()),
                LayerNode("tools", _tools_provider(), deps={"execution_context": "execution_context"}),
            ]
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client, compositor.enter(
            configs={"execution_context": _execution_context_config(), "tools": _tools_config()}
        ) as run:
            layer = run.get_layer("tools", DifyPluginToolsLayer)
            tools = await layer.get_tools(http_client=client, dify_api_http_client=client)
            result = await Agent(FunctionModel(model), tools=tools).run("Inspect the pages")
        assert result.output == "Images inspected"
        history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(result.all_messages()))
        request = next(
            m for m in history if isinstance(m, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in m.parts)
        )
        observation = next(p for p in request.parts if isinstance(p, ToolReturnPart))
        assert "Original page text" in str(observation.content)
        assert "first-image" not in str(observation.content)
        visual = next(p for p in request.parts if isinstance(p, UserPromptPart))
        assert not isinstance(visual.content, str)
        assert "tool_call_id=image-call" in visual.content[0]
        assert [p.data for p in visual.content if isinstance(p, BinaryContent)] == image_bytes
        # This is the actual adapter boundary used by Dify's model gateway.
        wire = [m.model_dump(mode="json") for m in _map_model_request_to_prompt_messages(request)]
        assert [m["role"] for m in wire] == ["tool", "user"]
        images = [p for p in wire[1]["content"] if p["type"] == "image"]
        assert len(images) == 2
        assert [p["mime_type"] for p in images] == ["image/png", "image/jpeg"]
        assert base64.b64decode(images[0]["base64_data"]) == image_bytes[0]

    asyncio.run(scenario())


def test_image_urls_are_native_content_and_text_only_results_stay_unchanged() -> None:
    text = DifyPluginToolInvokeMessage(type="text", message=DifyPluginToolInvokeMessage.TextMessage(text="original"))
    assert _convert_tool_response([text], tool_name="read", tool_call_id=None) == "original"
    result = _convert_tool_response(
        [
            text,
            DifyPluginToolInvokeMessage(
                type="image_link", message=DifyPluginToolInvokeMessage.TextMessage(text="https://example.com/page.png")
            ),
        ],
        tool_name="read",
        tool_call_id="call-1",
    )
    assert isinstance(result, ToolReturn)
    assert result.return_value == "original\n1 image(s) attached for inspection."
    assert any(isinstance(p, ImageUrl) and p.url == "https://example.com/page.png" for p in result.content)


@pytest.mark.parametrize("mime,data", [("image/svg+xml", b"<svg>"), ("image/png", b"")])
def test_unsupported_or_empty_image_blobs_fail_without_stringifying_bytes(mime, data) -> None:
    message = DifyPluginToolInvokeMessage(
        type="blob", message=DifyPluginToolInvokeMessage.BlobMessage(blob=data), meta={"mime_type": mime}
    )
    with pytest.raises(ValueError, match="tool image"):
        _convert_tool_response([message], tool_name="read", tool_call_id=None)
