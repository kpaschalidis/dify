import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import TypeAdapter, ValidationError

from fields.conversation_fields import MessageResponseSource
from fields.message_fields import (
    ExploreMessageListItem,
    MessageInfiniteScrollPagination,
    MessageListItem,
    WebMessageListItem,
)
from models.model import Message


def _base_kwargs():
    return {
        "id": "m1",
        "conversation_id": "c1",
        "inputs": {},
        "query": "hi",
        "answer": "answer",
        "retriever_resources": [],
        "agent_thoughts": [],
        "message_files": [],
        "status": "normal",
        "extra_contents": [],
    }


class TestExploreMessageListItem:
    def test_exposes_metadata_for_history_rehydration(self):
        # The Explore/installed-app surface must surface message_metadata (incl. reasoning)
        # so the chat-with-history client can rehydrate the thinking panel on reload.
        item = ExploreMessageListItem(**_base_kwargs(), metadata={"reasoning": {"llm": "thinking..."}})

        payload = item.model_dump(mode="json")

        assert payload["metadata"] == {"reasoning": {"llm": "thinking..."}}

    def test_metadata_defaults_to_none(self):
        item = ExploreMessageListItem(**_base_kwargs())
        assert item.model_dump(mode="json")["metadata"] is None

    def test_base_message_list_item_has_no_metadata(self):
        # Guard the public service-API contract: the base item must not leak metadata.
        payload = MessageListItem(**_base_kwargs()).model_dump(mode="json")
        assert "metadata" not in payload

    def test_message_list_item_exposes_usage_fields(self):
        payload = MessageListItem(
            **_base_kwargs(),
            message_tokens=7,
            answer_tokens=11,
            provider_response_latency=1.25,
            total_price=Decimal("0.0001234"),
            currency="USD",
        ).model_dump(mode="json")

        assert payload["message_tokens"] == 7
        assert payload["answer_tokens"] == 11
        assert payload["total_tokens"] == 18
        assert payload["provider_response_latency"] == 1.25
        assert payload["total_price"] == "0.0001234"
        assert payload["currency"] == "USD"

    def test_web_message_list_item_exposes_usage_and_metadata(self):
        payload = WebMessageListItem(
            **_base_kwargs(),
            metadata={"usage": {"total_tokens": 18}},
            message_tokens=7,
            answer_tokens=11,
        ).model_dump(mode="json")

        assert payload["metadata"] == {"usage": {"total_tokens": 18}}
        assert payload["message_tokens"] == 7
        assert payload["answer_tokens"] == 11
        assert payload["total_tokens"] == 18


@pytest.mark.parametrize("model", [MessageListItem, WebMessageListItem, ExploreMessageListItem])
@pytest.mark.parametrize("resources", [None, []])
def test_history_normalizes_absent_retrieval(model, resources):
    kwargs = _base_kwargs()
    kwargs["retriever_resources"] = resources
    assert model(**kwargs).model_dump(mode="json")["retriever_resources"] == []
    del kwargs["retriever_resources"]
    assert model(**kwargs).model_dump(mode="json")["retriever_resources"] == []


@pytest.mark.parametrize("metadata", [None, {}, {"usage": {"total_tokens": 18}}, {"retriever_resources": None}])
def test_history_accepts_model_metadata_without_retrieval(metadata):
    # AgentAppRunner._persist_message_usage writes usage-only metadata, then
    # Message.retriever_resources returns None for that missing retrieval key.
    message = Message(message_metadata=json.dumps(metadata) if metadata is not None else None)
    kwargs = _base_kwargs()
    kwargs["retriever_resources"] = message.retriever_resources
    # Exercise the same attribute adapter and Pydantic boundary as GET /v1/messages.
    source = SimpleNamespace(**kwargs)
    source.re_sign_file_url_answer = kwargs["answer"]
    source.inputs_with_session = Mock(return_value={})
    source.user_feedback_with_session = Mock(return_value=None)
    source.agent_thoughts_with_session = Mock(return_value=[])
    source.message_files_with_session = Mock(return_value=[])
    item = TypeAdapter(MessageListItem).validate_python(MessageResponseSource(source, session=Mock()))
    assert item.retriever_resources == []
    assert item.answer == "answer"
    assert message.message_metadata == (json.dumps(metadata) if metadata is not None else None)


@pytest.mark.parametrize("resources", [False, 0, "", {}, [None], [{"position": "invalid"}]])
def test_history_retains_validation_of_malformed_retrieval(resources):
    kwargs = _base_kwargs()
    kwargs["retriever_resources"] = resources
    with pytest.raises(ValidationError) as exc:
        MessageListItem(**kwargs)
    assert exc.value.errors()[0]["loc"][0] == "retriever_resources"


def test_history_preserves_citations_messages_and_pagination():
    citation = {
        "id": "resource",
        "message_id": "m1",
        "position": 1,
        "dataset_id": "dataset",
        "document_id": "document",
        "document_name": "Policy",
        "segment_id": "segment",
        "content": "Insured cover",
        "score": 0.9,
    }
    first = _base_kwargs() | {"retriever_resources": [citation]}
    second = _base_kwargs() | {"id": "m2", "retriever_resources": None}
    payload = MessageInfiniteScrollPagination(limit=2, has_more=True, data=[first, second]).model_dump(mode="json")
    assert payload["limit"] == 2
    assert payload["has_more"] is True
    assert [item["id"] for item in payload["data"]] == ["m1", "m2"]
    assert payload["data"][1]["retriever_resources"] == []
    for key, value in citation.items():
        assert payload["data"][0]["retriever_resources"][0][key] == value
