"""Raw-export boundary regressions for missing children inference."""

import json

import pytest
from pydantic import ValidationError

from convoviz.models import ConversationCollection, Node
from convoviz.models.node import build_node_tree


@pytest.mark.parametrize("explicit", [False, True])
def test_raw_json_distinguishes_missing_and_empty_children(*, explicit: bool) -> None:
    root = {"id": "root"}
    if explicit:
        root["children"] = []
    mapping = {
        "root": Node.model_validate_json(json.dumps(root)),
        "child": Node.model_validate_json('{"id":"child","parent":"root"}'),
    }
    build_node_tree(mapping)
    assert ("children" in mapping["root"].model_fields_set) is explicit
    assert (mapping["child"].parent_node is mapping["root"]) is not explicit


def test_raw_json_null_children_keeps_existing_validation_error() -> None:
    # Null is not an omitted list and must not silently become inferred edges.
    with pytest.raises(ValidationError) as exc:
        Node.model_validate_json('{"id":"root","children":null}')
    assert exc.value.errors()[0]["loc"] == ("children",)


def test_inference_is_scoped_to_each_conversation_mapping() -> None:
    def record(identifier: str, mapping: dict) -> dict:
        return {
            "conversation_id": identifier,
            "create_time": 0,
            "update_time": 0,
            "current_node": "child",
            "mapping": mapping,
        }

    collection = ConversationCollection.model_validate_json(
        json.dumps(
            {
                "conversations": [
                    record("one", {"root": {"id": "root"}}),
                    record("two", {"child": {"id": "child", "parent": "root"}}),
                    record(
                        "three",
                        {
                            "root": {"id": "root"},
                            "child": {"id": "child", "parent": "root"},
                        },
                    ),
                ]
            }
        )
    )
    first, second, third = collection.conversations
    assert first.node_mapping["root"].children_nodes == []
    assert [node.id for node in second.ordered_nodes] == ["child"]
    assert [node.id for node in third.ordered_nodes] == ["root", "child"]
    assert third.node_mapping["child"].parent_node is third.mapping["root"]
    assert second.node_mapping["child"].parent_node is None
    assert first.node_mapping["root"].children_nodes == []
