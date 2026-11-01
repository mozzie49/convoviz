"""Synthetic regressions for exports that omit node children (issue #70)."""

from datetime import UTC, datetime

import pytest

from convoviz.config import AuthorHeaders, ConversationConfig
from convoviz.models import Conversation, Node
from convoviz.models.node import build_node_tree
from convoviz.renderers import render_conversation
from convoviz.renderers.markdown import _ordered_nodes_full


def _conversation(mapping: dict[str, Node], current: str) -> Conversation:
    now = datetime(2026, 10, 1, tzinfo=UTC)
    return Conversation(
        create_time=now,
        update_time=now,
        conversation_id="synthetic-parent-only",
        mapping=mapping,
        current_node=current,
    )


def _branching_mapping() -> dict[str, Node]:
    # Deliberately not chronological: branch selection must follow parent IDs.
    parents = {
        "selected": "prompt",
        "alternate": "prompt",
        "prompt": "root",
        "root": None,
    }
    return {
        key: Node.model_validate(
            {
                "id": key,
                "parent": parent,
                "message": None
                if key == "root"
                else {
                    "id": key,
                    "author": {"role": "user" if key == "prompt" else "assistant"},
                    "content": {"content_type": "text", "parts": [f"Text {key}"]},
                },
            }
        )
        for key, parent in parents.items()
    }


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ("selected", ["root", "prompt", "selected"]),
        ("alternate", ["root", "prompt", "alternate"]),
        ("prompt", ["root", "prompt"]),
        ("root", ["root"]),
        ("absent", []),
    ],
)
def test_parent_only_active_branch(current: str, expected: list[str]) -> None:
    conversation = _conversation(_branching_mapping(), current)
    assert [node.id for node in conversation.ordered_nodes] == expected


def test_parent_only_rendering_preserves_selected_branch_and_alternatives() -> None:
    conversation = _conversation(_branching_mapping(), "selected")
    markdown = render_conversation(conversation, ConversationConfig(), AuthorHeaders())
    assert markdown.index("Text prompt") < markdown.index("Text selected")
    assert "Text alternate" not in markdown
    assert [node.id for node in _ordered_nodes_full(conversation)] == [
        "root",
        "prompt",
        "alternate",
        "selected",
    ]
    assert set(conversation.mapping) == {"root", "prompt", "alternate", "selected"}
    config = ConversationConfig()
    config.markdown.render_order = "full"
    full_markdown = render_conversation(conversation, config, AuthorHeaders())
    for key in ("prompt", "selected", "alternate"):
        assert full_markdown.count(f"Text {key}") == 1


@pytest.mark.parametrize("render_order", ["active", "full"])
def test_parent_only_matches_equivalent_legacy_rendering(render_order: str) -> None:
    parent_only = _conversation(_branching_mapping(), "selected")
    legacy = _conversation(_branching_mapping(), "selected")
    legacy.mapping["root"].children = ["prompt"]
    legacy.mapping["prompt"].children = ["selected", "alternate"]
    legacy.mapping["selected"].children = []
    legacy.mapping["alternate"].children = []
    config = ConversationConfig.model_validate(
        {"markdown": {"render_order": render_order}}
    )
    assert render_conversation(
        parent_only, config, AuthorHeaders()
    ) == render_conversation(legacy, config, AuthorHeaders())


def test_inferred_edges_are_repeatable_and_leave_export_fields_unchanged() -> None:
    mapping = _branching_mapping()
    for _ in range(2):
        assert build_node_tree(mapping) is mapping
        assert [node.id for node in mapping["prompt"].children_nodes] == [
            "selected",
            "alternate",
        ]
        assert mapping["selected"].parent_node is mapping["prompt"]
        assert mapping["prompt"].children == []
        assert "children" not in mapping["prompt"].model_fields_set


@pytest.mark.parametrize("children", [[], ["other"], ["missing"]])
def test_explicit_children_are_not_supplemented(children: list[str]) -> None:
    mapping = {
        "root": Node(id="root", children=children),
        "child": Node(id="child", parent="root"),
        "other": Node(id="other", parent="root"),
    }
    build_node_tree(mapping)
    assert mapping["child"].parent_node is None
    assert [node.id for node in mapping["root"].children_nodes] == [
        key for key in children if key in mapping
    ]


def test_mixed_export_retains_explicit_order_and_fills_omitted_children() -> None:
    mapping = {
        "root": Node(id="root", children=["second", "first"]),
        "first": Node(id="first", parent="root"),
        "second": Node(id="second", parent="root", children=[]),
        "leaf": Node(id="leaf", parent="first"),
    }
    conversation = _conversation(mapping, "leaf")
    assert [node.id for node in conversation.ordered_nodes] == ["root", "first", "leaf"]
    assert [node.id for node in mapping["root"].children_nodes] == ["second", "first"]


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_conflicting_parent_is_not_overridden(*, reverse: bool) -> None:
    mapping = {
        "claimed": Node(id="claimed"),
        "explicit": Node(id="explicit", children=["child"]),
        "child": Node(id="child", parent="claimed"),
    }
    if reverse:
        mapping = dict(reversed(mapping.items()))
    build_node_tree(mapping)
    assert mapping["child"].parent_node is mapping["explicit"]
    assert mapping["claimed"].children_nodes == []


def test_missing_parent_and_disconnected_roots_are_not_grafted() -> None:
    mapping = {
        "root": Node(id="root"),
        "orphan": Node(id="orphan", parent="missing"),
        "leaf": Node(id="leaf", parent="orphan"),
        "unrelated": Node(id="unrelated"),
    }
    conversation = _conversation(mapping, "leaf")
    assert [node.id for node in conversation.ordered_nodes] == ["orphan", "leaf"]
    assert mapping["orphan"].parent_node is None
    assert mapping["root"].children_nodes == []
    assert mapping["unrelated"].children_nodes == []
    assert {node.id for node in _ordered_nodes_full(conversation)} == set(mapping)


@pytest.mark.parametrize("self_cycle", [False, True])
def test_parent_only_cycles_keep_existing_cycle_safe_traversal(
    *, self_cycle: bool
) -> None:
    mapping = {"root": Node(id="root", parent="root" if self_cycle else "child")}
    if not self_cycle:
        mapping["child"] = Node(id="child", parent="root")
    current = "root" if self_cycle else "child"
    conversation = _conversation(mapping, current)
    assert [node.id for node in conversation.ordered_nodes] == list(mapping)
    assert {node.id for node in _ordered_nodes_full(conversation)} == set(mapping)
