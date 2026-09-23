"""Tests for synchronous llm_stream_text transformation before live delivery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.stream_delivery import StreamDeliveryMixin


class _Agent(StreamDeliveryMixin):
    session_id = "session-1"
    model = "model-1"
    provider = "openrouter"
    platform = "cli"
    show_commentary = True
    _current_turn_id = "turn-1"
    _current_api_request_id = "request-1"
    _api_call_count = 1
    _stream_callback = None
    _stream_needs_break = False
    _stream_think_scrubber = None
    _stream_context_scrubber = None
    _stream_reasoning_hooks_enabled = False

    def __init__(self):
        self.stream_delta_callback = None
        self.reasoning_callback = None
        self.interim_assistant_callback = None
        self._streamed_assistant_text_parts = []
        self._delivered_interim_texts = set()

    @staticmethod
    def _strip_think_blocks(text):
        return text

    def _stream_writer_superseded(self):
        return False


def test_text_is_transformed_before_display_and_observer(monkeypatch):
    seen_context = []
    observed = []
    delivered = []

    def transform(text, *, kind, **context):
        seen_context.append((kind, context))
        return f"restored:{text}"

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        transform,
    )
    monkeypatch.setattr(
        _Agent,
        "_enqueue_stream_hook",
        lambda self, event, **fields: observed.append((event, fields)),
    )

    agent = _Agent()
    agent.stream_delta_callback = delivered.append
    agent._fire_stream_delta("token")

    assert delivered == ["restored:token"]
    assert observed[-1] == (
        "on_stream_delta",
        {"delta": "restored:token", "kind": "text"},
    )
    kind, context = seen_context[-1]
    assert kind == "text"
    assert context["session_id"] == "session-1"
    assert context["turn_id"] == "turn-1"
    assert context["api_request_id"] == "request-1"
    assert context["provider"] == "openrouter"
    assert context["model"] == "model-1"


def test_reasoning_is_transformed_before_reasoning_callback(monkeypatch):
    delivered = []

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        lambda text, *, kind, **context: f"{kind}:{text}",
    )

    agent = _Agent()
    agent.reasoning_callback = delivered.append
    agent._fire_reasoning_delta("token")

    assert delivered == ["reasoning:token"]


def test_interim_is_transformed_before_interim_callback(monkeypatch):
    delivered = []

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        lambda text, *, kind, **context: f"{kind}:{text}",
    )

    agent = _Agent()
    agent.interim_assistant_callback = (
        lambda text, *, already_streamed=False: delivered.append(
            (text, already_streamed)
        )
    )
    agent._emit_interim_assistant_message({"content": "token"})

    assert delivered == [("interim:token", False)]


def test_fail_closed_transform_error_prevents_text_delivery(monkeypatch):
    delivered = []

    def fail(*args, **kwargs):
        raise RuntimeError("privacy transform failed")

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        fail,
    )

    agent = _Agent()
    agent.stream_delta_callback = delivered.append

    with pytest.raises(RuntimeError, match="privacy transform failed"):
        agent._fire_stream_delta("must-not-display")

    assert delivered == []
