"""Regression tests for selectable plugin middleware failure policies."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from hermes_cli.middleware import (
    run_llm_execution_middleware,
    run_llm_stream_text_middleware,
)
from hermes_cli.plugins import PluginManager


def _load_plugin(tmp_path, name: str, register_body: str) -> PluginManager:
    home = tmp_path / "home"
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({
            "name": name,
            "version": "0.1.0",
            "description": "middleware failure-mode test",
        }),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n" + textwrap.indent(register_body.strip() + "\n", "    "),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [name]}}),
        encoding="utf-8",
    )

    import os

    os.environ["HERMES_HOME"] = str(home)
    manager = PluginManager()
    manager.discover_and_load()
    return manager


def _use_manager(monkeypatch, manager: PluginManager) -> None:
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)


def test_execution_failure_mode_is_selected_per_plugin_registration(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        "failure-policy",
        """
def fail_open(**kwargs):
    raise RuntimeError("open failure")

def fail_closed(**kwargs):
    raise RuntimeError("closed failure")

ctx.register_middleware("llm_execution", fail_open, failure_mode="open")
ctx.register_middleware("tool_execution", fail_closed, failure_mode="closed")
""",
    )

    assert manager._middleware["llm_execution"][0]._hermes_failure_mode == "open"
    assert manager._middleware["tool_execution"][0]._hermes_failure_mode == "closed"


def test_fail_open_execution_keeps_legacy_fallthrough(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        "fail-open",
        """
def protect(**kwargs):
    raise RuntimeError("plugin unavailable")

ctx.register_middleware("llm_execution", protect, failure_mode="open")
""",
    )
    _use_manager(monkeypatch, manager)

    calls = []

    def provider(request):
        calls.append(request)
        return {"ok": True}

    result = run_llm_execution_middleware({"messages": []}, provider)

    assert result == {"ok": True}
    assert calls == [{"messages": []}]


def test_fail_closed_execution_blocks_before_provider(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        "fail-closed-pre",
        """
def protect(**kwargs):
    raise RuntimeError("privacy boundary unavailable")

ctx.register_middleware("llm_execution", protect, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    calls = []

    def provider(request):
        calls.append(request)
        return {"ok": True}

    with pytest.raises(RuntimeError, match="privacy boundary unavailable"):
        run_llm_execution_middleware({"messages": []}, provider)

    assert calls == []


def test_fail_closed_execution_propagates_post_provider_failure(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        "fail-closed-post",
        """
def protect(**kwargs):
    kwargs["next_call"](kwargs["request"])
    raise RuntimeError("restore failed")

ctx.register_middleware("llm_execution", protect, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    calls = []

    def provider(request):
        calls.append(request)
        return {"unsafe": "provider response"}

    with pytest.raises(RuntimeError, match="restore failed"):
        run_llm_execution_middleware({"messages": []}, provider)

    assert calls == [{"messages": []}]


def test_fail_open_execution_preserves_post_provider_result(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        "fail-open-post",
        """
def observe(**kwargs):
    kwargs["next_call"](kwargs["request"])
    raise RuntimeError("observer failed")

ctx.register_middleware("llm_execution", observe)
""",
    )
    _use_manager(monkeypatch, manager)

    result = run_llm_execution_middleware(
        {"messages": []},
        lambda request: {"result": request},
    )

    assert result == {"result": {"messages": []}}


def test_live_text_transform_receives_request_identity_and_rewrites(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        "stream-transform",
        """
def transform(**kwargs):
    return {"text": f"{kwargs['kind']}:{kwargs['api_request_id']}:{kwargs['text']}"}

ctx.register_middleware("llm_stream_text", transform, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    result = run_llm_stream_text_middleware(
        "tokenized",
        kind="text",
        provider="openrouter",
        model="model",
        session_id="session-1",
        turn_id="turn-1",
        api_request_id="request-1",
    )

    assert result == "text:request-1:tokenized"


def test_live_text_failure_policy_can_be_open_or_closed(tmp_path, monkeypatch):
    manager = _load_plugin(
        tmp_path,
        "stream-open",
        """
def transform(**kwargs):
    raise RuntimeError("stream transform failed")

ctx.register_middleware("llm_stream_text", transform, failure_mode="open")
""",
    )
    _use_manager(monkeypatch, manager)

    assert run_llm_stream_text_middleware("visible", kind="text") == "visible"

    manager = _load_plugin(
        tmp_path,
        "stream-closed",
        """
def transform(**kwargs):
    raise RuntimeError("stream transform failed")

ctx.register_middleware("llm_stream_text", transform, failure_mode="closed")
""",
    )
    _use_manager(monkeypatch, manager)

    with pytest.raises(RuntimeError, match="stream transform failed"):
        run_llm_stream_text_middleware("must-not-deliver", kind="text")


def test_invalid_failure_mode_is_rejected(tmp_path):
    with pytest.raises(Exception, match="failure_mode"):
        _load_plugin(
            tmp_path,
            "bad-mode",
            """
def callback(**kwargs):
    return None

ctx.register_middleware("llm_execution", callback, failure_mode="maybe")
""",
        )
