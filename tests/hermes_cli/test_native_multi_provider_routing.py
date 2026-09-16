"""Contract tests for the native OAuth/provider fallback route."""
from __future__ import annotations

from hermes_cli.fallback_config import get_fallback_chain
from hermes_cli.providers import get_provider
from hermes_cli.models import _PROVIDER_MODELS


def test_native_multi_provider_route_uses_supported_adapters():
    config = {
        "model": {"provider": "openai-codex", "model": "gpt-5.6-luna"},
        "fallback_providers": [
            {"provider": "xai-oauth", "model": "grok-4.3"},
            {"provider": "nous", "model": "inclusionai/ling-3.0-flash-fin:free"},
            {"provider": "nous", "model": "poolside/laguna-xs-2.1:free"},
            {"provider": "nous", "model": "poolside/laguna-s-2.1:free"},
            {"provider": "nous", "model": "upstage/solar-pro4:free"},
            {"provider": "nous", "model": "meituan/longcat-2.0:free"},
            {"provider": "nous", "model": "stepfun/step-3.7-flash:free"},
            {"provider": "opencode-free", "model": "nemotron-3.5-lightning-free"},
            {"provider": "opencode-free", "model": "hy3-free"},
            {"provider": "opencode-free", "model": "deepseek-v4-flash-free"},
            {"provider": "opencode-free", "model": "mimo-v2.5-free"},
            {"provider": "opencode-free", "model": "muse-spark-1.2-contributor-free"},
        ],
    }

    assert config["model"] == {
        "provider": "openai-codex",
        "model": "gpt-5.6-luna",
    }
    chain = get_fallback_chain(config)
    assert [(entry["provider"], entry["model"]) for entry in chain] == [
        ("xai-oauth", "grok-4.3"),
        ("nous", "inclusionai/ling-3.0-flash-fin:free"),
        ("nous", "poolside/laguna-xs-2.1:free"),
        ("nous", "poolside/laguna-s-2.1:free"),
        ("nous", "upstage/solar-pro4:free"),
        ("nous", "meituan/longcat-2.0:free"),
        ("nous", "stepfun/step-3.7-flash:free"),
        ("opencode-free", "nemotron-3.5-lightning-free"),
        ("opencode-free", "hy3-free"),
        ("opencode-free", "deepseek-v4-flash-free"),
        ("opencode-free", "mimo-v2.5-free"),
        ("opencode-free", "muse-spark-1.2-contributor-free"),
    ]

    # Every configured model is present in Hermes' offline validation floor.
    for entry in chain:
        assert entry["model"] in _PROVIDER_MODELS[entry["provider"]]

    # Resolve through Hermes' built-in definitions, not an external gateway.
    for provider in ("openai-codex", "xai-oauth", "nous", "opencode-free"):
        assert get_provider(provider, allow_network=False) is not None


def test_omniroute_is_not_part_of_native_route():
    config = {
        "fallback_providers": [
            {"provider": "xai-oauth", "model": "grok-4.3"},
            {"provider": "nous", "model": "inclusionai/ling-3.0-flash-fin:free"},
            {"provider": "nous", "model": "poolside/laguna-xs-2.1:free"},
            {"provider": "nous", "model": "poolside/laguna-s-2.1:free"},
            {"provider": "nous", "model": "upstage/solar-pro4:free"},
            {"provider": "nous", "model": "meituan/longcat-2.0:free"},
            {"provider": "nous", "model": "stepfun/step-3.7-flash:free"},
            {"provider": "opencode-free", "model": "nemotron-3.5-lightning-free"},
            {"provider": "opencode-free", "model": "hy3-free"},
            {"provider": "opencode-free", "model": "deepseek-v4-flash-free"},
            {"provider": "opencode-free", "model": "mimo-v2.5-free"},
            {"provider": "opencode-free", "model": "muse-spark-1.2-contributor-free"},
        ]
    }
    assert all(entry["provider"] != "omniroute" for entry in get_fallback_chain(config))
