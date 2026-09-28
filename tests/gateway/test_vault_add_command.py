"""Gateway ``/vault-add``: private chat + owner allowlist only."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.slash_commands_vault_enroll import GatewayVaultEnrollCommandsMixin


class _Runner(GatewayVaultEnrollCommandsMixin):
    def __init__(self, adapter=None):
        self._adapter = adapter

    def _delivery_adapter_for(self, source):
        return self._adapter

    def _thread_metadata_for_source(self, source):
        return {}


def _event(*, platform="telegram", chat_type="dm", chat_id="42", user_id="42"):
    source = SimpleNamespace(
        platform=SimpleNamespace(value=platform),
        chat_type=chat_type,
        chat_id=chat_id,
        user_id=user_id,
    )
    return SimpleNamespace(source=source)


@pytest.fixture
def ready(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "000000000:TEST_BOT_TOKEN_FOR_TESTS_ONLY")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setattr(
        "gateway.slash_commands_vault_enroll.enrollment_prereqs",
        lambda: SimpleNamespace(
            ready=True,
            enroll_url="https://vault.test/vault/enroll",
            allowlist=("42",),
            public_origin="https://vault.test",
            reason="",
        ),
    )


@pytest.mark.asyncio
async def test_rejects_group_chat(ready):
    runner = _Runner()
    out = await runner._handle_vault_add_command(_event(chat_type="group", chat_id="-1001"))
    assert "private" in out.lower() or "Groups" in out


@pytest.mark.asyncio
async def test_rejects_non_owner(ready):
    runner = _Runner()
    out = await runner._handle_vault_add_command(_event(user_id="777"))
    assert "allowlisted" in out.lower() or "owners" in out.lower()


@pytest.mark.asyncio
async def test_rejects_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        "gateway.slash_commands_vault_enroll.enrollment_prereqs",
        lambda: SimpleNamespace(
            ready=False, reason="disabled", public_origin="", enroll_url="", allowlist=(),
        ),
    )
    runner = _Runner()
    out = await runner._handle_vault_add_command(_event())
    assert "not ready" in out.lower()


@pytest.mark.asyncio
async def test_sends_webapp_prompt_for_owner_dm(ready):
    adapter = SimpleNamespace(send_vault_enroll_prompt=AsyncMock(return_value=SimpleNamespace(success=True)))
    runner = _Runner(adapter)
    out = await runner._handle_vault_add_command(_event())
    assert out == ""
    adapter.send_vault_enroll_prompt.assert_awaited_once()
    args = adapter.send_vault_enroll_prompt.await_args
    assert args.args[1] == "https://vault.test/vault/enroll"
