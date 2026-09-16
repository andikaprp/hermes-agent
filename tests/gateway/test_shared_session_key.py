"""Regression tests for opt-in cross-platform shared session keys."""
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import Platform


def _source(platform, chat_id):
    return SessionSource(platform=platform, chat_type="dm", chat_id=chat_id, user_id="user")


def test_shared_name_converges_across_platforms():
    assert build_session_key(_source(Platform.SLACK, "s"), shared_session_name="Ningning") == build_session_key(
        _source(Platform.TELEGRAM, "t"), shared_session_name="Ningning"
    )


def test_shared_name_retains_profile_isolation():
    source = _source(Platform.LOCAL, "cli")
    assert build_session_key(source, shared_session_name="Ningning", profile="default") != build_session_key(
        source, shared_session_name="Ningning", profile="coder"
    )


def test_unset_shared_name_preserves_legacy_keys():
    assert build_session_key(_source(Platform.SLACK, "s")) != build_session_key(_source(Platform.TELEGRAM, "t"))
