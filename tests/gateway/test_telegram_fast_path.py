"""The no-task fast path: which turns get the one-hop note, and which must never get it.

LAB-3: 67 of 100 live Telegram turns spent more than one model call, and casual turns were
2-3x slower than the one-call median purely from entering the agentic loop. The decision this
file pins is the risky half of that change -- routing a real task into a "no tools" hint would
silently under-serve the user, so the gate has to be provably conservative.
"""

import pytest

from agent.memory_provider import is_trivial_prompt
from gateway.run_turn_fast_path import (
    FAST_PATH_NOTE,
    _ACK_WORDS,
    _DIRECTIVE_WORDS,
    apply_fast_path_note,
    classify_fast_path,
    fast_path_reason,
)

ASSISTANT_ASKED = [{"role": "assistant", "content": "Shall I deploy it to production?"}]
ASSISTANT_QUESTION = [{"role": "assistant", "content": "Want me to retry that?"}]
ASSISTANT_STATED = [{"role": "assistant", "content": "Deployed. Everything is green."}]


# ── turns that should take the fast path ──────────────────────────────────


@pytest.mark.parametrize("text", ["hi", "hey", "hello", "thanks", "thank you", "cool", "nice", "great"])
def test_social_replies_take_the_fast_path_regardless_of_context(text):
    for history in (None, ASSISTANT_STATED, ASSISTANT_ASKED):
        assert classify_fast_path(text, history=history) == "social"


@pytest.mark.parametrize("text", ["yes", "ok", "sure", "yeah", "nope", "got it"])
def test_acks_take_the_fast_path_when_the_assistant_proposed_nothing(text):
    assert classify_fast_path(text, history=ASSISTANT_STATED) == "ack"


def test_chat_elongation_is_still_an_ack():
    """LAB-3's worst turn was ``yess`` -> 24 model calls, 23 tool calls, 206 s.

    ``TRIVIAL_PROMPT_RE`` tolerates trailing punctuation but not a repeated letter, so the
    elongated spelling read as a substantive prompt.
    """
    assert not is_trivial_prompt("yess")
    assert classify_fast_path("yess", history=ASSISTANT_STATED) == "ack"
    assert classify_fast_path("okkk", history=ASSISTANT_STATED) == "ack"
    assert classify_fast_path("hiii", history=None) == "social"


# ── turns that must NOT take the fast path ────────────────────────────────


@pytest.mark.parametrize("text", ["yes", "yess", "ok", "sure", "yeah", "got it", "lgtm"])
@pytest.mark.parametrize("history", [ASSISTANT_ASKED, ASSISTANT_QUESTION])
def test_an_ack_answering_a_proposal_is_a_go_ahead_and_keeps_its_tools(text, history):
    """The failure mode this gate exists to avoid: "yes" to "shall I deploy?" is real work."""
    assert classify_fast_path(text, history=history) is None


@pytest.mark.parametrize("text", sorted(_DIRECTIVE_WORDS))
def test_directive_words_never_take_the_fast_path(text):
    for history in (None, ASSISTANT_STATED, ASSISTANT_ASKED):
        assert classify_fast_path(text, history=history) is None


def test_an_ack_after_a_pending_tool_call_keeps_its_tools():
    history = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}]
    assert classify_fast_path("ok", history=history) is None


@pytest.mark.parametrize("text", [
    "deploy the api",
    "what did I ask you yesterday?",
    "run the tests again",
    "hey there, can you check the logs for me",
])
def test_real_tasks_never_take_the_fast_path(text):
    assert classify_fast_path(text, history=ASSISTANT_STATED) is None


def test_slash_commands_never_take_the_fast_path():
    """``is_trivial_prompt`` calls these trivial; for routing they are commands."""
    for text in ("/stop", "/new", "/status"):
        assert is_trivial_prompt(text)
        assert classify_fast_path(text, history=None) is None


def test_media_and_quoted_replies_fall_out_via_their_inbound_notes():
    """Inbound preprocessing folds these into the text, and the trivial match is anchored."""
    assert classify_fast_path("[Image: cat.png]\nhi", history=None) is None
    assert classify_fast_path("[Replying to: ship it]\n\nok", history=None) is None


def test_native_multimodal_content_is_not_a_fast_path_turn():
    assert classify_fast_path([{"type": "text", "text": "hi"}], history=None) is None


@pytest.mark.parametrize("history", [
    [{"role": "assistant", "content": None}],
    [{"role": "assistant", "content": [{"type": "text", "text": "?"}]}],
    ["not-a-dict"],
])
def test_an_unreadable_last_assistant_turn_fails_towards_the_normal_loop(history):
    assert classify_fast_path("ok", history=history) is None


# ── surface scoping ───────────────────────────────────────────────────────


def test_only_telegram_dms_are_routed():
    assert fast_path_reason("hi", platform_key="telegram", chat_type="dm", history=None) == "social"
    assert fast_path_reason("hi", platform_key="telegram", chat_type="private", history=None) == "social"
    assert fast_path_reason("hi", platform_key="telegram", chat_type="group", history=None) is None
    assert fast_path_reason("hi", platform_key="slack", chat_type="dm", history=None) is None
    assert fast_path_reason("hi", platform_key=None, chat_type="dm", history=None) is None


# ── invariants of the gate itself ─────────────────────────────────────────


def test_the_gate_is_never_broader_than_the_shipped_trivial_prompt_judgement():
    """Eligibility is a strict subset of "this prompt carries no semantic signal".

    ``is_trivial_prompt`` is the single source of truth the memory prefetch already trusts;
    the fast path may only ever narrow it, never reach past it.
    """
    from gateway.run_turn_fast_path import _bare_word

    corpus = [
        "hi", "thanks", "yes", "ok", "yess", "okkk", "deploy the api", "what time is it",
        "run the tests", "hey can you look at this", "sooo", "no worries at all",
    ]
    for text in corpus:
        if classify_fast_path(text, history=ASSISTANT_STATED) is not None:
            assert is_trivial_prompt(text) or is_trivial_prompt(_bare_word(text)), text


def test_directive_and_ack_lexicons_do_not_overlap():
    assert not (_DIRECTIVE_WORDS & _ACK_WORDS)


# ── the note ──────────────────────────────────────────────────────────────


def test_the_note_keeps_the_user_text_and_carries_an_escape_hatch():
    noted = apply_fast_path_note("hi", "social", chat_id="123")
    assert noted.endswith("hi")
    assert noted.startswith(FAST_PATH_NOTE)
    # Fall through and say so: the model must never silently drop work it needs a tool for.
    assert "ignore this note" in FAST_PATH_NOTE
    assert "say plainly what you are doing" in FAST_PATH_NOTE


def test_the_note_never_publishes_a_raw_chat_id(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="gateway.run_turn"):
        apply_fast_path_note("hi", "social", chat_id="998877")
    assert "998877" not in caplog.text
    assert "fast_path_turn" in caplog.text


# ── wiring: the real _prepare_turn_message seam ───────────────────────────


def _turn_runner(message, *, chat_type="dm", platform=None):
    """A real TurnRunner over a stub gateway runner, as test_subagent_failure_notice does."""
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    class _StubGatewayRunner:
        session_store = None

        def _adapter_for_source(self, source):
            return None

    class _Source:
        def __init__(self):
            self.platform = platform or Platform.TELEGRAM
            self.chat_id = "4242"
            self.chat_type = chat_type

    ctx = TurnContext(
        source=_Source(), message=message, session_key="telegram:4242",
        history=[], _run_still_current=lambda: True,
    )
    return TurnRunner(_StubGatewayRunner(), ctx), ctx


def test_an_ordinary_dm_turn_gets_the_note_without_polluting_the_transcript():
    runner, ctx = _turn_runner("hi")
    persist_override, _ = runner._prepare_turn_message(list(ASSISTANT_STATED))

    assert ctx.message.startswith(FAST_PATH_NOTE)  # what the model sees
    assert persist_override == "hi"  # what the transcript keeps


def test_a_turn_that_needs_tools_reaches_the_model_untouched():
    runner, ctx = _turn_runner("deploy the api")
    persist_override, _ = runner._prepare_turn_message(list(ASSISTANT_STATED))

    assert ctx.message == "deploy the api"
    assert persist_override is None


def test_an_ack_answering_a_proposal_reaches_the_model_untouched():
    runner, ctx = _turn_runner("yes")
    runner._prepare_turn_message(list(ASSISTANT_ASKED))
    assert ctx.message == "yes"


def test_a_telegram_group_turn_is_not_routed():
    runner, ctx = _turn_runner("hi", chat_type="group")
    runner._prepare_turn_message(list(ASSISTANT_STATED))
    assert ctx.message == "hi"


def test_an_interrupted_turn_keeps_its_tools_even_for_a_bare_ack():
    """A recovery note means there is state to reconcile; that is not a social reply."""
    runner, ctx = _turn_runner("ok")
    interrupted = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "partial output"},
    ]
    runner._prepare_turn_message(interrupted)

    assert FAST_PATH_NOTE not in ctx.message
