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
    FAST_PATH_OUTCOME_MARKER,
    _ACK_WORDS,
    _ACTION_VERBS,
    _DIRECTIVE_WORDS,
    _NON_OBJECT_WORDS,
    apply_fast_path_note,
    classify_fast_path,
    fast_path_reason,
    is_fast_path_enabled,
    log_fast_path_outcome,
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


@pytest.mark.parametrize("text", ["ya", "oke", "okey", "iya"])
def test_indonesian_acks_take_the_fast_path(text):
    """This deployment's primary user writes mixed Indonesian/English.

    ``ya`` / ``oke`` / ``okey`` are single-token acknowledgements indistinguishable
    from yes/ok. Routing them through the full agentic loop is the 3x-cost failure
    LAB-3 measured. Nothing else in the suite would catch a regression on the
    single most common acknowledgement this user sends.
    """
    assert classify_fast_path(text, history=ASSISTANT_STATED) == "ack"
    assert classify_fast_path(text, history=None) == "ack"


@pytest.mark.parametrize("text", ["yup", "fine", "good", "not really"])
def test_english_acks_that_were_missing_from_the_first_lexicon(text):
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


@pytest.mark.parametrize("text", ["yes", "yess", "ok", "sure", "yeah", "got it", "lgtm",
                                  "ya", "oke", "okey", "yup", "fine", "good", "not really"])
@pytest.mark.parametrize("history", [ASSISTANT_ASKED, ASSISTANT_QUESTION])
def test_an_ack_answering_a_proposal_is_a_go_ahead_and_keeps_its_tools(text, history):
    """The failure mode this gate exists to avoid: "yes" to "shall I deploy?" is real work."""
    assert classify_fast_path(text, history=history) is None


@pytest.mark.parametrize("text", sorted(_DIRECTIVE_WORDS))
def test_directive_words_never_take_the_fast_path(text):
    for history in (None, ASSISTANT_STATED, ASSISTANT_ASKED):
        assert classify_fast_path(text, history=history) is None


def test_a_timestamp_rendered_message_is_still_classified():
    """Production prepends a timestamp render to every inbound user message.

    ``gateway.message_timestamps.enabled`` is on for this deployment, so the classifier is
    handed ``[Thu 2026-09-17 21:54:28 WIB] hi``, not ``hi``. That leading ``[`` used to read
    as a machine work trigger, which meant *every* live message was judged not-ordinary and
    the fast path never fired in production — while every test here passed, because every
    test passes a clean string.
    """
    from zoneinfo import ZoneInfo

    from gateway.message_timestamps import render_user_content_with_timestamp

    tz = ZoneInfo("Asia/Jakarta")
    for text, expected in (("hi", "social"), ("ok", "ack"), ("thanks", "social"),
                           ("oke nice", "ack"), ("ya", "ack")):
        rendered = render_user_content_with_timestamp(text, 1758113668.0, tz=tz)
        assert rendered.startswith("[") and rendered.endswith(text), rendered
        assert classify_fast_path(rendered, history=ASSISTANT_STATED) == expected, rendered


def test_stripping_the_timestamp_does_not_blunt_the_machine_trigger_guard():
    """The one test that keeps the timestamp fix from becoming a hole.

    Stripping the render must leave a real bracketed machine notice still rejected.
    """
    from zoneinfo import ZoneInfo

    from gateway.message_timestamps import render_user_content_with_timestamp

    tz = ZoneInfo("Asia/Jakarta")
    for text in (
        "[ASYNC DELEGATION BATCH 3] check the results",
        "[IMPORTANT: Background process completed] report to user",
        "[cron] nightly backup finished",
    ):
        rendered = render_user_content_with_timestamp(text, 1758113668.0, tz=tz)
        assert classify_fast_path(rendered, history=None) is None, rendered


@pytest.mark.parametrize("text", ["done", "done?"])
def test_done_stays_off_the_fast_path_deliberately(text):
    """Deliberate, not a miss. Do not "fix" this.

    A bare ``done`` from a user usually reports a finished step and expects the
    agent to take the next one. Routing it onto the no-tool path would drop that
    work. Trailing ``?`` is decoration: ``_bare_word`` strips it, so ``done?``
    is the same token.
    """
    for history in (None, ASSISTANT_STATED, ASSISTANT_ASKED):
        assert classify_fast_path(text, history=history) is None


@pytest.mark.parametrize("text", ["test", "this", "ping", "hry"])
def test_ambiguous_probes_stay_off_the_fast_path(text):
    """Genuinely ambiguous: a liveness probe, the start of a task, or ``hurry``.

    Leave them on the full loop. The fall-through note is not a substitute for
    knowing which one they are.
    """
    assert classify_fast_path(text, history=ASSISTANT_STATED) is None
    assert classify_fast_path(text, history=None) is None


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


# ── structural guards: shape, not a word list ─────────────────────────────
# Each case is a *shape* a first-token ack lexicon would misroute. None of
# these strings are the review corpus; a fix that only special-cases those
# literals will fail a fresh set of the same shapes.


@pytest.mark.parametrize("text", [
    "yeah ship the release",          # ack + object, no listed action verb
    "sure, publish the notes",        # punctuation between ack and object
    "oke lanjutkan",                  # Indonesian ack + attached verb
    "go ahead then",                  # directive phrase + extra token
    "yep, once you finish",           # ack + clause
])
def test_an_ack_with_an_object_is_a_command(text):
    """``yes`` is an ack. ``yes <anything>`` is an order. No history needed."""
    for history in (None, ASSISTANT_STATED, ASSISTANT_ASKED):
        assert classify_fast_path(text, history=history) is None


@pytest.mark.parametrize("text", [
    "commit the staged files",
    "review the latest diff",
    "install the missing wheel",
    "please build a fresh image",
    "kindly delete the stale branch",
])
def test_an_action_verb_is_never_ordinary(text):
    """Independent of the ack list: a verb of action is a command."""
    assert classify_fast_path(text, history=ASSISTANT_STATED) is None
    assert classify_fast_path(text, history=None) is None


@pytest.mark.parametrize("text", [
    "https://github.com/nousresearch/hermes-agent/pull/1",
    "www.example.org/status",
    "see https://example.com/a/b and say hi",
])
def test_a_url_is_never_ordinary(text):
    assert classify_fast_path(text, history=ASSISTANT_STATED) is None


@pytest.mark.parametrize("text", [
    "[CRON COMPLETE] inspect the logs",
    "[WATCHDOG: child exited] send the summary",
    "[ok]",  # stripping [] must not recover a bare ack
])
def test_a_bracketed_system_prefix_is_never_ordinary(text):
    assert classify_fast_path(text, history=ASSISTANT_STATED) is None
    assert classify_fast_path(text, history=None) is None


@pytest.mark.parametrize("text", [
    "what time is it in Lisbon",
    "how's the rollout going",
    "why did the job fail",
    "can you look at the leftover invoices",
])
def test_a_consultative_question_stays_on_the_full_loop(text):
    """Needs the world or named state. ``ok?`` is decoration, not this."""
    assert classify_fast_path(text, history=ASSISTANT_STATED) is None
    assert classify_fast_path("ok?", history=ASSISTANT_STATED) == "ack"


def test_proposal_context_still_blocks_a_bare_ack():
    """Re-check: even without an object, a yes to 'shall I …?' keeps its tools."""
    assert classify_fast_path("yes", history=ASSISTANT_ASKED) is None
    assert classify_fast_path("oke", history=ASSISTANT_ASKED) is None
    # And an ack+object is out even when the assistant proposed nothing.
    assert classify_fast_path("sure publish it", history=ASSISTANT_STATED) is None


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


def test_every_ordinary_verdict_is_justified_by_a_known_opening_or_a_social_lexeme():
    """Social and ack verdicts must be traceable to a closed vocabulary.

    The object-free vocabulary is what lets ``oke nice`` and ``wkwkwkw so cute`` be
    ordinary without letting ``ok make it faster`` be. If a verdict ever stops being
    justifiable this way, the exemption has grown into a hole.
    """
    from gateway.run_turn_fast_path import _LAUGH_RE, _SOCIAL_WORDS, _bare_word, _known_leading_len, _tokens

    corpus = [
        "hi", "thanks", "yes", "ok", "yess", "okkk", "ya", "oke", "okey",
        "deploy the api", "what time is it", "run the tests",
        "hey can you look at this", "sooo", "no worries at all",
        "lol", "hmm", "huh", "ey", "np", "yeayy", "wkwkwkw so cute",
        "oke nice", "yeahh i am", "good morning", "haha",
    ]
    for text in corpus:
        kind = classify_fast_path(text, history=ASSISTANT_STATED)
        if kind == "social":
            bare = _bare_word(text)
            assert (
                is_trivial_prompt(text)
                or is_trivial_prompt(bare)
                or bare in _SOCIAL_WORDS
                or _LAUGH_RE.match(bare)
                or _known_leading_len(_tokens(text))  # known opening + object-free tail
            ), text
        elif kind == "ack":
            toks = _tokens(text)
            n = _known_leading_len(toks)
            assert n and " ".join(toks[:n]) in _ACK_WORDS, text
        else:
            assert kind is None, text


def test_directive_and_ack_lexicons_do_not_overlap():
    assert not (_DIRECTIVE_WORDS & _ACK_WORDS)
    assert not (_ACTION_VERBS & _ACK_WORDS)
    assert not (_ACTION_VERBS & _DIRECTIVE_WORDS)
    assert not (_ACTION_VERBS & _NON_OBJECT_WORDS), "an exempt word cannot also be a verb"
    assert not (_DIRECTIVE_WORDS & _NON_OBJECT_WORDS)


@pytest.mark.parametrize("text", [
    "oke nice",           # ack + evaluative adjective
    "yeahh i am",         # ack + pronoun + copula
    "wkwkwkw so cute",    # laughter + filler + evaluative
    "ok then",
    "fine now",
    "iya it",
])
def test_an_ack_carrying_only_object_free_words_is_still_ordinary(text):
    """Recall recovered by bounding the object rule, not by widening ``_ACK_WORDS``.

    These were falling to the full loop purely because *any* second token used to be
    read as an object. Evaluative adjectives, pronouns, copulas and discourse
    particles cannot carry an instruction, so they do not make it one.
    """
    assert classify_fast_path(text, history=ASSISTANT_STATED) in ("ack", "social")


@pytest.mark.parametrize("text", [
    "yes ship it",              # verb outside _ACTION_VERBS: caught by the object rule
    "sure, send the report",    # ditto
    "ok make it faster",
    "yes take a look",
    "ok open the PR",
    "cool, tell her",
    "ok go",
    "yes go",
    "ok merge it",
    "yes merge it",
])
def test_an_ack_carrying_a_real_object_is_still_a_command(text):
    """The contract that makes the exemption safe.

    ``ship`` / ``send`` / ``make`` / ``look`` / ``open`` / ``tell`` / ``go`` are not in
    ``_ACTION_VERBS``. Nothing but the object rule keeps these off the no-tool path, so
    this is the test that has to fail first if the exemption is ever loosened.
    """
    assert classify_fast_path(text, history=ASSISTANT_STATED) is None
    assert classify_fast_path(text, history=None) is None


def test_the_exempt_vocabulary_is_inert_after_a_known_opening():
    """Every exempt word, appended to ``ok``, must stay ordinary.

    This is the bound on the exemption: it cannot be widened by adding a word that
    happens to appear in a command, because the appended word is the whole input here.
    """
    for word in sorted(_NON_OBJECT_WORDS):
        assert classify_fast_path(f"ok {word}", history=ASSISTANT_STATED) is not None, word


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


def test_disabled_config_stops_the_note_from_being_prepended():
    runner, ctx = _turn_runner("ya")
    ctx.user_config = {"gateway": {"telegram": {"fast_path": False}}}
    runner._prepare_turn_message(list(ASSISTANT_STATED))
    assert ctx.message == "ya"
    assert ctx.fast_path_taken is None


def test_fast_path_defaults_on_when_the_key_is_absent():
    """The gateway does not merge DEFAULT_CONFIG; absence must still mean on."""
    assert is_fast_path_enabled({}) is True
    assert is_fast_path_enabled(None) is True
    assert fast_path_reason(
        "hi", platform_key="telegram", chat_type="dm", history=None, user_config={},
    ) == "social"


def test_fast_path_disabled_from_user_yaml_through_the_gateway_loader(tmp_path):
    """Invariant: set the key in a real config.yaml and the gateway loader honours it.

    ``load_user_config_effective`` is the loader ``_load_gateway_config`` uses — no
    DEFAULT_CONFIG merge. A user who writes ``fast_path: false`` must be able to
    disable the path without a code change.
    """
    from hermes_cli.config_effective import load_user_config_effective

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("gateway:\n  telegram:\n    fast_path: false\n", encoding="utf-8")
    cfg = load_user_config_effective(cfg_path)
    assert is_fast_path_enabled(cfg) is False
    assert fast_path_reason(
        "ya", platform_key="telegram", chat_type="dm",
        history=ASSISTANT_STATED, user_config=cfg,
    ) is None


@pytest.mark.parametrize("raw", [False, "false", "off", "0", "no"])
def test_fast_path_off_tokens(raw):
    assert is_fast_path_enabled({"gateway": {"telegram": {"fast_path": raw}}}) is False


def test_fast_path_outcome_counts_only_this_turns_tool_calls(caplog):
    """A long session must not make a fast-path turn look tool-heavy.

    ``result["messages"]`` is the entire conversation — the gateway slices new rows with
    ``offset = len(agent_history)`` itself. Counting all of it reported the session's lifetime
    tool calls: a live turn that used no tools at all logged ``tool_calls=255``. The turn
    boundary is the history length the turn started with.
    """
    import logging

    history = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": f"old{i}"}]}
        for i in range(120)
    ]
    this_turn = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "new1"}]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    result = {"api_calls": 2, "messages": history + this_turn}

    with caplog.at_level(logging.INFO, logger="gateway.run_turn"):
        log_fast_path_outcome("social", result, chat_id="4242", since=len(history))

    assert "tool_calls=1" in caplog.text       # this turn, not the session
    assert "tool_calls=121" not in caplog.text  # the session total must never be reported
    assert "api_calls=2" in caplog.text


def test_fast_path_outcome_between_turns_reports_zero_for_a_tool_free_turn(caplog):
    """The live shape that produced ``tool_calls=255``: a fast-path turn with no tools.

    A boundary one row past the whole conversation is what a caller sees when the turn added
    nothing, and it must report 0 rather than crashing or falling back to the session total.
    """
    import logging

    result = {
        "api_calls": 1,
        "messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "old"}]}],
    }
    with caplog.at_level(logging.INFO, logger="gateway.run_turn"):
        log_fast_path_outcome("social", result, chat_id="1", since=len(result["messages"]))

    assert "tool_calls=0" in caplog.text


def test_fast_path_outcome_records_call_counts(caplog):
    import logging

    result = {
        "api_calls": 1,
        "messages": [
            {"role": "user", "content": "ya"},
            {"role": "assistant", "content": "sip"},
        ],
    }
    with caplog.at_level(logging.INFO, logger="gateway.run_turn"):
        log_fast_path_outcome("ack", result, chat_id="4242")
    assert FAST_PATH_OUTCOME_MARKER in caplog.text
    assert "api_calls=1" in caplog.text
    assert "tool_calls=0" in caplog.text
    assert "4242" not in caplog.text


def test_fast_path_outcome_counts_tool_calls_from_assistant_rows(caplog):
    import logging

    result = {
        "api_calls": 3,
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}, {"id": "2"}]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "done", "tool_calls": [{"id": "3"}]},
        ],
    }
    with caplog.at_level(logging.INFO, logger="gateway.run_turn"):
        log_fast_path_outcome("ack", result, chat_id="1")
    assert "api_calls=3" in caplog.text
    assert "tool_calls=3" in caplog.text
