"""Content-aware Telegram reaction styles: receipt vs tone, and the reaction-only guard.

Two things are being pinned here, and they fail differently:

* **Glyph policy** — the default style must keep emitting exactly the glyphs Telegram already
  sent (👀 then 👍/👎), and the content style may only ever emit a glyph Telegram curates,
  because a rejected reaction emoji is swallowed by the adapter (debug log, no raise) and would
  therefore fail *silently*.
* **The reaction-only guard** — a reaction may be the whole reply only for a purely social
  message. Every question, implicit ask, mixed, work-bearing, decision-pending, sensitive, or
  merely unrecognized message must resolve to text. This is the failure mode that matters: an
  assistant that answers a real question with a heart and no words is worse than one that
  always sends a thumbs up.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.telegram_reaction_style as style
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource

EYES, THUMBS_UP, THUMBS_DOWN = "\U0001f440", "\U0001f44d", "\U0001f44e"


def _make_adapter(**env):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    return adapter


def _make_event(text="hello", chat_id="123", message_id="456"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="private",
            user_id="42", user_name="TestUser"),
        message_id=message_id,
    )


def _reactions_on(monkeypatch, style_value=None):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    if style_value is None:
        monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_REACTION_STYLE", style_value)


def _sent_glyph(adapter):
    kwargs = adapter._bot.set_message_reaction.call_args.kwargs
    return kwargs.get("reaction")


def _verdict_reason(result):
    return json.dumps(result)


# ── style normalization: the default is the current behaviour ────────────────────────


@pytest.mark.parametrize("value", [None, "", "   ", "LIFECYCLE", "LifeCycle", 0, 1,
                                   "nonsense", "contentt", [], object()])
def test_unknown_style_values_normalize_to_the_default_style(value):
    assert style.normalize_style(value) == style.STYLE_LIFECYCLE


@pytest.mark.parametrize("value", ["content", "CONTENT", " content ", "content-aware",
                                   "content_aware", "aware", "tone"])
def test_content_style_is_recognized_in_the_spellings_a_human_types(value):
    assert style.normalize_style(value) == style.STYLE_CONTENT


def test_default_style_constant_is_the_lifecycle_style():
    assert style.DEFAULT_STYLE == style.STYLE_LIFECYCLE
    assert set(style.STYLES) == {style.STYLE_LIFECYCLE, style.STYLE_CONTENT}


# ── the regression guarantee: default style emits the original glyphs ────────────────

_ALL_MESSAGES = [
    "hello there", "thank you so much 🙏", "haha that's hilarious 😂", "what's the status?",
    "the build is broken", "deploy it now", "", None, "love you ❤️", "congrats 🎉",
]


@pytest.mark.parametrize("message", _ALL_MESSAGES)
def test_lifecycle_style_keeps_the_original_start_and_success_glyphs(message):
    assert style.start_reaction(style.STYLE_LIFECYCLE, message) == EYES
    assert style.success_reaction(style.STYLE_LIFECYCLE, message) == THUMBS_UP


@pytest.mark.parametrize("message", _ALL_MESSAGES)
def test_no_style_or_an_unreadable_style_keeps_the_original_success_glyph(message):
    assert style.success_reaction(None, message) == THUMBS_UP
    assert style.success_reaction("nonsense", message) == THUMBS_UP


@pytest.mark.parametrize("message", _ALL_MESSAGES)
def test_the_start_glyph_stays_the_ack_receipt_under_every_style(message):
    """👀 says "checking" — a receipt, so no style makes it content-aware."""
    assert style.start_reaction(style.STYLE_CONTENT, message) == EYES
    assert style.start_reaction(None, message) == EYES


@pytest.mark.parametrize("style_value", [None, style.STYLE_LIFECYCLE, style.STYLE_CONTENT])
def test_failure_always_reports_the_failure_glyph(style_value):
    for message in _ALL_MESSAGES:
        assert style.failure_reaction(style_value, message) == THUMBS_DOWN


# ── tone selection under the content style ───────────────────────────────────────────

_TONE_CASES = [
    ("thank you so much 🙏", style.TONE_GRATITUDE, "🙏", style.JOB_TONE),
    ("haha that's hilarious 😂", style.TONE_AMUSEMENT, "🤣", style.JOB_TONE),
    ("love you ❤️", style.TONE_AFFECTION, "❤", style.JOB_TONE),
    ("so sad 😭", style.TONE_SYMPATHY, "😭", style.JOB_TONE),
    ("nice work on this 👏", style.TONE_PRAISE, "👏", style.JOB_TONE),
    ("congrats on the launch 🎉", style.TONE_CELEBRATION, "🎉", style.JOB_TONE),
    ("good morning!", style.TONE_GREETING, "🤗", style.JOB_TONE),
    ("ok sounds good", style.TONE_ACKNOWLEDGEMENT, "👌", style.JOB_TONE),
    # Receipt side of the split: same glyphs the lifecycle style sends.
    ("what's the status?", style.TONE_QUESTION, THUMBS_UP, style.JOB_RECEIPT),
    ("the build is broken", style.TONE_TROUBLE, THUMBS_UP, style.JOB_RECEIPT),
    ("urgent: prod is down", style.TONE_URGENT, THUMBS_UP, style.JOB_RECEIPT),
    ("I moved the meeting to 4pm", style.TONE_ROUTINE, THUMBS_UP, style.JOB_RECEIPT),
    # This DM mixes Indonesian; the classifier has to read the affection it actually gets.
    ("sayaang", style.TONE_AFFECTION, "❤", style.JOB_TONE),
    ("cintaku", style.TONE_AFFECTION, "❤", style.JOB_TONE),
    ("kangen kamu", style.TONE_AFFECTION, "❤", style.JOB_TONE),
    ("makasih!", style.TONE_GRATITUDE, "🙏", style.JOB_TONE),
    ("terima kasih", style.TONE_GRATITUDE, "🙏", style.JOB_TONE),
    ("keren banget", style.TONE_PRAISE, "👏", style.JOB_TONE),
    ("wkwkwk lucu", style.TONE_AMUSEMENT, "🤣", style.JOB_TONE),
    ("halo", style.TONE_GREETING, "🤗", style.JOB_TONE),
    # "selamat pagi" is a greeting; bare "selamat" is a celebration — the lookahead splits them.
    ("selamat pagi", style.TONE_GREETING, "🤗", style.JOB_TONE),
    ("selamat ya", style.TONE_CELEBRATION, "🎉", style.JOB_TONE),
    ("oke siap", style.TONE_ACKNOWLEDGEMENT, "👌", style.JOB_TONE),
    ("kasihan", style.TONE_SYMPATHY, "😭", style.JOB_TONE),
    ("gagal terus", style.TONE_TROUBLE, THUMBS_UP, style.JOB_RECEIPT),
    # Stretched spellings are how excitement actually arrives ("yeayyy"); a tone pattern that
    # only matches the dictionary spelling reads a real celebration as routine prose.
    ("yeayyy", style.TONE_CELEBRATION, "🎉", style.JOB_TONE),
    ("yeay", style.TONE_CELEBRATION, "🎉", style.JOB_TONE),
    ("yesss", style.TONE_CELEBRATION, "🎉", style.JOB_TONE),
    ("asyik", style.TONE_CELEBRATION, "🎉", style.JOB_TONE),
    ("akhirnya", style.TONE_CELEBRATION, "🎉", style.JOB_TONE),
    ("mantaaap", style.TONE_PRAISE, "👏", style.JOB_TONE),
    ("cintaaa", style.TONE_AFFECTION, "❤", style.JOB_TONE),
    ("kangeeeen", style.TONE_AFFECTION, "❤", style.JOB_TONE),
    ("makasi", style.TONE_GRATITUDE, "🙏", style.JOB_TONE),
    ("thanksss", style.TONE_GRATITUDE, "🙏", style.JOB_TONE),
    ("wkwkwkwk", style.TONE_AMUSEMENT, "🤣", style.JOB_TONE),
    ("pagiii", style.TONE_GREETING, "🤗", style.JOB_TONE),
    ("good girl", style.TONE_PRAISE, "👏", style.JOB_TONE),
    ("good bot", style.TONE_PRAISE, "👏", style.JOB_TONE),
    ("okieee", style.TONE_ACKNOWLEDGEMENT, "👌", style.JOB_TONE),
    ("okie", style.TONE_ACKNOWLEDGEMENT, "👌", style.JOB_TONE),
    ("okey", style.TONE_ACKNOWLEDGEMENT, "👌", style.JOB_TONE),
]


@pytest.mark.parametrize("message,tone,glyph,job", _TONE_CASES)
def test_content_style_success_glyph_and_job_per_message(message, tone, glyph, job):
    assert style.classify_content(message) == tone
    assert style.success_reaction(style.STYLE_CONTENT, message) == glyph
    assert style.reaction_job(message) == job


def test_receipt_and_tone_jobs_are_disjoint_and_cover_every_tone():
    assert style.SOCIAL_TONES.isdisjoint(style.RECEIPT_TONES)
    assert style.SOCIAL_TONES == set(style.TONE_GLYPHS)
    assert style.TONE_ROUTINE in style.RECEIPT_TONES
    assert style.TONE_QUESTION in style.RECEIPT_TONES


def test_unrecognized_content_is_a_receipt_never_a_guess_at_tone():
    assert style.classify_content("qxz 77 blorp") == style.TONE_ROUTINE
    assert style.reaction_job("qxz 77 blorp") == style.JOB_RECEIPT
    assert style.success_reaction(style.STYLE_CONTENT, "qxz 77 blorp") == THUMBS_UP


# ── curated glyphs: a rejected emoji fails silently, so nothing uncurated may ship ────


@pytest.mark.parametrize("glyph", [EYES, THUMBS_UP, THUMBS_DOWN, *style.TONE_GLYPHS.values()])
def test_every_selectable_glyph_is_in_telegrams_curated_reaction_set(glyph):
    assert glyph in style.TELEGRAM_REACTION_EMOJI


def test_uncurated_tone_glyph_degrades_to_the_receipt_glyph(monkeypatch):
    """Negative control: with a glyph Telegram would reject, the receipt glyph is used."""
    monkeypatch.setitem(style.TONE_GLYPHS, style.TONE_AFFECTION, "TEXT-LOOKING")
    assert style.success_reaction(style.STYLE_CONTENT, "love you ❤️") == THUMBS_UP
    assert style.tone_emoji(style.TONE_AFFECTION) is None


def test_tone_glyph_for_a_non_social_tone_is_none():
    assert style.tone_emoji(style.TONE_QUESTION) is None
    assert style.tone_emoji(style.TONE_ROUTINE) is None
    assert style.tone_emoji(None) is None


def test_receipt_module_echoes_tone_glyphs_as_glyphs_never_as_text():
    """The receipt must record a glyph, not a payload — one bad table entry would leak text."""
    from gateway.telegram_reaction_receipt import safe_emoji_token

    for tone, glyph in style.TONE_GLYPHS.items():
        assert safe_emoji_token(glyph) == glyph, tone


# ── totality: cosmetics must never break a turn ──────────────────────────────────────


class _Hostile:
    def __str__(self):  # pragma: no cover — exercised through the selectors
        raise RuntimeError("hostile __str__")

    def __bool__(self):
        raise RuntimeError("hostile __bool__")


@pytest.mark.parametrize("value", [None, "", 0, 3.5, _Hostile(), object(), b"bytes",
                                   "x" * 50_000])
def test_glyph_selectors_are_total_over_any_input(value):
    assert style.start_reaction(style.STYLE_CONTENT, value) == EYES
    assert style.success_reaction(style.STYLE_CONTENT, value) in {THUMBS_UP, *style.TONE_GLYPHS.values()}
    assert style.failure_reaction(style.STYLE_CONTENT, value) == THUMBS_DOWN
    assert style.normalize_style(value) in style.STYLES
    assert isinstance(style.classify_content(value), str)


def test_success_glyph_degrades_when_classification_faults(monkeypatch):
    monkeypatch.setattr(style, "classify_content", lambda content: (_ for _ in ()).throw(RuntimeError("boom")))
    assert style.success_reaction(style.STYLE_CONTENT, "love you ❤️") == THUMBS_UP


def test_guard_denies_when_the_guard_itself_faults(monkeypatch):
    monkeypatch.setattr(style, "_IMPLICIT_ASK", SimpleNamespace(search=lambda text: (_ for _ in ()).throw(RuntimeError("boom"))))
    decision = style.decide_reaction_only("love you ❤️", style=style.STYLE_CONTENT)
    assert decision.text_required is True
    assert decision.reason == style.REASON_GUARD_ERROR


# ── the reaction-only guard corpus ───────────────────────────────────────────────────

# Purely social, unambiguous, nothing pending: the reaction IS the reply.
_SOCIAL_CORPUS = [
    ("thank you so much 🙏", style.TONE_GRATITUDE, "🙏"),
    ("thanks!", style.TONE_GRATITUDE, "🙏"),
    ("love you ❤️", style.TONE_AFFECTION, "❤"),
    ("miss you", style.TONE_AFFECTION, "❤"),
    ("haha that's hilarious 😂", style.TONE_AMUSEMENT, "🤣"),
    ("lol", style.TONE_AMUSEMENT, "🤣"),
    ("good morning!", style.TONE_GREETING, "🤗"),
    ("hello there", style.TONE_GREETING, "🤗"),
    ("nice work on this 👏", style.TONE_PRAISE, "👏"),
    ("you're amazing", style.TONE_PRAISE, "👏"),
    ("congrats on the launch 🎉", style.TONE_CELEBRATION, "🎉"),
    ("yay we did it", style.TONE_CELEBRATION, "🎉"),
    ("so sad 😭", style.TONE_SYMPATHY, "😭"),
    ("sorry to hear that", style.TONE_SYMPATHY, "😭"),
    ("ok sounds good", style.TONE_ACKNOWLEDGEMENT, "👌"),
    ("got it", style.TONE_ACKNOWLEDGEMENT, "👌"),
    # The language he actually flirts in: these must be allowed to stand in for a reply.
    ("sayaang", style.TONE_AFFECTION, "❤"),
    ("cintaku", style.TONE_AFFECTION, "❤"),
    ("kangen kamu", style.TONE_AFFECTION, "❤"),
    ("terima kasih!", style.TONE_GRATITUDE, "🙏"),
    ("keren banget", style.TONE_PRAISE, "👏"),
    ("wkwkwk lucu", style.TONE_AMUSEMENT, "🤣"),
    ("halo", style.TONE_GREETING, "🤗"),
    ("oke siap", style.TONE_ACKNOWLEDGEMENT, "👌"),
    ("kasihan", style.TONE_SYMPATHY, "😭"),
    # Stretched excitement, the shape it actually arrives in.
    ("yeayyy", style.TONE_CELEBRATION, "🎉"),
    ("asyik", style.TONE_CELEBRATION, "🎉"),
    ("mantaaap", style.TONE_PRAISE, "👏"),
    ("cintaaa", style.TONE_AFFECTION, "❤"),
    ("thanksss", style.TONE_GRATITUDE, "🙏"),
    ("wkwkwkwk", style.TONE_AMUSEMENT, "🤣"),
    ("pagiii", style.TONE_GREETING, "🤗"),
    ("good girl", style.TONE_PRAISE, "👏"),
    ("good bot", style.TONE_PRAISE, "👏"),
]

# Indonesian work/ask messages. Recognising Indonesian *social* tone created this obligation: a
# guard that only sees English asks would let "sayaang, tolong cek ya" replace a reply it must not.
# The trailing "ya" tag counts as an ask, so "makasih ya" still requires text — conservative by
# design, and the 🙏 glyph is emitted either way.
_INDONESIAN_TEXT_REQUIRED = [
    ("tolong cek dong", style.REASON_IMPLICIT_ASK),
    ("mohon dicek", style.REASON_IMPLICIT_ASK),
    ("dong", style.REASON_IMPLICIT_ASK),
    ("makasih ya", style.REASON_IMPLICIT_ASK),
    ("sayaang ya", style.REASON_IMPLICIT_ASK),
    ("kirim laporan ya", style.REASON_IMPLICIT_ASK),
    ("pagi, tolong kirim invoice", style.REASON_IMPLICIT_ASK),
    ("cek ya", style.REASON_IMPLICIT_ASK),
    ("bisa bantu?", style.REASON_QUESTION_ASKED),
    ("gimana statusnya", style.REASON_QUESTION_ASKED),
    ("kapan rilis", style.REASON_QUESTION_ASKED),
    ("hapus file itu", style.REASON_MIXED_OR_WORK),
    ("bayar tagihan", style.REASON_MIXED_OR_WORK),
    ("ganti password", style.REASON_MIXED_OR_WORK),
    ("transfer uang", style.REASON_MIXED_OR_WORK),
    ("perbaiki bug-nya", style.REASON_MIXED_OR_WORK),
    ("gagal terus", style.REASON_NOT_SOCIAL),
    ("rusak", style.REASON_NOT_SOCIAL),
]

# Everything that must still get words. Grouped by why, so a failure names the rule that broke.
_QUESTION_CORPUS = [
    "what's the status?",
    "why is the gateway down?",
    "is the deploy finished?",
    "how do I rotate the api key?",
    "did you finish the report?",
    "can you check the logs",
]

_IMPLICIT_ASK_CORPUS = [
    "please send me the file",
    "pls share the report",
    "any idea when the build lands",
    "let me know what you find",
    "send me the summary",
]

_MIXED_OR_WORK_CORPUS = [
    ("thanks, the tests pass now", style.REASON_MIXED_OR_WORK),
    ("love it, but the backup failed", style.REASON_MIXED_OR_WORK),
    ("ok, verify the backup first", style.REASON_MIXED_OR_WORK),
    ("thanks — see https://example.com/build/17", style.REASON_MIXED_OR_WORK),
    ("thanks! the file is /tmp/report.md", style.REASON_MIXED_OR_WORK),
    ("haha, the build is broken btw", style.REASON_MIXED_OR_WORK),
    ("thanks for the fix, the tests took 40s", style.REASON_MIXED_OR_WORK),
    ("the build is broken", style.REASON_MIXED_OR_WORK),
    ("deployed to prod", style.REASON_MIXED_OR_WORK),
]

# The arm of the guard the user called out by name: money, credentials, deletion, deploy, and
# anything still pending verification. None of these may be answered with a reaction alone.
_SENSITIVE_CORPUS = [
    ("deployed to prod", style.REASON_MIXED_OR_WORK),
    ("delete the old backups", style.REASON_MIXED_OR_WORK),
    ("the invoice is paid", style.REASON_MIXED_OR_WORK),
    ("the api key expired", style.REASON_MIXED_OR_WORK),
    ("rotate the token", style.REASON_MIXED_OR_WORK),
    ("thanks, deploy it", style.REASON_MIXED_OR_WORK),
]

_NOT_SOCIAL_OR_SILENT_CORPUS = [
    ("urgent: prod is down", style.REASON_NOT_SOCIAL),
    ("502 bad gateway", style.REASON_NOT_SOCIAL),
    ("shift the offsite to thursday", style.REASON_NOT_SOCIAL),
    ("", style.REASON_NO_CONTENT),
    ("   ", style.REASON_NO_CONTENT),
    # Purely social but far longer than a reaction can carry: an essay gets text.
    ("love you " * 40, style.REASON_TOO_LONG),
]


@pytest.mark.parametrize("message,tone,glyph", _SOCIAL_CORPUS)
def test_guard_allows_reaction_only_for_the_social_corpus(message, tone, glyph):
    decision = style.decide_reaction_only(message, style=style.STYLE_CONTENT)
    assert decision.mode == style.REPLY_REACTION_ONLY, _verdict_reason({"message": message, "reason": decision.reason})
    assert decision.allowed is True and decision.text_required is False
    assert decision.reason == style.REASON_SOCIAL
    assert decision.tone == tone
    assert decision.job == style.JOB_TONE
    assert decision.emoji == glyph
    # The guard's glyph is the same one the success reaction uses — one table, two readers.
    assert decision.emoji == style.success_reaction(style.STYLE_CONTENT, message)


@pytest.mark.parametrize("message", _QUESTION_CORPUS)
def test_guard_forces_text_for_every_question(message):
    decision = style.decide_reaction_only(message, style=style.STYLE_CONTENT)
    assert decision.mode == style.REPLY_TEXT
    assert decision.text_required is True
    assert decision.reason == style.REASON_QUESTION_ASKED
    assert decision.emoji is None


@pytest.mark.parametrize("message", _IMPLICIT_ASK_CORPUS)
def test_guard_forces_text_for_every_implicit_ask(message):
    decision = style.decide_reaction_only(message, style=style.STYLE_CONTENT)
    assert decision.mode == style.REPLY_TEXT
    assert decision.reason == style.REASON_IMPLICIT_ASK


@pytest.mark.parametrize("message,reason", _MIXED_OR_WORK_CORPUS)
def test_guard_forces_text_for_mixed_messages(message, reason):
    decision = style.decide_reaction_only(message, style=style.STYLE_CONTENT)
    assert decision.mode == style.REPLY_TEXT
    assert decision.reason == reason


@pytest.mark.parametrize("message,reason", _NOT_SOCIAL_OR_SILENT_CORPUS)
def test_guard_forces_text_for_work_bearing_and_unrecognized_messages(message, reason):
    decision = style.decide_reaction_only(message, style=style.STYLE_CONTENT)
    assert decision.mode == style.REPLY_TEXT
    assert decision.reason == reason


@pytest.mark.parametrize("message,reason", _SENSITIVE_CORPUS)
def test_guard_forces_text_for_money_credentials_deletion_and_deploy(message, reason):
    decision = style.decide_reaction_only(message, style=style.STYLE_CONTENT)
    assert decision.mode == style.REPLY_TEXT
    assert decision.text_required is True
    assert decision.reason == reason


@pytest.mark.parametrize("message,reason", _INDONESIAN_TEXT_REQUIRED)
def test_guard_stays_hard_in_indonesian(message, reason):
    """Recognising Indonesian tone must not open a hole in the guard: asks and work still force text."""
    decision = style.decide_reaction_only(message, style=style.STYLE_CONTENT)
    assert decision.mode == style.REPLY_TEXT, f"{message!r} was allowed to replace a reply"
    assert decision.text_required is True
    assert decision.reason == reason


def test_no_social_corpus_message_is_ever_denied_for_an_unexpected_reason():
    """The social set must be *allowed*, not merely non-denied for a reason we did not intend."""
    reasons = {style.decide_reaction_only(m, style=style.STYLE_CONTENT).reason
               for m, _, _ in _SOCIAL_CORPUS}
    assert reasons == {style.REASON_SOCIAL}


def test_every_question_corpus_message_also_survives_a_stricter_caller():
    """Vetoes stack: a question stays text-required with any combination of flags set."""
    for message in [*_QUESTION_CORPUS, *_IMPLICIT_ASK_CORPUS]:
        for kwargs in ({}, {"confident": False}, {"reports_work": True},
                       {"pending_decision": True}, {"sensitive": True}, {"needs_next_step": True}):
            decision = style.decide_reaction_only(message, style=style.STYLE_CONTENT, **kwargs)
            assert decision.text_required is True, (message, kwargs, decision.reason)


# ── the guard under the default style: nothing changes ───────────────────────────────


@pytest.mark.parametrize("message", [m for m, _, _ in _SOCIAL_CORPUS] + _QUESTION_CORPUS)
@pytest.mark.parametrize("style_value", [None, "", "nonsense", style.STYLE_LIFECYCLE])
def test_reaction_only_is_refused_under_the_default_style(message, style_value):
    decision = style.decide_reaction_only(message, style=style_value)
    assert decision.mode == style.REPLY_TEXT
    assert decision.reason == style.REASON_STYLE_DISABLED
    assert decision.emoji is None


def test_the_default_style_denial_never_depends_on_the_content():
    for message in [m for m, _, _ in _SOCIAL_CORPUS]:
        assert style.decide_reaction_only(message).reason == style.REASON_STYLE_DISABLED


# ── caller vetoes can only ever tighten the guard ────────────────────────────────────


@pytest.mark.parametrize("flag,reason", [
    ("pending_decision", style.REASON_PENDING_DECISION),
    ("reports_work", style.REASON_WORK_REPORT),
    ("needs_next_step", style.REASON_NEXT_STEP),
    ("sensitive", style.REASON_SENSITIVE),
])
def test_each_veto_flag_forces_text_on_an_otherwise_social_message(flag, reason):
    decision = style.decide_reaction_only("thank you so much 🙏", style=style.STYLE_CONTENT, **{flag: True})
    assert decision.mode == style.REPLY_TEXT
    assert decision.reason == reason


def test_uncertainty_forces_text():
    decision = style.decide_reaction_only("thank you so much 🙏", style=style.STYLE_CONTENT, confident=False)
    assert decision.mode == style.REPLY_TEXT
    assert decision.reason == style.REASON_UNCERTAIN


def test_flags_are_veto_only_and_cannot_grant_permission():
    """Omitting every flag still denies a question: the text checks run regardless."""
    question = style.decide_reaction_only("is the backup done?", style=style.STYLE_CONTENT)
    flagged = style.decide_reaction_only("is the backup done?", style=style.STYLE_CONTENT,
                                         confident=True, pending_decision=False, reports_work=False,
                                         needs_next_step=False, sensitive=False)
    assert question.text_required is True and flagged.text_required is True


def test_reaction_only_decision_is_total_over_hostile_input():
    for value in [None, "", 0, _Hostile(), object(), b"bytes"]:
        decision = style.decide_reaction_only(value, style=style.STYLE_CONTENT)
        assert decision.mode in {style.REPLY_TEXT, style.REPLY_REACTION_ONLY}


# ── adapter integration: glyphs, opt-in claim, bridging ──────────────────────────────


@pytest.mark.asyncio
async def test_adapter_default_install_emits_the_original_glyph_pair(monkeypatch):
    _reactions_on(monkeypatch)
    adapter = _make_adapter()
    event = _make_event(text="thank you so much 🙏")
    await adapter.on_processing_start(event)
    assert _sent_glyph(adapter) == EYES
    adapter._bot.set_message_reaction.reset_mock()
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert _sent_glyph(adapter) == THUMBS_UP


@pytest.mark.asyncio
async def test_adapter_content_style_swaps_a_social_success_glyph(monkeypatch):
    _reactions_on(monkeypatch, "content")
    adapter = _make_adapter()
    event = _make_event(text="thank you so much 🙏")
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert _sent_glyph(adapter) == "🙏"


@pytest.mark.asyncio
async def test_adapter_content_style_keeps_the_receipt_glyph_for_a_question(monkeypatch):
    _reactions_on(monkeypatch, "content")
    adapter = _make_adapter()
    event = _make_event(text="what's the status?")
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert _sent_glyph(adapter) == THUMBS_UP


@pytest.mark.asyncio
async def test_adapter_content_style_keeps_the_failure_glyph(monkeypatch):
    _reactions_on(monkeypatch, "content")
    adapter = _make_adapter()
    event = _make_event(text="love you ❤️")
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
    assert _sent_glyph(adapter) == THUMBS_DOWN


@pytest.mark.asyncio
async def test_adapter_reads_the_style_from_per_profile_extra_when_env_is_absent(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    adapter = _make_adapter()
    adapter.config.extra = {"reaction_style": "content"}
    await adapter.on_processing_complete(_make_event(text="haha 😂"), ProcessingOutcome.SUCCESS)
    assert _sent_glyph(adapter) == "🤣"


@pytest.mark.asyncio
async def test_plain_react_is_unchanged_and_carries_no_verdict(monkeypatch):
    """Opt-in check: without the claim, the react result is byte-for-byte what it was."""
    _reactions_on(monkeypatch, "content")
    adapter = _make_adapter()
    result = await adapter.add_reaction("123", emoji="👍", message_id="456")
    assert result == {"success": True, "message_id": "456"}


@pytest.mark.asyncio
async def test_react_only_claim_is_refused_for_a_question(monkeypatch):
    _reactions_on(monkeypatch, "content")
    adapter = _make_adapter()
    result = await adapter.add_reaction("123", emoji="❤", message_id="456",
                                        stand_in_for_reply=True, content="what's the status?")
    assert result["success"] is True              # the reaction itself still goes out
    assert result["reaction_only"] is False
    assert result["text_required"] is True
    assert result["reason"] == style.REASON_QUESTION_ASKED
    assert adapter._bot.set_message_reaction.call_args.kwargs["reaction"] == "❤"


@pytest.mark.asyncio
async def test_react_only_claim_is_refused_under_the_default_style(monkeypatch):
    _reactions_on(monkeypatch)
    adapter = _make_adapter()
    result = await adapter.add_reaction("123", emoji="❤", message_id="456",
                                        stand_in_for_reply=True, content="thank you 🙏")
    assert result["text_required"] is True
    assert result["reason"] == style.REASON_STYLE_DISABLED


@pytest.mark.asyncio
async def test_react_only_claim_is_granted_for_a_purely_social_message(monkeypatch):
    _reactions_on(monkeypatch, "content")
    adapter = _make_adapter()
    result = await adapter.add_reaction("123", emoji="🙏", message_id="456",
                                        stand_in_for_reply=True, content="thank you so much 🙏")
    assert result["reaction_only"] is True
    assert result["text_required"] is False
    assert result["reason"] == style.REASON_SOCIAL
    assert result["tone"] == style.TONE_GRATITUDE
    assert result["job"] == style.JOB_TONE


@pytest.mark.asyncio
async def test_react_only_claim_uses_the_text_the_adapter_saw_for_that_chat(monkeypatch):
    _reactions_on(monkeypatch, "content")
    adapter = _make_adapter()
    await adapter.on_processing_start(_make_event(text="please send me the file", chat_id="123"))
    result = await adapter.add_reaction("123", emoji="👀", message_id="456", stand_in_for_reply=True)
    assert result["text_required"] is True
    assert result["reason"] == style.REASON_IMPLICIT_ASK


@pytest.mark.asyncio
async def test_react_only_claim_for_an_unseen_chat_fails_toward_text(monkeypatch):
    _reactions_on(monkeypatch, "content")
    adapter = _make_adapter()
    result = await adapter.add_reaction("999", emoji="👍", message_id="456", stand_in_for_reply=True)
    assert result["text_required"] is True
    assert result["reason"] == style.REASON_NO_CONTENT


@pytest.mark.asyncio
async def test_react_only_claim_is_judged_per_chat(monkeypatch):
    _reactions_on(monkeypatch, "content")
    adapter = _make_adapter()
    await adapter.on_processing_start(_make_event(text="love you ❤️", chat_id="111"))
    await adapter.on_processing_start(_make_event(text="is the build done?", chat_id="222"))
    social = await adapter.add_reaction("111", emoji="❤", message_id="1", stand_in_for_reply=True)
    work = await adapter.add_reaction("222", emoji="👀", message_id="2", stand_in_for_reply=True)
    assert social["reaction_only"] is True and work["text_required"] is True


@pytest.mark.asyncio
async def test_the_inbound_text_memory_stays_bounded(monkeypatch):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    _reactions_on(monkeypatch, "content")
    adapter = _make_adapter()
    limit = TelegramAdapter.REACTION_INBOUND_MEMORY
    for index in range(limit + 20):
        await adapter.on_processing_start(_make_event(text=f"hi {index}", chat_id=str(index)))
    assert len(adapter._reaction_inbound_text) <= limit
    assert adapter._inbound_text("0") == ""          # oldest evicted
    assert adapter._inbound_text(str(limit + 19)) == f"hi {limit + 19}"


def test_yaml_bridge_seeds_the_style_into_extra_and_env(monkeypatch):
    from plugins.platforms.telegram.adapter import _apply_yaml_config

    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    extras = _apply_yaml_config({}, {"reaction_style": "content"})
    assert extras == {"reaction_style": "content"}
    assert __import__("os").environ["TELEGRAM_REACTION_STYLE"] == "content"


def test_yaml_bridge_does_not_invent_a_style_key():
    from plugins.platforms.telegram.adapter import _apply_yaml_config

    extras = _apply_yaml_config({}, {"reactions": True})
    assert extras == {"reactions": True}


def test_adapter_advertises_reaction_only_support():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    assert TelegramAdapter.supports_reaction_only_reply is True
