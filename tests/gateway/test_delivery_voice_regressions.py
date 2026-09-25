"""Conversation-sequence guards for final user-visible delivery voice."""

import re

import pytest

from gateway.delivery_voice import final_delivery_voice_check
from gateway.run import _sanitize_gateway_final_response
from gateway.platforms.base import Platform


def test_interrupted_turn_correction_has_one_clean_final_delivery():
    """The interrupted draft is control-plane noise; only the correction is deliverable."""
    raw_turns = ["[response interrupted]", "The corrected answer is ready."]
    delivered = [final_delivery_voice_check(reply) for reply in raw_turns]
    delivered = [reply for reply in delivered if reply]

    assert delivered == ["The corrected answer is ready."]
    assert all("[response interrupted]" not in reply for reply in delivered)
    assert all(not any(word in reply.casefold() for word in ("queue", "cancel", "status")) for reply in delivered)


@pytest.mark.parametrize(
    "raw",
    [
        "[response interrupted]",
        "Operation interrupted.",
        "Operation cancelled.",
        "Operation canceled.",
        "[response interrupted]\nOperation interrupted.",
    ],
)
def test_internal_markers_are_fully_suppressed(raw):
    assert final_delivery_voice_check(raw) == ""


def test_internal_marker_line_does_not_eat_real_prose():
    raw = "[response interrupted]\nSleep well, sayang."
    assert final_delivery_voice_check(raw) == "Sleep well, sayang."


def test_em_dash_and_en_dash_are_replaced_without_flattening_voice():
    assert "—" not in final_delivery_voice_check("Mimi — your fix is tiny.")
    assert "–" not in final_delivery_voice_check("Mimi – your fix is tiny.")
    assert final_delivery_voice_check("Mimi — your fix is tiny.") == "Mimi, your fix is tiny."


EXAMPLES = [
    ("Got it — the tea is ready.", "The tea is ready."),
    ("Understood, I saved it.", "I saved it."),
    ("Here's what I found: logs are clean.", "Logs are clean."),
    ("What would you like to do next? The deploy is ready.", "The deploy is ready."),
    ("Let me know if you need anything else. Sleep now.", "Sleep now."),
    ("Mimi — your fix is tiny.", "Mimi, your fix is tiny."),
    ("Got it, cariño. We can keep the joke.", "Cariño, we can keep the joke."),
    ("Understood — the command is safe.", "The command is safe."),
    ("Here's what I found: the token is absent.", "The token is absent."),
    ("Got it: aku cek dulu.", "Aku cek dulu."),
    ("Let me know if you need anything else. Sleep now.", "Sleep now."),
    ("The patch is small — tests cover it.", "The patch is small, tests cover it."),
    ("Understood! Error is isolated.", "Error is isolated."),
    ("Got it — restart is not needed.", "Restart is not needed."),
    ("Here's what I found: done, sayang.", "Done, sayang."),
]


def _without_persona(text: str) -> str:
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"\b(?:ningy|mimi|sayang|cariño|babe)\b", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" .,!?;:").lower()


@pytest.mark.parametrize("before, after", EXAMPLES)
def test_before_after_examples_keep_content_without_ai_scaffolding(before, after):
    checked = _sanitize_gateway_final_response(Platform.TELEGRAM, before)
    assert "—" not in checked
    assert _without_persona(checked) == _without_persona(after)


@pytest.mark.parametrize(
    "kind, reply",
    [
        ("casual", "The weather is soft today."),
        ("playful", "Tiny bug, big drama."),
        ("affectionate", "Sleep well, sayang."),
        ("technical", "The retry loop stops at three."),
        ("correction", "The earlier path was wrong; this one is right."),
        ("uncertainty", "I cannot verify that from here."),
        ("error", "The provider is unavailable."),
        ("completion", "Done. Tests pass."),
        ("restart", "The old turn is closed; this one starts clean."),
    ],
)
def test_conversation_sequence_does_not_turn_into_generic_assistant_output(kind, reply):
    checked = _sanitize_gateway_final_response(Platform.TELEGRAM, reply)
    assert checked
    assert _without_persona(checked) == _without_persona(reply)
    assert not re.search(
        r"\b(?:got it|understood|here(?:'s| is) what i found|what would you like to do next|let me know if you need anything else)\b",
        checked,
        re.I,
    ), kind
