"""Deterministic em/en dash enforcement for outbound gateway text.

The house rule (no em dashes in chat replies) is enforced here mechanically,
because a rule that lives only in prompt text is context the model can drift
from. This filter runs outside the model on the text that actually reaches the
transport (``gateway.platforms.base.send_final_ledgered``, shared by the
Telegram adapter), so a drifting reply is corrected at the wire.

Policy under test:
  * a spaced dash (a parenthetical aside) becomes ", ";
  * a dash glued to its neighbours (a range or hyphenation) becomes "-";
  * both U+2014 and U+2013 are handled;
  * fenced code blocks and inline code spans are copied byte-for-byte;
  * the transform is idempotent and never doubles punctuation.
"""

import pytest

from gateway.delivery_voice import final_delivery_voice_check

EM_DASH = "\u2014"
EN_DASH = "\u2013"
DASHES = (EM_DASH, EN_DASH)


def test_exports_the_em_dash_guard_used_by_the_send_seam():
    """Pin the exact object the Telegram send path calls, not a copy of the policy."""
    from gateway.platforms import base as base_module
    from gateway.run import _final_delivery_voice_check

    assert base_module.final_delivery_voice_check is final_delivery_voice_check
    assert _final_delivery_voice_check is final_delivery_voice_check


@pytest.mark.parametrize("dash", DASHES)
def test_spaced_dash_becomes_comma_and_space(dash):
    assert final_delivery_voice_check(f"Mimi {dash} your fix is tiny.") == "Mimi, your fix is tiny."


@pytest.mark.parametrize("dash", DASHES)
def test_glued_dash_becomes_plain_hyphen(dash):
    assert final_delivery_voice_check(f"The wait is 3{dash}5 seconds.") == "The wait is 3-5 seconds."


@pytest.mark.parametrize("dash", DASHES)
def test_neither_dash_survives_in_prose(dash):
    checked = final_delivery_voice_check(f"One {dash} two {dash} three")

    assert EM_DASH not in checked
    assert EN_DASH not in checked


def test_parenthetical_aside_reads_as_a_comma_pair():
    raw = "The retry is cheap \u2014 it only re-reads the ledger \u2014 so I kept it."
    assert final_delivery_voice_check(raw) == (
        "The retry is cheap, it only re-reads the ledger, so I kept it."
    )


def test_comma_adjacent_dash_never_doubles_the_comma():
    assert final_delivery_voice_check("The list: apples, \u2014 oranges.") == (
        "The list: apples, oranges."
    )


def test_trailing_aside_does_not_leave_a_dangling_hyphen():
    assert final_delivery_voice_check("The patch is applied \u2014") == "The patch is applied"


def test_dash_bullets_stay_bullets():
    assert final_delivery_voice_check("Plan:\n\u2014 step one\n\u2014 step two") == (
        "Plan:\n- step one\n- step two"
    )


FENCED_CODE = "label = 'before \u2014 after'  # keep  two  spaces\nprint(label)\n"


def test_fenced_code_block_is_byte_identical():
    raw = f"Here is the snippet:\n\n```python\n{FENCED_CODE}```\n\nThat is all."
    checked = final_delivery_voice_check(raw)

    assert FENCED_CODE in checked
    assert checked == f"Here is the snippet:\n\n```python\n{FENCED_CODE}```\n\nThat is all."


def test_fenced_code_block_survives_an_unclosed_fence():
    """An unterminated fence is data too: protecting it is safer than guessing.

    The message is asserted byte-identical here because the guard strips only
    the message-level leading/trailing whitespace (pre-existing behaviour); the
    fenced tail itself must come through untouched.
    """
    raw = "Here:\n\n```\nlabel = 'before \u2014 after'  # keep  two  spaces\nprint(label)"
    assert final_delivery_voice_check(raw) == raw


def test_inline_code_span_is_byte_identical():
    span = "git log --pretty=%s \u2014 %an"
    checked = final_delivery_voice_check(f"Run `{span}` now \u2014 it lists authors.")

    assert f"`{span}`" in checked
    assert checked == f"Run `{span}` now, it lists authors."


def test_inline_code_span_prose_around_it_is_still_normalized():
    checked = final_delivery_voice_check("Range 3\u20135 lives in `a \u2014 b` here \u2014 ok.")

    assert checked == "Range 3-5 lives in `a \u2014 b` here, ok."


IDEMPOTENCE_SAMPLES = (
    "Mimi \u2014 your fix is tiny.",
    "The wait is 3\u20135 seconds.",
    "The retry is cheap \u2014 it only re-reads the ledger \u2014 so I kept it.",
    "The list: apples, \u2014 oranges.",
    "The patch is applied \u2014",
    "Plan:\n\u2014 step one\n\u2014 step two",
    "Here is the snippet:\n\n```python\n" + FENCED_CODE + "```\n\nThat is all.",
    "Run `git log --pretty=%s \u2014 %an` now \u2014 it lists authors.",
    "Got it \u2014 the tea is ready.",
    "Understood \u2014 the command is safe.",
)


@pytest.mark.parametrize("raw", IDEMPOTENCE_SAMPLES)
def test_transform_is_idempotent(raw):
    once = final_delivery_voice_check(raw)

    assert final_delivery_voice_check(once) == once


@pytest.mark.parametrize("raw", IDEMPOTENCE_SAMPLES)
def test_transform_never_doubles_punctuation(raw):
    checked = final_delivery_voice_check(raw)

    assert ",," not in checked
    assert " ," not in checked
    assert ",," not in final_delivery_voice_check(checked)


def test_em_dash_filter_keeps_code_untouched_when_prose_also_has_a_dash():
    raw = "Run this \u2014 it is safe:\n\n```sh\ndocker ps \u2014 format '{{.Names}}'\n```\n"

    assert final_delivery_voice_check(raw) == (
        "Run this, it is safe:\n\n```sh\ndocker ps \u2014 format '{{.Names}}'\n```"
    )


FILLER_CLOSERS = (
    "Got it, what's next?",
    "Got it. what's next?",
    "Let me know if you need anything else.",
    "Let me know how it goes!",
    "Let me know how things go.",
    "Feel free to reach out.",
    "Any questions?",
)


@pytest.mark.parametrize("raw", FILLER_CLOSERS)
def test_filler_closers_are_stripped_at_the_final_seam(raw: str):
    assert final_delivery_voice_check(raw) == ""


def test_filler_closer_strip_keeps_real_content_and_recapitalizes():
    raw = "it's landed, sayangkuu. any questions?"

    assert final_delivery_voice_check(raw) == "It's landed, sayangkuu."
