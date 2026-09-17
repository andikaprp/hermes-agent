"""The voice-standard battery: approved goldens plus the failure modes they came from.

LAB-3 naturalness slice. The corpus is the set Andika approved on 2026-09-17: 10 good
replies that must pass untouched, and 11 real delivered replies that must be rejected with
the reason codes they actually trip.

The reason this battery exists instead of a length check: screening 141 real delivered
replies against the standard found 89 violations (em dash 78, bullets on a social turn 23,
headers 17, process narration 14, tables 8, canned openings 2), and the length-based
evaluator accepts most of them. Voice and length are separate concerns.
"""

import json
from pathlib import Path

import pytest

from gateway.naturalness_voice import (
    REASON_CODES,
    audit_corpus,
    audit_voice,
    codes_for,
)

CORPUS = Path(__file__).parent / "fixtures" / "lab3_voice_standard_corpus.json"


def _entries() -> list[dict]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))["entries"]


def test_the_approved_corpus_scores_exactly_as_approved():
    """Every approved golden passes with no findings; every rejected reply is rejected.

    This is the whole battery. A change that makes a golden fail is a voice regression, and
    a change that lets a rejected reply through is a hole.
    """
    report = audit_corpus(_entries())
    assert report["total"] == len(_entries())
    assert report["failed"] == [], report["failed"]
    assert report["passed"] == report["total"]


def test_every_declared_reason_code_has_a_reply_that_trips_it():
    """A reason code with no sample is a rule nobody is testing.

    Keeps the code table and the corpus in lockstep: adding a detector without a failing
    reply fails here rather than shipping untested.
    """
    exercised = {code for e in _entries() for code in e.get("codes", [])}
    assert set(REASON_CODES) - exercised == set()


def test_an_em_dash_is_caught_wherever_it_sits():
    """The single largest real violation (55% of delivered replies)."""
    assert codes_for("It worked — finally.") == ["em_dash"]
    assert codes_for("It worked.") == []


def test_structure_on_a_social_turn_is_the_failure_not_the_content():
    """The same information is fine as prose and wrong as a list on a social turn."""
    listed = "Tailscale is up:\n\n- active\n- hostname set"
    prose = "Tailscale is up and the hostname is set."
    assert "social_structure" in codes_for(listed, register="social")
    assert codes_for(prose, register="social") == []


def test_internals_are_licensed_only_when_evidence_was_asked_for():
    """Technical detail on an ordinary chat turn is a voice failure unless he asked for it."""
    reply = "The fix is in `gateway/run_turn.py`."
    assert "unneeded_internal_detail" in codes_for(reply)
    assert "unneeded_internal_detail" not in codes_for(reply, asks_for_evidence=True)


def test_offering_to_do_unasked_work_is_a_filler_question_but_asking_what_to_do_is_not():
    """`Want me to ...?` is filler; `What do you want me to verify first?` is a real question."""
    assert "filler_question" in codes_for("Want me to take a look?")
    assert "filler_question" not in codes_for("What do you want me to verify first?")


def test_a_verified_claim_needs_live_evidence():
    """`Verified` without a live observation is the claim the acceptance rules forbid."""
    assert "unsupported_verified_claim" in codes_for("The delivery is verified.")
    assert "unsupported_verified_claim" not in codes_for(
        "The delivery is verified: the receipt shows it arrived at 22:10."
    )


@pytest.mark.parametrize("code", REASON_CODES)
def test_each_reason_code_is_reportable_on_a_minimal_reply(code):
    """Every code must be reachable through the public entry point, not just declared."""
    samples = {
        "em_dash": "fine — done",
        "canned_phrase": "Acknowledged.",
        "social_structure": "ok:\n\n- one\n- two",
        "process_narration": "Testing the classifier now.",
        "buried_answer": "**the result:**\nit worked",
        "filler_question": "Want me to check?",
        "unneeded_internal_detail": "done in `gateway/run.py`",
        "unsupported_verified_claim": "Verified.",
    }
    assert code in codes_for(samples[code], register="social")


def test_audit_returns_findings_in_a_stable_order():
    """Callers group by code; ordering must not depend on regex evaluation order."""
    findings = audit_voice("Verified — Acknowledged.", register="social")
    positions = [f.code for f in findings]
    assert positions == sorted(positions, key=REASON_CODES.index)
