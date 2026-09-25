"""Runtime hook for requesting-code-review / GitHub PR review (LAB-62).

The review flow is skill-driven (no in-process reviewer). The hook is the
real call path: ``run_review_triage_hook`` plus ``python -m agent.jev_review_triage``.

Contracts:
- enabled -> calls ``triage_pr_review`` and may post one comment
- disabled -> does not call triage, does not post, does not touch gh
- one comment per PR at the hook (second call does not post)
- never auto-approves (event is always COMMENT; no APPROVE submitter call)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from agent.jev_review_triage import (
    FileDiffMeta,
    JevReviewTriageConfig,
    ReviewTriageInput,
    VERDICT_COMMENT_MARKER,
    parse_jev_review_triage_config,
)

REPO = Path(__file__).resolve().parents[2]


def _payload() -> ReviewTriageInput:
    return ReviewTriageInput(
        pr_number=62,
        title="Wire review triage hook",
        head_sha="abc123",
        base_ref="main",
        files=[FileDiffMeta(path="a.py", patch_sha256="abc", additions=1, deletions=0)],
        diff_excerpt="diff --git a/a.py b/a.py\n+x\n",
    )


def _enabled() -> JevReviewTriageConfig:
    return parse_jev_review_triage_config(
        {
            "enabled": True,
            "threshold": 0.85,
            "model": "jev-latest",
            "timeout_seconds": 30,
            "max_diff_chars": 12000,
        }
    )


def test_hook_symbol_is_the_review_flow_entry():
    """Red on base: the review flow has no runtime hook yet."""
    import agent.jev_review_triage as mod

    assert hasattr(mod, "run_review_triage_hook"), "review flow hook missing"
    assert hasattr(mod, "main"), "python -m entry missing"


def test_hook_does_not_fire_when_disabled(monkeypatch):
    import agent.jev_review_triage as mod

    def _boom(*_a, **_k):
        raise AssertionError("triage_pr_review must not run when disabled")

    monkeypatch.setattr(mod, "triage_pr_review", _boom)
    monkeypatch.setattr(mod, "_post_systemone", _boom)
    posted: List[str] = []
    submitted: List[str] = []

    result = mod.run_review_triage_hook(
        _payload(),
        existing_bodies=[],
        poster=posted.append,
        review_submitter=lambda event, _body: submitted.append(event),
        cfg=parse_jev_review_triage_config({"enabled": False}),
    )

    assert result["fired"] is False
    assert result["enabled"] is False
    assert result["posted"] is False
    assert result["reason"] == "disabled"
    assert result["auto_approved"] is False
    assert posted == []
    assert submitted == []


def test_hook_fires_when_enabled_and_never_auto_approves(monkeypatch):
    import agent.jev_review_triage as mod

    calls: List[Any] = []

    def _triage(payload, **kwargs):
        calls.append((payload, kwargs))
        return mod.JevReviewTriageResult(
            verdict="approve",
            confidence=0.95,
            reason="ok",
            should_post=True,
        )

    monkeypatch.setattr(mod, "triage_pr_review", _triage)
    posted: List[str] = []
    submitted: List[str] = []

    result = mod.run_review_triage_hook(
        _payload(),
        existing_bodies=[],
        poster=posted.append,
        review_submitter=lambda event, _body: submitted.append(event),
        cfg=_enabled(),
    )

    assert result["fired"] is True
    assert len(calls) == 1
    assert calls[0][0].pr_number == 62
    assert result["verdict"] == "approve"
    assert result["event"] == "COMMENT"
    assert result["auto_approved"] is False
    assert result["posted"] is True
    assert len(posted) == 1
    assert VERDICT_COMMENT_MARKER in posted[0]
    assert submitted == []
    assert "APPROVE" not in posted[0]


def test_hook_one_comment_dedupe(monkeypatch):
    import agent.jev_review_triage as mod

    def _triage(payload, **kwargs):
        return mod.JevReviewTriageResult(
            verdict="needs-human",
            confidence=0.4,
            reason="below_threshold",
            should_post=True,
        )

    monkeypatch.setattr(mod, "triage_pr_review", _triage)
    posted: List[str] = []
    cfg = _enabled()

    first = mod.run_review_triage_hook(
        _payload(),
        existing_bodies=[],
        poster=posted.append,
        cfg=cfg,
    )
    second = mod.run_review_triage_hook(
        _payload(),
        existing_bodies=list(posted),
        poster=posted.append,
        cfg=cfg,
    )

    assert first["posted"] is True
    assert second["fired"] is True
    assert second["posted"] is False
    assert second["post_reason"] == "already_posted"
    assert len(posted) == 1


def test_hook_real_triage_path_posts_comment_not_approve(monkeypatch):
    """Enabled hook must call the real triage_pr_review, not a parallel grader."""
    import agent.jev_review_triage as mod

    monkeypatch.setattr(mod, "resolve_typesafe_api_key", lambda: "test-key")

    class _Http:
        def post(self, url, headers=None, json=None):
            class _Resp:
                status_code = 200
                text = ""

                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "answers": {
                            "verdict": {"choice": "approve", "confidence": 0.99}
                        }
                    }

            return _Resp()

    posted: List[str] = []
    submitted: List[Any] = []
    result = mod.run_review_triage_hook(
        _payload(),
        existing_bodies=[],
        poster=posted.append,
        review_submitter=lambda *a, **k: submitted.append((a, k)),
        cfg=_enabled(),
        http_client=_Http(),
        api_key="test-key",
    )
    assert result["fired"] is True
    assert result["verdict"] == "approve"
    assert result["event"] == "COMMENT"
    assert result["auto_approved"] is False
    assert len(posted) == 1
    assert submitted == []


def test_main_disabled_does_not_call_gh(monkeypatch, capsys):
    import agent.jev_review_triage as mod

    monkeypatch.setattr(
        mod,
        "load_jev_review_triage_config",
        lambda user_config=None: JevReviewTriageConfig(enabled=False),
    )

    def _gh(_args):
        raise AssertionError("gh must not run when triage is disabled")

    code = mod.main(["--pr", "62"], gh_runner=_gh)
    assert code == 0
    assert "disabled" in capsys.readouterr().out


def test_main_enabled_posts_one_comment_never_approve(monkeypatch, capsys):
    import agent.jev_review_triage as mod

    monkeypatch.setattr(
        mod,
        "load_jev_review_triage_config",
        lambda user_config=None: _enabled(),
    )
    gh_calls: List[List[str]] = []

    def _gh(args):
        gh_calls.append(list(args))
        joined = " ".join(args)
        if "pr view" in joined:
            return (
                '{"number":62,"title":"Hook","headRefOid":"abc","baseRefName":"main",'
                '"files":[{"path":"a.py","additions":1,"deletions":0,"changeType":"MODIFIED"}],'
                '"comments":[],"reviews":[]}'
            )
        if "pr diff" in joined:
            return "diff --git a/a.py b/a.py\n+++ b/a.py\n+x\n"
        if "pr comment" in joined:
            return ""
        raise AssertionError(f"unexpected gh: {args}")

    monkeypatch.setattr(mod, "resolve_typesafe_api_key", lambda: "test-key")

    class _Http:
        def post(self, url, headers=None, json=None):
            class _Resp:
                status_code = 200
                text = ""

                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "answers": {
                            "verdict": {"choice": "request-changes", "confidence": 0.91}
                        }
                    }

            return _Resp()

    monkeypatch.setattr(mod, "_default_http_client", lambda: _Http())
    code = mod.main(["--pr", "62"], gh_runner=_gh)
    assert code == 0
    comment_calls = [c for c in gh_calls if "comment" in c]
    approve_calls = [c for c in gh_calls if "review" in c or "--approve" in c]
    assert len(comment_calls) == 1
    assert approve_calls == []
    assert VERDICT_COMMENT_MARKER in comment_calls[0][comment_calls[0].index("--body") + 1]
    out = capsys.readouterr().out
    assert "request-changes" in out
    assert "COMMENT" in out


def test_review_docs_call_entry_not_pasted_snippet():
    code_review = (
        REPO / "skills/software-development/github/references/code-review.md"
    ).read_text(encoding="utf-8")
    requesting = (
        REPO / "skills/software-development/requesting-code-review/SKILL.md"
    ).read_text(encoding="utf-8")
    entry = "python -m agent.jev_review_triage"
    assert entry in code_review
    assert entry in requesting
    assert "from agent.jev_review_triage import" not in code_review
    assert "run_review_triage_hook" in code_review
