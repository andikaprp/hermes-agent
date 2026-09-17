"""Content-aware Telegram reaction *policy*: receipt vs tone glyphs, and when a reaction may
stand in for the reply.

Two jobs, named separately because they are chosen by different rules:

* **receipt** — 👀 while checking, then a done-glyph. This is the lifecycle ack the adapter
  already had, and it is what a non-social message gets.
* **tone** — the reaction *is* the emotional reply (❤ for affection, 🤣 for a joke, 😭 for
  sympathy, …), so the glyph on a purely social message is matched to what was sent.

The adapter reacts twice per turn (``on_processing_start`` / ``on_processing_complete``,
see ``plugins/platforms/telegram/adapter.py``) with fixed glyphs. :data:`STYLE_LIFECYCLE`
(default) keeps those glyphs byte-for-byte; :data:`STYLE_CONTENT` adds the tone-matched
success glyph plus the guarded reaction-only path below. An install that never sets
``telegram.reaction_style`` cannot observe a change.

Three properties are load-bearing, and the tests pin all three:

* **Default is the current behaviour.** Absent, blank, or unrecognized style normalizes to
  :data:`STYLE_LIFECYCLE`; the start glyph is 👀 in both styles, and the receipt job's
  done/failure glyphs are the pre-existing 👍/👎. Only a *social* inbound message under the
  content style gets a different success glyph.
* **Only curated glyphs leave the process.** Telegram rejects a reaction emoji it does not
  curate (``REACTION_INVALID``), and the adapter swallows that error (debug log, False), so a
  wrong glyph would fail *silently*. Every selectable glyph is validated against
  :data:`TELEGRAM_REACTION_EMOJI` and degrades to the lifecycle glyph on a miss. Two glyphs
  the naming suggests are NOT in Telegram's curated set and are substituted — ``✅`` for
  "done" (→ 👍) and ``😂`` (→ 🤣); ``❤️`` is sent in Telegram's curated form ``❤``.
* **Content-aware never reads the assistant's reply.** Classification runs on the inbound
  message text — the same string both lifecycle hooks already receive — so no model output
  can influence, or leak through, a reaction or its receipt. Matching is pure regex plus dict
  lookups: no I/O, no model call, no added turn latency.

The reaction-only guard is deliberately lopsided. A reaction may be the *whole* reply only
for a message that is purely social and unambiguous; anything else — a question, an implicit
ask, a work/status report, a pending decision, money/credential/destructive/deploy context,
an uncertainty flag, or simply content the classifiers do not recognize — forces text. Bad
silence is much worse than a redundant glyph, so every unknown resolves toward text, and the
caller-supplied veto flags can only ever *add* a reason to send text, never grant permission.

Fail-open on cosmetics, fail-closed on silence: every function here is total over any input
(``None``, non-strings, hostile ``__str__``) because the adapter's reaction seam must never
break a delivery path — while :func:`decide_reaction_only` denies on any internal fault.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

# ── styles ─────────────────────────────────────────────────────────────────────────────

STYLE_LIFECYCLE = "lifecycle"
STYLE_CONTENT = "content"
DEFAULT_STYLE = STYLE_LIFECYCLE
STYLES: Tuple[str, ...] = (STYLE_LIFECYCLE, STYLE_CONTENT)

# Tolerant of the spellings a human types for the opt-in; anything else is unknown.
_STYLE_ALIASES: Dict[str, str] = {
    "content-aware": STYLE_CONTENT,
    "content_aware": STYLE_CONTENT,
    "contentaware": STYLE_CONTENT,
    "aware": STYLE_CONTENT,
    "smart": STYLE_CONTENT,
    "semantic": STYLE_CONTENT,
    "tone": STYLE_CONTENT,
    "instinct": STYLE_CONTENT,
}

# ── jobs ───────────────────────────────────────────────────────────────────────────────

JOB_RECEIPT = "receipt"
JOB_TONE = "tone"

# ── reply modes (the reaction-only guard's verdict) ────────────────────────────────────

REPLY_TEXT = "text"
REPLY_REACTION_ONLY = "reaction_only"

# ── the lifecycle glyphs (the pre-existing behaviour, unchanged) ────────────────────────

START_EMOJI = "\U0001f440"          # 👀  "checking" — a receipt in both styles
RECEIPT_DONE_EMOJI = "\U0001f44d"   # 👍  done (intent ✅; Telegram curates no ✅)
FAILURE_EMOJI = "\U0001f44e"        # 👎  failed — outcome-driven, never tone-driven

# ── glyph curation ─────────────────────────────────────────────────────────────────────

# The emoji Telegram's Bot API accepts as a message reaction. A glyph outside this set is
# rejected with REACTION_INVALID, so every selectable glyph is validated against it and
# degrades to the lifecycle glyph on a miss. Kept as one literal string so the set and the
# glyphs read side by side.
TELEGRAM_REACTION_EMOJI: FrozenSet[str] = frozenset(
    "👍👎❤🔥🥰👏😁🤔🤯😱🤬😢🎉🤩🤮💩🙏👌🕊🤡🥱🥴😍🐳❤️‍🔥❤️‍🩹🌚🌭💯🤣⚡🍌🏆💔🤨😐🍓🍾💋🖕😈😴😭🤓👻👨‍💻👀🎃🙈"
    "😇😨🤝✍🤗🫡🎅🎄☃💅🤪🗿🆒💘🙉🦄😘💊🙊😎👾🤷🤷‍♂🤷‍♀😡"
)

# ── tones ──────────────────────────────────────────────────────────────────────────────

TONE_QUESTION = "question"
TONE_TROUBLE = "trouble"
TONE_URGENT = "urgent"
TONE_GRATITUDE = "gratitude"
TONE_PRAISE = "praise"
TONE_CELEBRATION = "celebration"
TONE_AFFECTION = "affection"
TONE_AMUSEMENT = "amusement"
TONE_SYMPATHY = "sympathy"
TONE_GREETING = "greeting"
TONE_ACKNOWLEDGEMENT = "acknowledgement"
TONE_ROUTINE = "routine"

# tone → glyph. Only social tones appear: these are the messages where the reaction is the
# natural reply, i.e. the TONE job. ``TONE_ROUTINE`` (unrecognized content) is deliberately
# absent — unknown content is a receipt, never a guess at affection.
TONE_GLYPHS: Dict[str, str] = {
    TONE_AFFECTION: "\u2764",        # ❤   (intent ❤️ — Telegram's curated form has no VS16)
    TONE_AMUSEMENT: "\U0001f923",    # 🤣   (intent 😂 — Telegram curates 🤣, not 😂)
    TONE_SYMPATHY: "\U0001f62d",     # 😭
    TONE_GRATITUDE: "\U0001f64f",    # 🙏
    TONE_PRAISE: "\U0001f44f",       # 👏
    TONE_CELEBRATION: "\U0001f389",  # 🎉
    TONE_GREETING: "\U0001f917",     # 🤗
    TONE_ACKNOWLEDGEMENT: "\U0001f44c",  # 👌
}

SOCIAL_TONES: FrozenSet[str] = frozenset(TONE_GLYPHS)
RECEIPT_TONES: FrozenSet[str] = frozenset(
    {TONE_QUESTION, TONE_TROUBLE, TONE_URGENT, TONE_ROUTINE})

# Explicit "?"/full-width "？" or an interrogative opener. Named (not inline in the table) so
# the guard's question check cannot silently drift onto another tone if the order changes.
# Indonesian openers are matched alongside English: this DM mixes both, and an ask the guard
# cannot see is the one failure mode that actually matters.
_QUESTION_ASK = re.compile(
    r"[?？]|^\s*(?:how|what|why|when|where|which|who|whom|whose|can|could|would|should|"
    r"shall|do|does|did|is|are|was|were|will|has|have|had|may|might|"
    r"apa(?:kah)?|gimana|gmn|bagaimana|kenapa|knp|napa|kapan|dimana|di mana|kemana|"
    r"berapa|siapa|mana)\b",
    re.IGNORECASE)

# Matching order IS the precedence, most actionable and least ambiguous first. A bug report
# phrased as a question ("why does this crash?") is still a bug report; urgency is a modifier
# on intent, so it outranks the phrasing carrying it. Tone order follows the same rule: a
# named feeling beats a generic greeting or ack.
_TONE_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (TONE_TROUBLE, re.compile(
        r"\b(?:error|errors|bug|bugs|broken|broke|crash(?:es|ed|ing)?|fail(?:s|ed|ing|ures?)?|"
        r"exception|traceback|stack ?trace|regression|outage|downtime|degraded|"
        r"rusa+k+|gaga+l+|lemo+t+|nge ?hang|gak (?:jalan|bisa)|ga (?:jalan|bisa)|"
        r"nggak (?:jalan|bisa)|tidak (?:jalan|bisa)|"
        r"not working|does ?n'?t work|does not work|wo ?n'?t work|is ?n'?t working|"
        # A bare number is ambiguous ("it costs 500"); a status code reads as trouble only
        # with an HTTP-ish verb in front of it, or a status phrase right after it.
        r"(?:gets?|getting|got|returns?|returning|throws?|gives?|sees?|http|status|code)\s*"
        r"(?:a\s+)?[45]\d{2}\b|"
        r"\b[45]\d{2}\s*[-–]?\s*(?:bad|errors?|internal|gateway|timeout|unavailable|forbidden|"
        r"unauthorized|not\s+found)\b|🐛)",
        re.IGNORECASE)),
    (TONE_URGENT, re.compile(
        r"\b(?:urgent|urgently|asap|emergency|critical|immediately|blocker|blocked|"
        r"right now|production|prod|hotfix|rollback|darurat|secepatnya|sekarang juga|buruan)\b",
        re.IGNORECASE)),
    (TONE_QUESTION, _QUESTION_ASK),
    (TONE_GRATITUDE, re.compile(
        r"\b(?:thanks+|thank you|thankyou|thx|tysm|ty|appreciate[ds]?|grateful|cheers|"
        r"maka+si+h?|terima ?kasih|trims|tengkyu|alhamdulillah)\b|🙏",
        re.IGNORECASE)),
    (TONE_PRAISE, re.compile(
        r"\b(?:nice work|good (?:job|work|girl|boy|bot)|great (?:job|work)|well done|nailed it|"
        r"impressive|amazing|awesome|"
        r"brilliant|perfect|legend|beautiful work|kere+n+|manta+p+|mantul+|heba+t+|bagu+s+)\b"
        r"|👏|🏆",
        re.IGNORECASE)),
    (TONE_CELEBRATION, re.compile(
        r"\b(?:congrats|congratulations|congratulate|celebrate|celebration|birthday|milestone|"
        # "selamat pagi" is a greeting, not a celebration — the lookahead keeps them apart.
        r"yay+|yea+y+|yey+|yess+|hooray|woo ?hoo|we did it|it works|hore+|asyik|akhirnya|"
        r"selamat(?!\s+(?:pagi|siang|sore|malam)))\b"
        r"|🎉|🎂|🥳",
        re.IGNORECASE)),
    (TONE_AFFECTION, re.compile(
        # "sayang"/"sayaang" is the DM's common endearment ("sayaang"); it is also the word for
        # "what a pity", where the glyph is merely a different social one — the guard's verdict
        # is identical either way, so the ambiguity costs nothing that matters.
        r"\b(?:i love (?:you|it|this)|love you|miss you|adore (?:you|this)|proud of you|"
        r"xoxo|hugs?|saya+ng(?:ku)?|cinta+(?:ku)?|kange+n+|pelu+k+)\b|❤|🥰|😍|💋",
        re.IGNORECASE)),
    (TONE_AMUSEMENT, re.compile(
        r"\b(?:lol|lmao|lmfao|rofl|haha+|hehe+|hilarious|so funny|that's funny|cracked me up|"
        r"wkwk(?:wk)*|wkwk+|ngakak+|lucu+|ketawa+)\b"
        r"|😂|🤣",
        re.IGNORECASE)),
    (TONE_SYMPATHY, re.compile(
        r"\b(?:so sorry|sorry to hear|that's rough|that sucks|condolences|rest in peace|rip|"
        r"feel better|thinking of you|aww+|turut berduka|sabar ya|kasiha+n+)\b|😭|💔",
        re.IGNORECASE)),
    (TONE_GREETING, re.compile(
        r"\b(?:hi|hello|hey|hey there|howdy|greetings|good morning|good afternoon|"
        r"good evening|good night|goodnight|halo+|hai+|hei+|pagi+)\b|"
        r"selamat (?:pagi|siang|sore|malam)",
        re.IGNORECASE)),
    (TONE_ACKNOWLEDGEMENT, re.compile(
        r"\b(?:ok|okay|okai|oke+|oki+e+|okey+|k|sure|got it|noted|understood|sounds good|"
        r"works for me|cool|yep|yup|"
        r"alright|fine by me|oke+|sia+p+|sip+|bai+k+)\b|👌|👍",
        re.IGNORECASE)),
)

# Classification reads intent from the head of a message ("the deploy is broken") and the
# tail of a long one adds nothing but cost; bound the scan so a pasted log cannot make this
# hot.
SCAN_LIMIT = 2000

# An implicit ask: a request or a question that carries no "?" — the guard treats it exactly
# like an explicit question. Indonesian forms are included because the classifier now recognizes
# Indonesian *social* tone: without matching asks, "sayaang, tolong cek ya" would be allowed to
# stand in for a reply it must not replace. A trailing "ya" is the confirmation tag ("sudah ya?")
# and counts as an ask — the conservative direction.
_IMPLICIT_ASK = re.compile(
    r"\b(?:please|pls|plz|can you|could you|would you|will you|let me know|tell me|"
    r"explain|remind me|send me|show me|share|what about|how about|any (?:idea|chance|"
    r"thoughts?)|i need|i want|i asked|waiting on|help|"
    r"tolong|mohon|coba|bisa(?:kah)?|minta|butuh|dong)\b|(?:^|\s)ya\s*[.!?…]*\s*$",
    re.IGNORECASE)

# Work/verification/risk vocabulary. A social message that carries any of this is MIXED, and
# mixed requires text: the reader may have to act on it. Words that are ordinary *subjects* of
# smalltalk ("thanks for the update") are deliberately absent — being mentioned is not the same
# as a status report — while the plain report vocabulary ("shipped", "deployed", "the tests
# pass") stays in, because a reaction must never stand in for a status report.
_WORK_SIGNAL = re.compile(
    r"\b(?:deploy(?:ed|ing|ment)?|shipped?|ship it|publish(?:ed)?|push(?:ed)?|commit(?:ted|s)?|"
    r"merge[ds]?|branch(?:es)?|pull request|\bpr\b|issue|ticket|build|built|ci\b|"
    r"test(?:s|ed|ing)?|verify|verified|confirm(?:ed)?|check(?:ed|ing)?|review(?:ed)?|"
    r"audit(?:ed)?|validation?|status|report|summary|summari[sz]e|"
    r"cek|periksa|pastikan|verifikasi|konfirmasi|laporan|lapor|tugas|kerjaan|rapat|deadline|"
    r"kirim|ubah|ganti|perbaiki|benerin|pasang|"
    r"refactor(?:ed)?|debug(?:ged)?|investigate[ds]?|configur(?:e|ed)|"
    r"backup(?:s)?|restart(?:ed)?|reboot(?:ed)?|rollback|migrat(?:e|ed|ion)|"
    r"invoice|payment|paid|refund|transfer|charge[ds]?|"
    r"uang|harga|biaya|tagihan|bayar|rekening|sandi|"
    r"price|cost|budget|\bmoney\b|password|passphrase|\btoken\b|api key|credential[s]?|"
    r"secret[s]?|\bssh\b|log ?in|delete[ds]?|deletion|remove[ds]?|\bdrop(?:ped)?\b|wipe[ds]?|"
    r"hapus|"
    r"rm -rf|revoke[ds]?|permission[s]?|bank|account|2fa|otp)\b",
    re.IGNORECASE)

# Links, code, and file paths: a message shaped like data is never purely social.
_MIXED_SHAPE = re.compile(
    r"https?://|\bwww\.|\b[\w./-]+\.(?:py|ts|tsx|js|jsx|md|json|ya?ml|log|sql|csv|sh|env)\b|"
    r"```|`[^`]+`|\d+\s*(?:%|ms|secs?|seconds?|mins?|minutes?|hours?|days?|usd|eur|idr|rp|"
    r"gb|mb|tb|files?|rows?|errors?|users?)",
    re.IGNORECASE)

# A purely social message is short. Past this, the chance it also carries substance is high
# enough that text is the safe answer.
SOCIAL_MAX_CHARS = 240

# Machine-readable reasons. Tests and logs key off these, never off prose.
REASON_SOCIAL = "social"
REASON_STYLE_DISABLED = "style_disabled"
REASON_NO_CONTENT = "no_content"
REASON_QUESTION_ASKED = "question_asked"
REASON_IMPLICIT_ASK = "implicit_ask"
REASON_MIXED_OR_WORK = "mixed_or_work"
REASON_NOT_SOCIAL = "not_social"
REASON_TOO_LONG = "too_long"
REASON_UNCERTAIN = "uncertain"
REASON_PENDING_DECISION = "pending_decision"
REASON_WORK_REPORT = "work_report"
REASON_NEXT_STEP = "next_step"
REASON_SENSITIVE = "sensitive"
REASON_GUARD_ERROR = "guard_error"


@dataclass(frozen=True)
class ReactionOnlyDecision:
    """Verdict on whether a reaction may be the whole reply to one message."""

    mode: str
    reason: str
    tone: str
    job: str
    emoji: Optional[str] = None

    @property
    def allowed(self) -> bool:
        """True only for :data:`REPLY_REACTION_ONLY`."""
        return self.mode == REPLY_REACTION_ONLY

    @property
    def text_required(self) -> bool:
        """True when the caller must send text instead of leaving the reaction as the reply."""
        return self.mode == REPLY_TEXT


def _coerce(content: Any) -> str:
    """Message text as a bounded string. Total: ``None`` → ``""``, hostile ``__str__`` → ``""``."""
    if content is None:
        return ""
    if isinstance(content, str):
        text = content
    else:
        try:
            text = str(content)
        except Exception:
            return ""
    return text[:SCAN_LIMIT]


def normalize_style(value: Any) -> str:
    """Configured value → a canonical style name; unknown, blank, or unreadable → the default.

    Never raises and never invents a style: an unknown token means the operator's intent is
    unreadable, and the only safe reading of that is today's behaviour.
    """
    if value is None:
        return DEFAULT_STYLE
    try:
        token = str(value).strip().lower()
    except Exception:
        return DEFAULT_STYLE
    if token in STYLES:
        return token
    return _STYLE_ALIASES.get(token, DEFAULT_STYLE)


def classify_content(content: Any) -> str:
    """Tone bucket for one inbound message, or :data:`TONE_ROUTINE` when nothing matches.

    Pure and total: no I/O, no model call, no state.
    """
    text = _coerce(content)
    if not text.strip():
        return TONE_ROUTINE
    for tone, pattern in _TONE_PATTERNS:
        try:
            if pattern.search(text):
                return tone
        except Exception:  # pragma: no cover — a broken pattern must not break a turn
            continue
    return TONE_ROUTINE


def is_social_tone(tone: Any) -> bool:
    """True when ``tone`` is one where a reaction can carry the emotional reply."""
    return tone in SOCIAL_TONES


def reaction_job(content: Any) -> str:
    """Which job the reaction does for this message: :data:`JOB_RECEIPT` or :data:`JOB_TONE`."""
    return JOB_TONE if is_social_tone(classify_content(content)) else JOB_RECEIPT


def tone_emoji(tone: Any) -> Optional[str]:
    """The tone's curated glyph, or ``None`` for a non-social (receipt) tone."""
    glyph = TONE_GLYPHS.get(tone) if isinstance(tone, str) else None
    return glyph if glyph in TELEGRAM_REACTION_EMOJI else None


def start_reaction(style: Any = None, content: Any = None) -> str:
    """Glyph for the in-progress ack: 👀 under every style.

    The ack says "I'm checking" — that is a receipt, not a tone, so it is not content-aware in
    either style. ``style``/``content`` are accepted so the three selectors stay
    call-compatible; a future tone-on-start change has to break this test on purpose.
    """
    return START_EMOJI


def success_reaction(style: Any, content: Any) -> str:
    """Glyph for a delivered turn: the tone glyph for a social message under content style.

    Everything else — every non-social message, any unreadable style, any uncurated glyph —
    returns the receipt done-glyph, i.e. exactly what the adapter sent before content styles
    existed.
    """
    try:
        if normalize_style(style) != STYLE_CONTENT:
            return RECEIPT_DONE_EMOJI
        emoji = tone_emoji(classify_content(content))
    except Exception:
        return RECEIPT_DONE_EMOJI
    return emoji if emoji else RECEIPT_DONE_EMOJI


def failure_reaction(style: Any = None, content: Any = None) -> str:
    """Always :data:`FAILURE_EMOJI` — failure is reported by outcome, never by tone.

    Deliberate asymmetry: a failed turn's glyph already carries the one bit a reader needs,
    and decorating it with a tone would bury the failure signal in the styling.
    """
    return FAILURE_EMOJI


def _decision(mode: str, reason: str, content: Any) -> ReactionOnlyDecision:
    tone = classify_content(content)
    return ReactionOnlyDecision(
        mode=mode, reason=reason, tone=tone,
        job=JOB_TONE if is_social_tone(tone) else JOB_RECEIPT,
        emoji=tone_emoji(tone) if mode == REPLY_REACTION_ONLY else None)


def decide_reaction_only(
    content: Any,
    *,
    style: Any = None,
    confident: bool = True,
    pending_decision: bool = False,
    reports_work: bool = False,
    needs_next_step: bool = False,
    sensitive: bool = False,
) -> ReactionOnlyDecision:
    """May a reaction be the whole reply to this message? Deny whenever the answer is unclear.

    The keyword vetoes (``confident``, ``pending_decision``, ``reports_work``,
    ``needs_next_step``, ``sensitive``) can only force text — a caller can never buy
    permission by leaving one out, because the text-derived checks below run regardless.
    Ordering is intentional: the style gate first (a default install is told
    ``style_disabled`` rather than a content verdict), then the vetoes, then the text checks
    from the most specific reason to the least.
    """
    try:
        if normalize_style(style) != STYLE_CONTENT:
            return _decision(REPLY_TEXT, REASON_STYLE_DISABLED, content)
        if not confident:
            return _decision(REPLY_TEXT, REASON_UNCERTAIN, content)
        if pending_decision:
            return _decision(REPLY_TEXT, REASON_PENDING_DECISION, content)
        if reports_work:
            return _decision(REPLY_TEXT, REASON_WORK_REPORT, content)
        if needs_next_step:
            return _decision(REPLY_TEXT, REASON_NEXT_STEP, content)
        if sensitive:
            return _decision(REPLY_TEXT, REASON_SENSITIVE, content)
        text = _coerce(content)
        if not text.strip():
            return _decision(REPLY_TEXT, REASON_NO_CONTENT, content)
        if _QUESTION_ASK.search(text):
            # Explicit "?" or an interrogative opener — he asked something.
            return _decision(REPLY_TEXT, REASON_QUESTION_ASKED, content)
        if _IMPLICIT_ASK.search(text):
            return _decision(REPLY_TEXT, REASON_IMPLICIT_ASK, content)
        if _WORK_SIGNAL.search(text) or _MIXED_SHAPE.search(text):
            return _decision(REPLY_TEXT, REASON_MIXED_OR_WORK, content)
        if len(text) > SOCIAL_MAX_CHARS:
            return _decision(REPLY_TEXT, REASON_TOO_LONG, content)
        if not is_social_tone(classify_content(text)):
            return _decision(REPLY_TEXT, REASON_NOT_SOCIAL, content)
        return _decision(REPLY_REACTION_ONLY, REASON_SOCIAL, content)
    except Exception:
        return _decision(REPLY_TEXT, REASON_GUARD_ERROR, content)
