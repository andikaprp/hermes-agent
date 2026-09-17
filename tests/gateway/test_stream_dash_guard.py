"""Em/en dash guard on the STREAMING delivery seams (the guard the final send runs is
suppressed on a streamed turn).

Why this file exists
--------------------
``final_delivery_voice_check`` (gateway/delivery_voice.py) normalizes dashes only on the
assembled final text, at ``gateway.platforms.base.send_final_ledgered``. When the model
STREAMS its answer to Telegram, the gateway logs "Suppressing normal final send ...
(final delivery already confirmed)" and the guarded send never runs: the frames the user
already read are the delivered message. A drifting em dash in a streamed frame therefore
shipped unguarded.

These tests pin the dash-only guard the stream seams now run:
  * ``GatewayStreamConsumer._send_or_edit`` — the funnel every streamed frame crosses
    (native frame, draft frame, edit-in-place, first send);
  * ``TelegramAdapter.send_draft`` — the draft-frame sender itself.

They assert the OUTGOING payload (what the adapter hands the Bot API), not the helper's
return value, and they drive the real adapter/consumer code paths. Async tests use
``asyncio.run`` (the pattern in tests/gateway/test_wecom.py) so they run without the
optional pytest-asyncio plugin.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.delivery_voice import normalize_stream_dashes
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

EM_DASH = "\u2014"
EN_DASH = "\u2013"
DASHES = (EM_DASH, EN_DASH)

REPLY_WITH_DASH = f"Mimi {EM_DASH} your fix is tiny.\n\nThe wait is 3{EN_DASH}5 seconds."


# --------------------------------------------------------------------------------------
# The transform itself: dash-only, code/URL aware, safe frame-to-frame.
# --------------------------------------------------------------------------------------


class TestFrameDashTransform:
    @pytest.mark.parametrize("dash", DASHES)
    def test_spaced_dash_becomes_comma_and_space(self, dash):
        assert normalize_stream_dashes(f"Mimi {dash} your fix is tiny.") == \
            "Mimi, your fix is tiny."

    @pytest.mark.parametrize("dash", DASHES)
    def test_glued_dash_becomes_plain_hyphen(self, dash):
        assert normalize_stream_dashes(f"The wait is 3{dash}5 seconds.") == \
            "The wait is 3-5 seconds."

    def test_dash_bullet_stays_a_hyphen(self):
        assert normalize_stream_dashes(f"Plan:\n{EM_DASH} one\n{EM_DASH} two") == \
            "Plan:\n- one\n- two"

    def test_no_frame_keeps_a_dash_in_prose(self):
        checked = normalize_stream_dashes(REPLY_WITH_DASH)

        assert EM_DASH not in checked
        assert EN_DASH not in checked

    @pytest.mark.parametrize("raw", (
        REPLY_WITH_DASH,
        f"apples, {EM_DASH} oranges",
        f"Range 3{EN_DASH}5 and an aside {EM_DASH} here",
        f"Plan:\n{EM_DASH} one\n{EM_DASH} two",
        f"Run `git log {EM_DASH} %an` now {EM_DASH} ok",
        f"```python\nx = 'a {EM_DASH} b'\n```",
        "no dashes at all",
    ))
    def test_idempotent_so_re_running_a_frame_is_a_no_op(self, raw):
        once = normalize_stream_dashes(raw)

        assert normalize_stream_dashes(once) == once

    @pytest.mark.parametrize("raw", (
        f"apples, {EM_DASH} oranges",
        f"the list {EM_DASH} really {EM_DASH} is fine",
        f"trailing aside {EM_DASH}",
        f"double  spaced {EM_DASH}  here",
    ))
    def test_never_doubles_punctuation_and_never_leaves_a_dangling_space(self, raw):
        checked = normalize_stream_dashes(raw)

        assert ",," not in checked
        assert " ," not in checked
        assert ",  " not in checked

    def test_matches_the_final_guard_on_prose(self):
        """Whatever the guarded final would print, the frame prints too."""
        from gateway.delivery_voice import final_delivery_voice_check

        for raw in (f"Mimi {EM_DASH} your fix is tiny.", f"apples, {EM_DASH} oranges",
                    f"the retry is cheap {EM_DASH} it re-reads the ledger {EM_DASH} so keep it"):
            assert normalize_stream_dashes(raw) == final_delivery_voice_check(raw)

    def test_inline_code_span_is_copied_byte_for_byte(self):
        span = f"git log --pretty=%s {EM_DASH} %an"
        checked = normalize_stream_dashes(f"Run `{span}` now {EM_DASH} it lists authors.")

        assert f"`{span}`" in checked
        assert checked == f"Run `{span}` now, it lists authors."

    def test_fenced_code_block_is_copied_byte_for_byte(self):
        body = f"label = 'before {EM_DASH} after'  # keep  two  spaces\nprint(label)\n"
        raw = f"Here is the snippet:\n\n```python\n{body}```\n\nThat is all."

        assert normalize_stream_dashes(raw) == raw

    def test_unclosed_fence_protects_the_growing_code_block(self):
        """A frame mid-code-block must not rewrite what the model is writing verbatim."""
        raw = f"Here:\n\n```\nlabel = 'before {EM_DASH} after'"

        assert normalize_stream_dashes(raw) == raw

    def test_dash_inside_a_url_is_never_rewritten(self):
        raw = f"see https://example.com/a{EM_DASH}b now {EM_DASH} then reply"

        assert f"https://example.com/a{EM_DASH}b" in normalize_stream_dashes(raw)
        assert normalize_stream_dashes(raw) == \
            f"see https://example.com/a{EM_DASH}b now, then reply"

    @pytest.mark.parametrize("raw", (
        f"Mimi {EM_DASH} your fix is tiny.\n\nThen restart the worker.",
        f"Plan:\n{EM_DASH} one\n{EM_DASH} two\nDone.",
        f"Here:\n\n```python\nx = 1 {EM_DASH} 2\ny = 3\n```\n\nThat is all.",
        f"The wait is 3{EN_DASH}5s {EM_DASH} ok.",
    ))
    def test_growing_frames_never_rewrite_text_already_shown(self, raw):
        """A frame is the whole accumulated text so far: its normalized form must remain a
        prefix of the final frame (trailing whitespace is the one exception, because a dash
        arriving at the very end of a frame re-punctuates the whitespace it lands on).
        """
        full = normalize_stream_dashes(raw)
        for cut in range(len(raw) + 1):
            frame = normalize_stream_dashes(raw[:cut])
            assert full.startswith(frame.rstrip()), (
                f"frame {raw[:cut]!r} normalized to {frame!r}, which is not a prefix of "
                f"{full!r}: the preview would visibly rewrite already-delivered text"
            )

    def test_frame_ending_on_a_dash_guesses_and_the_next_frame_corrects_it(self):
        """Known, bounded behaviour, pinned so it cannot degrade silently.

        ``"the fix is \u2014"`` (an aside) and ``"the fix is \u2014tiny"`` (a hyphenation) are
        indistinguishable until the next character arrives, so the frame that *ends* on the
        dash prints the aside reading and the frame after it re-punctuates that same
        position. Holding the frame back instead is not an option: a delayed frame is a
        visible stall. No word before the dash is ever rewritten.
        """
        assert normalize_stream_dashes("the fix is") == "the fix is"
        frame_ending_on_dash = normalize_stream_dashes(f"the fix is {EM_DASH}")
        next_frame = normalize_stream_dashes(f"the fix is {EM_DASH}tiny")

        assert frame_ending_on_dash == "the fix is, "
        assert next_frame == "the fix is -tiny"
        assert next_frame.startswith("the fix is")  # delivered words survive both readings

    def test_unclosed_inline_span_is_prose_until_the_closing_backtick(self):
        """Known, bounded cos: the guard must never stop guarding.

        Protecting the tail from an unmatched backtick would be prefix-stable, but a stray
        literal backtick in prose would then disable dash normalization for the rest of the
        reply \u2014 the leak this guard exists to stop. So an unclosed span reads as prose
        (same as ``final_delivery_voice_check``) and the closing backtick's frame restores
        the span byte-for-byte.
        """
        assert normalize_stream_dashes(f"restart `systemctl {EM_DASH} now") == \
            "restart `systemctl, now"
        assert normalize_stream_dashes(f"restart `systemctl {EM_DASH} now` please") == \
            f"restart `systemctl {EM_DASH} now` please"

    def test_scaffolding_is_NOT_stripped_from_frames(self):
        """Dash-only on purpose: removing content mid-stream would make the preview jump."""
        raw = f"Got it {EM_DASH} here's what i found: the fix is tiny."

        checked = normalize_stream_dashes(raw)

        assert "Got it" in checked          # final_delivery_voice_check would delete this
        assert "here's what i found" in checked
        assert EM_DASH not in checked

    def test_blank_and_non_string_input_is_passed_through(self):
        assert normalize_stream_dashes("") == ""
        assert normalize_stream_dashes(None) is None

    def test_a_raising_frame_never_escapes_the_helper(self):
        class ExplodingStr(str):
            def splitlines(self, *args, **kwargs):
                raise RuntimeError("boom")

        hostile = ExplodingStr(REPLY_WITH_DASH)

        assert normalize_stream_dashes(hostile) is hostile


# --------------------------------------------------------------------------------------
# Seams: the OUTGOING payload of each streaming transport.
# --------------------------------------------------------------------------------------


def _make_consumer(adapter, *, transport="edit", chat_type="dm", cursor=""):
    cfg = StreamConsumerConfig(transport=transport, chat_type=chat_type, cursor=cursor,
                               edit_interval=0.01, buffer_threshold=5)
    return GatewayStreamConsumer(adapter, "12345", cfg)


def _make_edit_adapter():
    """Base-platform adapter whose send/edit record their content (no drafts, no native)."""
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    EditAdapter = type("EditAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 4096})
    EditAdapter.__abstractmethods__ = frozenset()
    adapter = EditAdapter.__new__(EditAdapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.supports_draft_streaming = lambda chat_type=None, metadata=None, chat_id=None: False
    adapter.sent_payloads = []

    async def _send(*, chat_id, content, reply_to=None, metadata=None):
        adapter.sent_payloads.append(content)
        return SendResult(success=True, message_id="m1")

    async def _edit(*, chat_id, message_id, content, finalize=False, metadata=None):
        adapter.sent_payloads.append(content)
        return SendResult(success=True)

    adapter.send = _send
    adapter.edit_message = _edit
    return adapter


def _make_native_adapter():
    """WeCom-shaped adapter: every frame goes through send_stream_frame()."""
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    NativeAdapter = type(
        "NativeAdapter", (BasePlatformAdapter,),
        {"MAX_MESSAGE_LENGTH": 4096, "SUPPORTS_NATIVE_STREAMING": True},
    )
    NativeAdapter.__abstractmethods__ = frozenset()
    adapter = NativeAdapter.__new__(NativeAdapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.supports_draft_streaming = lambda chat_type=None, metadata=None, chat_id=None: False
    adapter.supports_native_streaming = lambda chat_type=None, metadata=None: True
    adapter.native_frames = []

    async def _frame(text, *, finalize=False, chat_id=None, reply_to=None, turn_id=None, **kw):
        adapter.native_frames.append(text)
        return SendResult(success=True)

    adapter.send_stream_frame = _frame
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m1"))
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True))
    return adapter


def _make_real_telegram_adapter():
    """The real TelegramAdapter with a fake Bot API client recording draft frames."""
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    return adapter


def _draft_payload_texts(adapter):
    return [call.kwargs["text"] for call in adapter._bot.send_message_draft.await_args_list]


class TestConsumerStreamSeamsCarryNoEmDash:
    def test_edit_in_place_frames_carry_no_dash(self):
        adapter = _make_edit_adapter()
        consumer = _make_consumer(adapter, transport="edit")

        delivered = asyncio.run(consumer._send_or_edit(REPLY_WITH_DASH, finalize=False))

        assert delivered is True
        assert adapter.sent_payloads == [normalize_stream_dashes(REPLY_WITH_DASH)]
        for payload in adapter.sent_payloads:
            assert EM_DASH not in payload and EN_DASH not in payload

    def test_native_stream_frames_carry_no_dash(self):
        adapter = _make_native_adapter()
        consumer = _make_consumer(adapter, transport="edit")
        consumer._use_native_streaming = True
        consumer._native_stream_opened = True

        delivered = asyncio.run(consumer._send_or_edit(REPLY_WITH_DASH, finalize=False))

        assert delivered is True
        assert adapter.native_frames, "expected a native frame on the wire"
        for frame in adapter.native_frames:
            assert EM_DASH not in frame and EN_DASH not in frame
        assert adapter.native_frames[-1] == normalize_stream_dashes(REPLY_WITH_DASH)

    def test_real_telegram_draft_frames_carry_no_dash_end_to_end(self):
        """Real consumer + real TelegramAdapter draft path: the frames the user reads."""
        adapter = _make_real_telegram_adapter()
        consumer = _make_consumer(adapter, transport="auto", chat_type="dm")

        async def _run():
            consumer.on_delta(f"Mimi {EM_DASH} your ")
            task = asyncio.create_task(consumer.run())
            await asyncio.sleep(0.05)
            consumer.on_delta(f"fix is tiny.\n\nThe wait is 3{EN_DASH}5 seconds.")
            await asyncio.sleep(0.05)
            consumer.finish()
            await task

        asyncio.run(_run())

        texts = _draft_payload_texts(adapter)
        assert texts, "expected at least one sendMessageDraft frame"
        for text in texts:
            assert EM_DASH not in text, f"em dash shipped in a draft frame: {text!r}"
            assert EN_DASH not in text, f"en dash shipped in a draft frame: {text!r}"
        # The frames are still the growing reply, not a mangled or truncated one.
        # (MarkdownV2 escapes "." and "-", so assert on the parts the formatter leaves alone.)
        assert "Mimi, your fix is tiny" in texts[-1]
        assert "The wait is 3" in texts[-1] and "5 seconds" in texts[-1]

    def test_draft_frames_stay_prefix_stable_across_ticks(self):
        adapter = _make_real_telegram_adapter()
        consumer = _make_consumer(adapter, transport="auto", chat_type="dm")

        async def _run():
            task = asyncio.create_task(consumer.run())
            for piece in ("Mimi ", f"{EM_DASH} ", "your ", "fix ", "is tiny."):
                consumer.on_delta(piece)
                await asyncio.sleep(0.03)
            consumer.finish()
            await task

        asyncio.run(_run())

        texts = _draft_payload_texts(adapter)
        assert len(texts) >= 2, f"expected several draft frames, got {texts!r}"
        for earlier, later in zip(texts, texts[1:]):
            assert later.startswith(earlier.rstrip()), (
                "a draft frame was not a prefix of the next one — the preview would rewrite "
                f"itself: {earlier!r} -> {later!r}"
            )

    def test_a_raising_normalizer_never_delays_or_drops_a_frame(self, monkeypatch):
        """A measurement/formatting helper must never break a send."""
        import gateway.stream_consumer_transport as transport_mod

        def _boom(text):
            raise RuntimeError("normalizer exploded")

        monkeypatch.setattr(transport_mod, "_normalize_stream_dashes", _boom)
        adapter = _make_edit_adapter()
        consumer = _make_consumer(adapter, transport="edit")

        delivered = asyncio.run(consumer._send_or_edit(REPLY_WITH_DASH, finalize=False))

        assert delivered is True, "the frame was dropped because normalization raised"
        assert adapter.sent_payloads == [REPLY_WITH_DASH], "frame was altered or never sent"


class TestTelegramAdapterDraftSeam:
    def test_send_draft_payload_has_no_dash(self):
        adapter = _make_real_telegram_adapter()

        result = asyncio.run(adapter.send_draft("123", 9, REPLY_WITH_DASH))

        assert result.success is True
        texts = _draft_payload_texts(adapter)
        assert texts, "expected a sendMessageDraft call"
        for text in texts:
            assert EM_DASH not in text
            assert EN_DASH not in text

    def test_send_draft_keeps_dashes_inside_code(self):
        adapter = _make_real_telegram_adapter()
        body = f"x = 'a {EM_DASH} b'"
        content = f"Run `git log {EM_DASH} %an` now {EM_DASH} see:\n\n```python\n{body}\n```"

        result = asyncio.run(adapter.send_draft("123", 9, content))

        assert result.success is True
        text = _draft_payload_texts(adapter)[-1]
        assert f"`git log {EM_DASH} %an`" in text, "inline code span was rewritten"
        assert body in text, "fenced code block was rewritten"
        # ...while the prose aside is still normalized.
        assert f"now {EM_DASH} see" not in text
        assert "now, see" in text

    def test_send_draft_still_ships_when_normalization_raises(self, monkeypatch):
        import plugins.platforms.telegram.adapter as tg_mod

        def _boom(text):
            raise RuntimeError("normalizer exploded")

        monkeypatch.setattr(tg_mod, "_normalize_stream_dashes", _boom)
        adapter = _make_real_telegram_adapter()

        result = asyncio.run(adapter.send_draft("123", 9, REPLY_WITH_DASH))

        assert result.success is True, "the draft frame was dropped by a broken normalizer"
        assert _draft_payload_texts(adapter), "no draft frame reached the Bot API"

    def test_draft_metadata_is_untouched_by_the_guard(self):
        """The guard rewrites text only: routing must come through verbatim."""
        adapter = _make_real_telegram_adapter()
        metadata = {"reply_to_message_id": "42", "message_thread_id": "7"}

        asyncio.run(adapter.send_draft("123", 9, REPLY_WITH_DASH, metadata=metadata))

        sent = adapter._bot.send_message_draft.await_args_list[-1].kwargs
        assert sent["draft_id"] == 9
        assert str(sent["chat_id"]).endswith("123")
        assert metadata == {"reply_to_message_id": "42", "message_thread_id": "7"}

    def test_the_seams_use_the_shared_helper(self):
        """Pin the shared object, not a private copy of the policy."""
        import gateway.stream_consumer_transport as transport_mod
        import plugins.platforms.telegram.adapter as tg_mod

        assert transport_mod._normalize_stream_dashes is normalize_stream_dashes
        assert tg_mod._normalize_stream_dashes is normalize_stream_dashes
