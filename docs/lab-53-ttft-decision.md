# LAB-53 decision: hosted lane models do not clear <2s

Measured 2026-09-23 on this host, through `gateway.run_turn_fast_lane.try_fast_lane` (quality gate forced off in the probe config only; live config was not changed). Three streaming samples each, tiny prompt `ok`. Median is the middle sample. No new accounts.

## Decision

A local tiny model is required for a sub-second / <2s first-visible-token target. None of the probed hosted models reach <2s end-to-end on median TTFT alone, before Telegram delivery. The xAI subscription route and the OpenCode Go subscription are already on this host; the gap is model latency, not a missing account.

Do not treat the currently configured local lane (`ollama` / `llama3.2:1b`) as that tiny model. It does not meet the target either, and llama.cpp is not running.

## Hosted lane TTFT (verified)

| Route | Model | Samples ttft_ms | Median ttft_ms | Result |
|---|---|---|---|---|
| xai-oauth | grok-4.7 | 2369.0, 2196.3, 1752.7 | 2196.3 | 3/3 replies |
| xai local proxy `:8645` | grok-4.7 | 2897.2, 2632.6, 2686.0 | 2686.0 | 3/3 replies |
| opencode-go | mimo-v2.6-pro | 7393.5, 3085.6, 2161.3 | 3085.6 | 3/3 replies |
| opencode-go | deepseek-v4-pro | 4080.7, 3751.7, 3063.3 | 3751.7 | 3/3 replies |
| opencode-go | deepseek-v4-flash | 3466.3, none (20s budget), 8852.5 | 6159.4 of the two successes | 2/3 replies |
| opencode-go | deepseek-v4.1 | n/a | n/a | unavailable |

deepseek-v4.1 with `x-opencode-session` set: `400 Model is unavailable`. The same call without the session header returns `400 MissingSessionID`, so the unavailable result is the model, not affinity. mimo-v2.6-pro is offered on opencode-go (the static catalog still lists mimo-v2.5-pro; the live relay served v2.6-pro).

Best hosted path is grok-4.7 via xai-oauth (median TTFT 2196 ms). One sample was 1753 ms. That is still the model call only.

## What the running gateway does now (verified from logs, not a restart)

Live `gateway.telegram.fast_lane` is `ollama` / `llama3.2:1b` with `quality_gate.enabled: true`. Top-level `fast_lane` (opencode-go / deepseek-v4-pro) is not what production reads.

After the 08:29 gateway start, a social DM (`im working`, 08:39) did not show a token in under 2s:

- lane fallback `ready_ms=12142.2` (ollama, no token)
- a later lane draft `ttft_ms=371.5` was discarded (`quality_gate` escalate, `fallback=true`)
- `stream_first_visible since_open_ms=18358.3` (`since_first_delta_ms=748.3`)
- `response ready` at 35.1s

A direct ollama `llama3.2:1b` stream of `ok` (one sample, not the lane wrapper): ttft 2080.1 ms, ready 5683.9 ms. `qwen3:1.7b` produced no content token in 25.3 s. Nothing is listening on llama.cpp ports 8080/8081/8000.

## Quiet window

The bypass is on main and unit-tested: a single lane-eligible DM returns delay 0 when `conversational_dm_batching` is on. Live config has that flag on. Inference, not a receive-timestamp measurement: recent short-DM flush lines sit 130–560 ms before the inbound log line, which is not a 2s wait. The 18s felt latency above is the lane/model/gate path, not the quiet window.

## Not done

- Live config was not changed and the gateway was not restarted.
- No llama.cpp install or local-model swap. That would be a config/runtime change, not a code change, and the probed hosted models do not make it unnecessary.
