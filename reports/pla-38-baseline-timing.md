# PLA-38 baseline timing report

Generated from local representative timing fixtures; no network, model, routing, deploy, restart, commit, or push.

## Method
- 3 representative gateway-turn fixtures, 30 repetitions each (90 stage samples/profile).
- Each fixture exercises the existing `TurnTiming` phase names with local bounded waits standing in for quiet window, queue, context, API, tool, and delivery work.
- Values are wall-clock measurements from `time.perf_counter()` around each fixture stage; they are a local baseline, not provider latency.
- Percentile: nearest-rank p90; worst is maximum.

## Stage timings (milliseconds)

### telegram_cached_no_tools
| Stage | Median | P90 | Worst |
|---|---:|---:|---:|
| `telegram_quiet_window` | 5.12 | 5.14 | 5.24 |
| `adapter_wait` | 1.10 | 1.13 | 1.16 |
| `queue_wait` | 1.10 | 1.12 | 1.14 |
| `gateway_prep` | 2.10 | 2.12 | 2.14 |
| `cached_agent_model_resolution` | 1.10 | 1.12 | 1.13 |
| `context_memory_pre_llm` | 4.12 | 4.15 | 4.18 |
| `api_start` | 1.10 | 1.12 | 1.13 |
| `api_first_chunk` | 15.13 | 15.14 | 15.19 |
| `api_end` | 40.13 | 40.16 | 40.21 |
| `tool_time` | 0.07 | 0.08 | 0.12 |
| `final_delivery` | 3.12 | 3.15 | 3.17 |

### telegram_cached_one_tool
| Stage | Median | P90 | Worst |
|---|---:|---:|---:|
| `telegram_quiet_window` | 5.14 | 5.15 | 5.16 |
| `adapter_wait` | 1.11 | 1.13 | 1.16 |
| `queue_wait` | 1.10 | 1.12 | 1.19 |
| `gateway_prep` | 2.11 | 2.13 | 2.20 |
| `cached_agent_model_resolution` | 1.10 | 1.13 | 1.13 |
| `context_memory_pre_llm` | 4.12 | 4.14 | 4.17 |
| `api_start` | 1.10 | 1.13 | 1.16 |
| `api_first_chunk` | 15.13 | 15.16 | 15.19 |
| `api_end` | 40.14 | 40.17 | 40.22 |
| `tool_time` | 20.14 | 20.17 | 20.26 |
| `final_delivery` | 3.13 | 3.16 | 5.81 |

### telegram_cold_context
| Stage | Median | P90 | Worst |
|---|---:|---:|---:|
| `telegram_quiet_window` | 5.14 | 5.24 | 5.30 |
| `adapter_wait` | 1.11 | 1.20 | 1.70 |
| `queue_wait` | 1.11 | 1.16 | 1.51 |
| `gateway_prep` | 2.11 | 2.14 | 2.16 |
| `cached_agent_model_resolution` | 8.13 | 8.17 | 8.59 |
| `context_memory_pre_llm` | 12.14 | 12.16 | 13.54 |
| `api_start` | 1.11 | 1.28 | 1.70 |
| `api_first_chunk` | 15.14 | 15.28 | 15.77 |
| `api_end` | 40.14 | 40.23 | 41.10 |
| `tool_time` | 0.07 | 0.12 | 0.43 |
| `final_delivery` | 3.14 | 3.16 | 3.37 |

## Bottlenecks
- `api_end` is the dominant measured stage in every fixture (the local API stand-in is 40 ms by design); it is the first optimization target only for this fixture baseline.
- `tool_time` is the second bottleneck in the one-tool fixture (20 ms); it is absent in no-tool and cold-context fixtures.
- `context_memory_pre_llm` and `cached_agent_model_resolution` are the largest pre-LLM stages in the cold-context fixture.
- Telegram quiet-window time is a fixed 5 ms fixture component and dominates ingress before adapter/queue work.
- `api_first_chunk` is a boundary mark, not a standalone duration in the production carrier; the report treats the interval ending at that mark as the observed stage.

## Coverage / limitations
- Production marks: `telegram_quiet_window`, `adapter_wait`, `queue_wait` (Telegram adapter); `gateway_prep`, `cached_agent_model_resolution`, `context_memory_pre_llm`, `api_start`, `api_first_chunk`, `api_end`, `tool_time` (`TurnRunner`); `final_delivery` via `TurnTiming.finish_delivery()` at `_hmwa_deliver_turn_response` (gateway deliver-decision, before adapter.send / Telegram RTT).
- `log_terminal()` is idempotent. Success-path emit is deferred until `finish_delivery()` so `final_delivery` is present. Empty-response and exception paths still emit without that mark (`final_delivery_ms=na`).
- The carrier still logs cumulative inter-mark intervals, not independent stopwatch spans.
- Test command: `python -m pytest -q tests/gateway/test_turn_timing.py` — **3 passed** (venv with pytest).

## Remaining live-latency gap (honest)
These fixture numbers are **not** live Telegram/user-perceived latency. Still unmeasured in production:

- Real Telegram quiet-window (typically hundreds of ms to seconds, not the 5 ms fixture wait).
- Provider TTFT / full completion (`api_first_chunk` / `api_end` are local stand-ins of ~15 ms / ~40 ms; live is usually seconds).
- Real tool execution (fixture 20 ms vs live tool/network time).
- Adapter `send` / `edit_message` network RTT after `finish_delivery()` — **still outside the carrier**.
- Persist/transcript/hooks between `api_end` and deliver-decision are folded into the `final_delivery` interval, not isolated.
- Non-Telegram platforms do not attach `_gateway_turn_timing` at ingress; they only get a bound carrier if `bind_turn_timing` ran.

Closing live gap needs DEBUG `gateway.timing` samples from a running gateway on real turns, not another local fixture pass. No deploy in this lab.
