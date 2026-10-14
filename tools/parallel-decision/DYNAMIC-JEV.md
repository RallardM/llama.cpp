# Dynamic Jev chat auto-route

Clients such as benchmark harnesses and agents call `/v1/chat/completions`. The decision engine lives at `/v1/decision`. Without a link between them a client never reaches it, and scores are identical with and without the decision engine. `--dynamic-jev` is that link.

## Flags

- `--dynamic-jev`: route every chat request through the decision engine first. Reserves 12 decision sequences when `--decision-seqs` is not set.
- `--dynamic-jev-policy FILE`: JSON policy (implies `--dynamic-jev`). Top-level keys replace the built-in ones. See `dynamic-jev-policy.example.json`.

## What happens to a chat request

1. The last user message is clipped (first two thirds and last third of `max_chars`, default 4000) and sent to the decision engine with the policy's router schema. The router answers, for example, `difficulty` and `kind`, with confidence signals.
2. The first rung whose `when` matches the answer is chosen. Rungs run from lowest to highest. If the router's gate fires (low probability, margin or mass), the request moves up `escalate_steps` rungs.
3. The rung sets, on the request: `enable_thinking`, `reasoning_budget_tokens` (-1 = unlimited), `max_tokens`, and `system_hint` (added to the system message).
4. What the client already set wins, unless the policy says `"override_client": true`.
5. The request continues through the normal completion path.

Any failure (router error, no free KV, bad policy value) leaves the request unchanged and is counted in `failed_open`.

If the model's chat template has no thinking mode, `enable_thinking` and `reasoning_budget_tokens` do nothing. Use `system_hint` and `max_tokens` per rung instead.

## Per-request controls (in the chat request body)

- `"dynamic_jev": false`: skip the router for this request.
- `"dynamic_jev": {"force_rung": "R1"}`: skip the router and use that rung. An unknown name fails open.

## Agent requests (tools)

A request is an agent request when it has a non-empty `tools` list, or its messages hold a `tool` role or `tool_calls`. Policy key `on_tools` decides what to do with it (default `"skip"`):

- `"skip"`: leave the request untouched. Header `skipped=agent; on_tools=skip`. Counters `skipped` and `agent_skipped`.
- `"protect"`: route, but never turn thinking off, add a system hint or cap the answer, so the model can still emit tool calls. Header carries `agent=protect`. Counter `agent_protected`.
- `"route"`: route exactly like plain chat (the old behaviour).

Per request: `"dynamic_jev": {"on_tools": "protect"}` overrides the policy. `force_rung` still applies to agent requests in every mode. Compare the three modes on your harness benchmark and keep the best.

## Seeing what happened

- Response header `X-Dynamic-Jev`, for example `rung=R1; difficulty=medium; kind=code; escalated=0; router_ms=412; cached=0; set=enable_thinking,reasoning_budget_tokens; client_wins=max_tokens`.
- Server log line `dynamic-jev: ...` for every routed request.
- `GET /dynamic-jev`: the policy in use and counters (`routed`, `cache_hits`, `failed_open`, `skipped`, `agent_skipped`, `agent_protected`, `router_ms_avg`, `by_rung`, `last_error`). If `routed` stays 0 while a client runs, the client is not reaching `/v1/chat/completions` or the server was started without the flag.

## Limits

- Only `/v1/chat/completions` (and its `/chat/completions` alias). Anthropic, Responses and plain completions are not routed.
- The router reads the clipped user text once more. Answers are cached per text (128 entries), so an agent that resends the same task each step pays once.
- The router needs free KV cells next to the slot caches. Use `-c 32768` or more for agent workloads. With `-c 8192` a long cached slot can leave no room and the router fails open.
- A wrong router answer costs quality or time, not correctness of the format. Tune thresholds and rungs on your own benchmark.

## Benchmark A/B protocol

Run the same benchmark four times against the same server and compare score, truncation percent and minutes, per difficulty and category:

1. Every request with `"dynamic_jev": {"force_rung": "R0"}`.
2. The same with `"R1"`.
3. The same with `"R2"`.
4. Automatic routing (no `dynamic_jev` key).

If the benchmark client sets `max_tokens` or `enable_thinking` itself, add `"override_client": true` to the policy, otherwise the client's values win and the rungs change nothing (the header shows `client_wins=`).

Routing helps only if the rungs differ in outcome: if R0 is as good as R2 on easy problems and much shorter, routing saves time; if R2 beats R0 on hard problems, routing keeps that quality. Without such a difference the automatic run will score the same as the best fixed rung.

## Tests

`tests/test_http.py` (Python standard library only) checks the decision contract, slot mode against normal mode, error codes, and the routing, headers and stats of this feature against a running server. Start the server with `tests/dynamic-jev-policy.test.json` (command at the top of the script).
