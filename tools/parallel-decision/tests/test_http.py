#!/usr/bin/env python3
"""HTTP end-to-end test for /v1/decision, slot mode and the --dynamic-jev chat route.

Start the server first, with the test policy:
  llama-server -m MODEL.gguf -c 2048 -np 1 --decision-seqs 12 --jinja --dynamic-jev-policy tools/parallel-decision/tests/dynamic-jev-policy.test.json --port 8099
Then:  python tools/parallel-decision/tests/test_http.py http://127.0.0.1:8099

Works with any model: it checks plumbing and contracts, not answer quality.
Python standard library only.
"""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8099"
FAILS = []


def call(method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, r.headers, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            j = json.loads(raw)
        except Exception:
            j = {"raw": raw}
        return e.code, e.headers, j


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


SCHEMA = {
    "phase": {"type": "enum", "choices": ["read", "locate", "plan", "write", "verify"], "description": "Which phase is this step?"},
    "needs_code": {"type": "boolean", "description": "Does the next step need new code?"},
    "level": {"type": "integer", "minimum": 0, "maximum": 9, "description": "Difficulty from 0 to 9"},
}

print("== 1. decision endpoint, normal mode")
st, _, r = call("POST", "/v1/decision", {"schema": SCHEMA, "contexts": ["Fix the failing parser test.", "Add a dark mode toggle."], "mass": True})
check(st == 200, "POST /v1/decision answers 200 (got %s)" % st)
res = (r or {}).get("results", [])
check(len(res) == 2, "one result per context")
if len(res) == 2:
    f = res[0]["fields"]
    check(set(res[0]["decision"].keys()) == set(SCHEMA.keys()), "decision holds every field")
    check(all("probability" in f[k] for k in f), "every field has a probability")
    check(all("margin" in f[k] and "entropy" in f[k] for k in f), "tree fields carry margin and entropy")
    check(all(0.0 < f[k]["mass"] <= 1.0 for k in f), "mass requested: present and in (0, 1]")
    check("results" in r and "timings" in r and "usage" in r and "gate" not in r, "response shape: results, timings, usage, no top-level gate")
    check("gate" not in res[0], "no gate object when the request has none")

st, _, r = call("POST", "/v1/decision", {"schema": SCHEMA, "contexts": ["Fix the failing parser test."], "gate": {"min_prob": 0.999999}})
g = (r or {}).get("results", [{}])[0].get("gate", {})
check(st == 200 and g.get("escalate") is True and len(g.get("reasons", [])) > 0, "an impossible threshold escalates, with reasons")
st, _, r = call("POST", "/v1/decision", {"schema": SCHEMA, "contexts": ["Fix the failing parser test."], "gate": {}})
check((r or {}).get("results", [{}])[0].get("gate", {}).get("escalate") is False, "an empty gate never escalates")

print("== 2. decision endpoint, errors (must be HTTP 400)")
bad = {"c": {"type": "enum", "description": "x", "values": ["a", "b"]}}
for name, body in [
    ("enum key 'values'", {"schema": bad, "contexts": ["x"]}),
    ("missing schema", {"contexts": ["x"]}),
    ("empty contexts", {"schema": SCHEMA, "contexts": []}),
    ("slot out of range", {"schema": SCHEMA, "contexts": ["x"], "slot": 99}),
    ("raw without slot", {"schema": SCHEMA, "contexts": ["x"], "raw": True}),
]:
    st, _, r = call("POST", "/v1/decision", body)
    check(st == 400, "%s -> HTTP %s" % (name, st))
st, _, r = call("POST", "/v1/decision", {"schema": SCHEMA, "contexts": ["x"], "slot": 0})
check(st == 400, "empty slot -> HTTP %s (%s)" % (st, str((r or {}).get("error", {}).get("message", ""))[:60]))

print("== 3. describe")
st, _, d = call("POST", "/v1/decision", {"schema": SCHEMA, "describe": True})
sys_text = (d or {}).get("system_text", "")
check(st == 200 and "needs_code" in sys_text, "describe returns the field catalogue")

print("== 4. slot mode equals normal mode")
SENT = "SENTINEL_X"
st, _, t = call("POST", "/apply-template", {"messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": SENT}]})
prompt = (t or {}).get("prompt", "")
at = prompt.find(SENT)
check(st == 200 and at > 0, "apply-template renders the prefix")
head = prompt[:at]
CTX = "Fix the failing parser test."
st, _, a = call("POST", "/v1/decision", {"schema": SCHEMA, "contexts": [CTX], "mass": True})
st2, _, p = call("POST", "/completion", {"prompt": head, "n_predict": 1, "id_slot": 0, "cache_prompt": True})
check(st2 == 200, "prime slot 0 with the prefix via /completion")
st3, _, tok = call("POST", "/tokenize", {"content": head, "add_special": True})
n_tok = len((tok or {}).get("tokens", []))
st4, _, b = call("POST", "/v1/decision", {"schema": SCHEMA, "contexts": [CTX], "mass": True, "slot": 0, "catalogue": False})
check(st4 == 200, "decision on slot 0 answers 200 (got %s)" % st4)
if st4 == 200 and st == 200:
    check(b.get("slot") == 0, "reply names the slot")
    check(b.get("slot_tokens") == n_tok, "slot_tokens (%s) equals the prefix token count (%s)" % (b.get("slot_tokens"), n_tok))
    fa, fb = a["results"][0]["fields"], b["results"][0]["fields"]
    worst = max(abs(fa[k]["probability"] - fb[k]["probability"]) for k in fa)
    same = all(a["results"][0]["decision"][k] == b["results"][0]["decision"][k] for k in fa)
    check(same, "same winners in slot mode and normal mode")
    check(worst < 0.02, "probabilities agree (max difference %.4f)" % worst)
    st5, _, c = call("POST", "/v1/decision", {"schema": SCHEMA, "contexts": [CTX], "mass": True, "slot": 0, "catalogue": False})
    check(st5 == 200 and c["results"][0]["fields"] == fb, "a second slot call is identical: the slot was not disturbed")
    st6, _, d6 = call("POST", "/v1/decision", {"schema": SCHEMA, "contexts": [CTX]})
    check(st6 == 200 and d6.get("usage", {}).get("cached_tokens", 0) > 0, "normal mode still hits its prefix cache after slot calls")

print("== 5. dynamic-jev chat route")
st, _, s = call("GET", "/dynamic-jev")
enabled = bool((s or {}).get("enabled"))
check(st == 200 and enabled, "/dynamic-jev reports enabled")
if enabled:
    CHAT = {"messages": [{"role": "user", "content": "Write a function that reverses a string."}], "max_tokens": 6, "temperature": 0}
    st, h, base = call("POST", "/v1/chat/completions", dict(CHAT, dynamic_jev=False))
    check(st == 200 and h.get("X-Dynamic-Jev") is None, "dynamic_jev:false -> served, no routing header")
    base_prompt = (base or {}).get("usage", {}).get("prompt_tokens", 0)

    st, h, r = call("POST", "/v1/chat/completions", {"messages": CHAT["messages"], "temperature": 0, "dynamic_jev": {"force_rung": "R0"}})
    tag = h.get("X-Dynamic-Jev") or ""
    check(st == 200 and "rung=R0" in tag and "forced=1" in tag, "force_rung R0 -> header '%s'" % tag)
    check("set=" in tag and "max_tokens" in tag and "enable_thinking" in tag and "system_hint" in tag, "R0 sets enable_thinking, max_tokens and system_hint")
    check((r or {}).get("usage", {}).get("completion_tokens", 99) <= 3, "R0 token cap of 3 applied (completion_tokens=%s)" % (r or {}).get("usage", {}).get("completion_tokens"))
    check((r or {}).get("usage", {}).get("prompt_tokens", 0) > base_prompt, "the system hint added prompt tokens (%s > %s)" % ((r or {}).get("usage", {}).get("prompt_tokens"), base_prompt))

    st, h, r = call("POST", "/v1/chat/completions", dict(CHAT, dynamic_jev={"force_rung": "R0"}))
    tag = h.get("X-Dynamic-Jev") or ""
    check(st == 200 and "client_wins=max_tokens" in tag, "a client max_tokens wins over the rung -> '%s'" % tag)
    check((r or {}).get("usage", {}).get("completion_tokens", 0) > 3, "the client's cap of 6 was kept (completion_tokens=%s)" % (r or {}).get("usage", {}).get("completion_tokens"))

    st, h, r = call("POST", "/v1/chat/completions", {"messages": CHAT["messages"], "max_tokens": 4, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}, "dynamic_jev": {"force_rung": "R0"}})
    tag = h.get("X-Dynamic-Jev") or ""
    check("client_wins=enable_thinking" in tag, "a client enable_thinking wins over the rung -> '%s'" % tag)

    st, h, r = call("POST", "/v1/chat/completions", dict(CHAT, dynamic_jev={"force_rung": "NOPE"}))
    check(st == 200 and h.get("X-Dynamic-Jev") is None, "an unknown force_rung fails open: the request is still served")

    st, h, r1 = call("POST", "/v1/chat/completions", dict(CHAT))
    tag1 = h.get("X-Dynamic-Jev") or ""
    check(st == 200 and "rung=" in tag1 and "difficulty=" in tag1 and "router_ms=" in tag1, "auto route: router ran -> '%s'" % tag1)
    st, h, r2 = call("POST", "/v1/chat/completions", dict(CHAT))
    check("cached=1" in (h.get("X-Dynamic-Jev") or ""), "the same request again reuses the router answer (cached=1)")

    TOOLS = [{"type": "function", "function": {"name": "get_time", "description": "Get the time", "parameters": {"type": "object", "properties": {}}}}]
    st, h, r = call("POST", "/v1/chat/completions", dict(CHAT, tools=TOOLS))
    tag = h.get("X-Dynamic-Jev") or ""
    check(st == 200 and "skipped=agent" in tag, "on_tools default (skip): a request with tools is not routed -> '%s'" % tag)

    st, h, r = call("POST", "/v1/chat/completions", dict(CHAT, tools=TOOLS, dynamic_jev={"force_rung": "R0", "on_tools": "protect"}))
    tag = h.get("X-Dynamic-Jev") or ""
    check(st == 200 and "agent=protect" in tag and "max_tokens" not in tag and "system_hint" not in tag and "enable_thinking" not in tag, "on_tools protect: R0 loses cap, hint and thinking-off -> '%s'" % tag)

    st, h, r = call("POST", "/v1/chat/completions", dict(CHAT, tools=TOOLS, dynamic_jev={"force_rung": "R0", "on_tools": "route"}))
    tag = h.get("X-Dynamic-Jev") or ""
    check(st == 200 and "agent=route" in tag and "system_hint" in tag, "on_tools route: the rung applies as for plain chat -> '%s'" % tag)

    st, h, r = call("POST", "/v1/chat/completions", dict(CHAT, dynamic_jev={"on_tools": "bogus"}))
    check(st == 200 and h.get("X-Dynamic-Jev") is None, "a bad on_tools value fails open")

    st, _, s = call("GET", "/dynamic-jev")
    stats = (s or {}).get("stats", {})
    check(stats.get("routed", 0) >= 6 and sum(stats.get("by_rung", {}).values()) == stats.get("routed"), "stats count routed requests per rung: %s" % json.dumps(stats.get("by_rung")))
    check(stats.get("failed_open", 0) >= 2 and ("NOPE" in str(stats.get("last_error", "")) or "bogus" in str(stats.get("last_error", ""))), "stats record the fail-open (last_error: %s)" % stats.get("last_error"))

print()
print("ALL %s CHECKS PASSED" % "HTTP" if not FAILS else "%d CHECK(S) FAILED:\n  - %s" % (len(FAILS), "\n  - ".join(FAILS)))
sys.exit(1 if FAILS else 0)
