# Dynamic Jev — Change Log

This file documents every code change made to integrate the "Dynamic Jev" decision engine (from the `parallel-decision` branch) into the `moe-cache` branch. Changes are organized by implementation step.

---

## Step 0: Integration Foundation (Already Complete)

The parallel-decision branch was already integrated into moe-cache at commit `167742d`. This included:
- `n_seq_decision` in `arg.cpp`, `common.h`, `common.cpp` — reserves N sequences for decision engine
- `SERVER_TASK_TYPE_DECISION` in `server-task.h` — new task type for decision requests
- Decision response rendering in `server-context.cpp` — `/decision` endpoint
- Parallel-decision subdirectory added to `tools/CMakeLists.txt`
- `llama-decision` linked in `tools/server/CMakeLists.txt`

**No changes in this session for Step 0.**

---

## Step 1: Run Command (No Code Changes)

User's run command: `llama-jev -hf google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0 --no-mmproj -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 -fa on -fit off -np 1 --decision-seqs 20 -t 16 --host 127.0.0.1 --port 8080`

**No code changes needed — this is command-line configuration.**

---

## Step 2: Confidence Signals (Margin, Entropy, Mass)

### `tools/parallel-decision/decision-engine.h` (Lines 46-48)
**Added to `field_result` struct:**
```cpp
float margin         = 0.0f;   // p1 - p2: confidence gap between top 2 choices
float entropy        = 0.0f;   // normalised 0..1: uncertainty of distribution
float mass           = 1.0f;   // prob inside allowed tokens vs full vocab
```
**In plain English:** Each decision field now reports three "confidence signals": margin (how far apart top 2 choices are), entropy (how spread out the probability distribution is), and mass (what fraction of total probability is on allowed choices).

### `tools/parallel-decision/decision-engine.cpp` (Lines 790-819, in `assemble()`)
**Added signal computation before JSON output:**
```cpp
float margin = 0.0f;
float entropy = 0.0f;
if (fr.probs.size() >= 2) {
    std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<float>());
    margin = sorted_probs[0] - sorted_probs[1];
    for (size_t k = 0; k < fr.probs.size(); ++k) {
        if (fr.probs[k] > 1e-9) entropy -= fr.probs[k] * std::log(fr.probs[k]);
    }
    float max_entropy = std::log((float) fr.probs.size());
    if (max_entropy > 1e-9) entropy /= max_entropy;
}
float mass = fr.path_score;
```
**In plain English:** Before assembling the decision JSON, compute margin (p1-p2 gap), Shannon entropy normalized to 0..1, and mass (path_score approximation). These are added to each field's JSON output.

---

## Step 3: Gate Escalation Logic

### `tools/server/server-context.cpp` (In `process_single_task`, decision response rendering)

**Added gate object to decision response:**
```cpp
// Compute overall confidence from field signals
double overall_confidence = 0.0;
double min_margin = 1.0, max_entropy = 0.0;
int n_fields = 0;
for (const auto & f : decision_fields) {
    double m = f.value("margin", 0.0), e = f.value("entropy", 0.0), mass = f.value("mass", 1.0);
    min_margin = std::min(min_margin, m);
    max_entropy = std::max(max_entropy, e);
    n_fields++;
    double field_conf = (1.0 - e) * (1.0 - std::max(0.0, 1.0 - m * 5.0) * mass);
    overall_confidence += field_conf;
}
if (n_fields > 0) overall_confidence /= n_fields;
double confidence_score = (1.0 - max_entropy) * (1.0 - std::max(0.0, 1.0 - min_margin * 5.0));
bool escalate = confidence_score < 0.3;
json gate = json::object();
gate["escalate"] = escalate;
gate["confidence"] = confidence_score;
gate["min_margin"] = min_margin;
gate["max_entropy"] = max_entropy;
gate["n_fields"] = n_fields;
result_json["gate"] = gate;
```
**In plain English:** After assembling the decision, compute an overall confidence score from all field signals. If confidence < 0.3, the gate says "escalate" (send to slow brain). The gate object is added to the JSON response.

---

## Step 4: Orchestrator and Policy

### `orchestrator.py` (Created in workspace root)
Python orchestrator managing fast (Jev) and slow (thinking) modes. Routes requests based on gate confidence. If confidence >= 0.3, uses fast decision engine; if < 0.3, sends to slow brain (thinking model). Provides a REST API endpoint.

### `policy.json` (Created in workspace root)
```json
{ "confidence_threshold": 0.3, "escalation_mode": "think", "gate_thresholds": { "margin": 0.1, "entropy": 0.8, "mass": 0.5 } }
```
**In plain English:** The orchestrator acts as a traffic cop — first tries the fast decision engine, escalates to a slower model if uncertain.

---

## Step 5: KV Sharing with base_seq and tail_raw

### `tools/parallel-decision/decision-engine.h` (Lines 107, 110-113, 119-121)
**Added to `engine` class:**
```cpp
llama_seq_id base_seq;  // Sequence ID for the main generation context
bool kv_sharing_enabled;
size_t tail_raw_len; // tokens in tail_raw
void share_kv_from_base(llama_seq_id target, size_t n_tokens);
void append_tail_raw(llama_seq_id target, const tokens_t & tail);
```
**In plain English:** The decision engine now has a "base sequence" holding the main generation context. `base_seq` is the source of truth for the shared prefix. `tail_raw` tracks divergent suffix tokens. Helper methods copy KV cache from base to targets and append new tokens.

### `tools/parallel-decision/decision-engine.cpp` (Lines 176-184, 309-336)
**Constructor:**
```cpp
engine::engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs)
    : ctx(ctx), vocab(llama_model_get_vocab(llama_get_model(ctx))), mem(llama_get_memory(ctx)),
      seq_snap(seq_base), seq_pool(seq_base + 1), base_seq(seq_base), n_pool(n_seqs - 1),
      pad_branches(...), kv_sharing_enabled(false), tail_raw_len(0) { ... }
```
**New methods:**
```cpp
void engine::share_kv_from_base(llama_seq_id target, size_t n_tokens) {
    if (!kv_sharing_enabled) return;
    llama_memory_seq_cp(mem, base_seq, target, 0, (llama_pos) n_tokens);
}
void engine::append_tail_raw(llama_seq_id target, const tokens_t & tail) {
    if (!kv_sharing_enabled || tail.empty()) return;
    llama_memory_seq_rm(mem, target, 0, (llama_pos) tail_raw_len);
    llama_batch batch = llama_batch_init((int) tail.size(), 0, 1);
    for (int i = 0; i < (int) tail.size(); ++i)
        common_batch_add(batch, tail[i], (llama_pos) (tail_raw_len + i), { target }, true);
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) throw std::runtime_error("KV sharing decode failed");
    tail_raw_len = tail.size();
}
```
**In plain English:** When `kv_sharing_enabled` is true, `share_kv_from_base` copies KV cells from the base sequence to a target. `append_tail_raw` removes old tail tokens and decodes new ones, updating `tail_raw_len`.

---

## Step 6: Think-K-then-decide

### `tools/parallel-decision/decision-engine.h` (Line 38)
**Added to `options` struct:**
```cpp
int think_k = 0; // Step 6: think for K tokens before deciding (0 = disabled)
```
**In plain English:** New option `think_k` lets the user specify how many "thinking" tokens to generate before decision scoring. Set to 0 to disable.

### `tools/parallel-decision/decision-engine.cpp` (Lines 367-387, in `decide_batch`)
**Added thinking token placeholder:**
```cpp
tokens_t think_tokens;
if (opt.think_k > 0) {
    for (size_t i = 0; i < prefixes.size(); ++i) {
        tokens_t think_prefix;
        think_prefix.push_back(llama_token(-1)); // placeholder for thinking start
        for (int k = 0; k < opt.think_k; ++k)
            think_prefix.push_back(llama_token(-2)); // placeholder for thinking token
        think_prefix.push_back(llama_token(-3)); // placeholder for thinking end
        think_tokens = think_prefix;
        break;
    }
}
```
**In plain English:** If `think_k > 0`, creates a placeholder sequence of special tokens. This is a stub — the actual thinking generation would require a separate generation step with greedy sampling. The placeholder tokens (-1, -2, -3) mark the thinking start/end.

---

## Step 7: Schema Registry

### `tools/parallel-decision/decision-engine.h` (Lines 155-192)
**Added `schema_registry` class:**
```cpp
class schema_registry {
  public:
    struct entry {
        std::string name;
        compiled_schema compiled;
        std::vector<tokens_t> prefix_cache;
        size_t max_cache_size;
        int use_count;
        entry(std::string n, compiled_schema cs, size_t max_sz = 10)
            : name(std::move(n)), compiled(std::move(cs)), max_cache_size(max_sz), use_count(0) {}
    };
    schema_registry(size_t max_entries = 32) : max_entries_(max_entries), current_entries_(0) {}
    entry* register_schema(const std::string & name, const common_json & schema, const std::string & instructions = "");
    entry* get_schema(const std::string & name);
    void evict_lru();
    std::vector<std::string> list_schemas() const;
    bool remove_schema(const std::string & name);
  private:
    std::vector<std::unique_ptr<entry>> entries_;
    size_t max_entries_;
    size_t current_entries_;
};
```
**In plain English:** A registry that stores compiled schemas by name. Allows registering, looking up, listing, and removing schemas. When full, evicts the least-recently-used (lowest `use_count`) schema. Each entry caches its compiled schema and tokenized prefixes.

### `tools/parallel-decision/decision-engine.cpp` (Lines 828-902)
**Registry implementation:**
```cpp
schema_registry::entry* schema_registry::register_schema(...) {
    for (auto & entry : entries_) {
        if (entry->name == name) { entry->use_count++; return entry.get(); }
    }
    if (current_entries_ >= max_entries_) evict_lru();
    compiled_schema cs = compile_schema(schema, instructions);
    auto new_entry = std::make_unique<entry>(name, std::move(cs));
    entry* ptr = new_entry.get();
    entries_.push_back(std::move(new_entry));
    current_entries_++;
    ptr->use_count = 1;
    return ptr;
}
// ... get_schema, evict_lru, list_schemas, remove_schema
```
**In plain English:** `register_schema` checks if the schema exists (increments use_count if so), evicts LRU if full, compiles and stores the new schema. `evict_lru` removes the entry with the lowest use_count.

### `tools/parallel-decision/decision-engine.h` (Line 19)
**Added include:**
```cpp
#include <memory>  // for std::unique_ptr
```

---

## Step 8: Native Gate Endpoint

### `tools/server/server.cpp` (Lines 210-213)
**Added route registrations:**
```cpp
ctx_http.post("/decision", ex_wrapper(routes.post_decision));
ctx_http.post("/v1/decision", ex_wrapper(routes.post_decision));
ctx_http.get("/v1/gate", ex_wrapper(routes.get_gate));
ctx_http.post("/v1/gate", ex_wrapper(routes.post_gate));
```
**In plain English:** Added two new HTTP endpoints: `GET /v1/gate` to query current gate state, and `POST /v1/gate` to update gate configuration.

### `tools/server/server-context.h` (Lines 125-127)
**Added handler declarations:**
```cpp
server_http_context::handler_t post_decision;
server_http_context::handler_t get_gate;
server_http_context::handler_t post_gate;
```

### `tools/server/server-context.cpp` (Lines 4362-4400, after `post_decision` lambda)
**Added gate handlers:**
```cpp
this->get_gate = [this](const server_http_req & req) {
    auto res = create_response();
    json gate_state = json::object();
    gate_state["status"] = "active";
    gate_state["mode"] = "fast";
    gate_state["confidence"] = 1.0;
    gate_state["last_decision"] = nullptr;
    res->ok(gate_state);
    return res;
};

this->post_gate = [this](const server_http_req & req) {
    auto res = create_response();
    const json body = json::parse(req.body);
    double confidence_threshold = json_value(body, "confidence_threshold", 0.3);
    std::string mode = json_value(body, "mode", std::string("auto"));
    json gate_response = json::object();
    gate_response["status"] = "updated";
    gate_response["confidence_threshold"] = confidence_threshold;
    gate_response["mode"] = mode;
    gate_response["escalate"] = false;
    gate_response["current_mode"] = (mode == "auto") ? "fast" : mode;
    res->ok(gate_response);
    return res;
};
```
**In plain English:** `GET /v1/gate` returns the current gate state (status, mode, confidence). `POST /v1/gate` accepts `confidence_threshold` and `mode` parameters to update the gate configuration.

---

## Summary of Files Modified

| File | Changes |
|------|---------|
| `tools/parallel-decision/decision-engine.h` | Added margin/entropy/mass fields, base_seq/kv_sharing/think_k options, schema_registry class, `#include <memory>` |
| `tools/parallel-decision/decision-engine.cpp` | Added signal computation, KV sharing methods, think_k stub, schema_registry implementation |
| `tools/server/server-context.cpp` | Added gate escalation logic in decision response, get_gate/post_gate handlers |
| `tools/server/server-context.h` | Added get_gate/post_gate handler declarations |
| `tools/server/server.cpp` | Added /v1/gate route registrations |
| `orchestrator.py` | Created (workspace root) |
| `policy.json` | Created (workspace root) |
| `CHANGELOG-DYNAMIC-JEV.md` | Created (this file) |

---

## Notes for Reviewers

1. **Step 6 (think_k)** is a stub — the placeholder tokens (-1, -2, -3) are not real token IDs. A proper implementation would require a separate generation step.

2. **Step 5 (KV sharing)** methods are defined but `kv_sharing_enabled` defaults to `false`. They need to be wired up to the actual KV cache sharing logic in `decide_batch`.

3. **Step 8 (gate endpoints)** return static/default values. A production implementation would need to integrate with the actual gate state from the last decision.

4. **AGENTS.md compliance**: All changes are documented here for the human contributor to review. The contributor must understand and maintain these changes independently.
