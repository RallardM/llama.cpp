// Functional test (CPU, tiny random model): decisions on a slot must equal decisions with a rendered prefix,
// leave the slot's KV untouched, and leave no branch sequences behind.
// Build and run: see tools/parallel-decision/tests/README.md
#include "decision-engine.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#ifdef DJ_NO_COMMON
// stand-ins for the three libcommon helpers the engine calls (only when built without libcommon)
void common_batch_clear(struct llama_batch & batch) { batch.n_tokens = 0; }
void common_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token[batch.n_tokens]    = id;
    batch.pos[batch.n_tokens]      = pos;
    batch.n_seq_id[batch.n_tokens] = (int32_t) seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}
std::vector<llama_token> common_tokenize(const struct llama_vocab * vocab, const std::string & text, bool add_special, bool parse_special) {
    int n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, add_special, parse_special);
    std::vector<llama_token> out((size_t) std::max(n, 0));
    int r = llama_tokenize(vocab, text.data(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, parse_special);
    out.resize(r < 0 ? 0 : (size_t) r);
    return out;
}

#endif

using namespace llama_decision;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); g_fail++; } } while (0)

static double cmp(const batch_result & a, const batch_result & b, const char * what) {
    double worst = 0;
    CHECK(a.items.size() == b.items.size(), "%s: item count", what);
    for (size_t i = 0; i < a.items.size() && i < b.items.size(); ++i) {
        for (size_t f = 0; f < a.items[i].fields.size(); ++f) {
            const auto & x = a.items[i].fields[f];
            const auto & y = b.items[i].fields[f];
            CHECK(x.winner == y.winner, "%s: ctx %zu field %zu winner %d vs %d", what, i, f, x.winner, y.winner);
            worst = std::max(worst, (double) std::fabs(x.path_score - y.path_score));
            CHECK(x.probs.size() == y.probs.size(), "%s: probs size", what);
            for (size_t k = 0; k < x.probs.size() && k < y.probs.size(); ++k) {
                worst = std::max(worst, (double) std::fabs(x.probs[k] - y.probs[k]));
            }
            if (x.margin >= 0 || y.margin >= 0) worst = std::max(worst, (double) std::fabs(x.margin - y.margin));
            if (x.entropy >= 0 || y.entropy >= 0) worst = std::max(worst, (double) std::fabs(x.entropy - y.entropy));
            CHECK(x.has_mass == y.has_mass, "%s: has_mass differs", what);
            if (x.has_mass && y.has_mass) worst = std::max(worst, (double) std::fabs(x.log_mass - y.log_mass));
        }
    }
    return worst;
}

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::printf("model load failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_slots = 1, n_dec = 12;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_seq_max = n_slots + n_dec; cp.kv_unified = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::printf("context failed\n"); return 2; }
    llama_memory_t mem = llama_get_memory(ctx);

    engine eng(ctx, (llama_seq_id) n_slots, n_dec);

    const auto schema = common_json::parse(R"({
      "phase":      {"type":"enum","choices":["read","locate","plan","write","verify"],"description":"Which phase of the task is this?"},
      "needs_code": {"type":"boolean","description":"Does the next step need new code?"},
      "risk":       {"type":"enum","choices":["low","mid","high"],"description":"How risky is a wrong answer?"},
      "level":      {"type":"integer","minimum":0,"maximum":9,"description":"Difficulty from 0 to 9"}
    })");
    const auto cs = compile_schema(schema, "Be precise.");

    const std::vector<std::string> ctxs = {
        "The project has a parser module and one failing test in tokenizer_test.",
        "Please add a dark mode toggle to the settings page.",
        "Rename the variable count to total in three files.",
    };
    std::string head; std::vector<std::string> tails;
    for (const auto & c : ctxs) {
        auto pr = render_prompt(nullptr, false, cs.system_text, c);
        head = pr.first; tails.push_back(pr.second);
    }

    // prime the "slot" (seq 0) with the shared prefix, exactly like a server slot that already read the project
    const auto head_tok = common_tokenize(vocab, head, true, true);
    {
        llama_batch b = llama_batch_init(512, 0, 1);
        size_t done = 0;
        while (done < head_tok.size()) {
            common_batch_clear(b);
            const size_t n = std::min<size_t>(512, head_tok.size() - done);
            for (size_t i = 0; i < n; ++i) common_batch_add(b, head_tok[done + i], (llama_pos) (done + i), { 0 }, false);
            if (llama_decode(ctx, b) != 0) { std::printf("prime decode failed\n"); return 2; }
            done += n;
        }
        llama_batch_free(b);
    }
    const llama_pos slot_len = (llama_pos) head_tok.size();
    CHECK(llama_memory_seq_pos_max(mem, 0) + 1 == slot_len, "slot length after priming");

    const char * modes[] = { "auto", "tree", "greedy" };
    for (const char * mode : modes) {
        for (int with_mass = 0; with_mass < 2; ++with_mass) {
            options on;  on.mode = mode; on.mass = with_mass != 0;
            options sl = on; sl.base_seq = 0; sl.base_len = slot_len;

            const auto A = eng.decide_batch(head, tails, cs.inputs, on);   // normal mode
            const auto B = eng.decide_batch("",   tails, cs.inputs, sl);   // slot mode
            const auto C = eng.decide_batch(head, tails, cs.inputs, on);   // normal again: cached prefix must still be valid
            const double d_ab = cmp(A, B, "normal-vs-slot");
            const double d_ac = cmp(A, C, "normal-vs-normal-after-slot");
            std::printf("mode=%-6s mass=%d  max|normal-slot|=%.2e  max|normal-again|=%.2e  slot_cache_hit=%d shared_tokens=%zu normal2_cache_hit=%d  scored_rows=%d\n",
                        mode, with_mass, d_ab, d_ac, (int) B.cache_hit, B.shared_tokens, (int) C.cache_hit, B.rows);
            CHECK(d_ab < 1e-4, "slot mode differs from normal mode by %.3e (mode %s mass %d)", d_ab, mode, with_mass);
            CHECK(d_ac < 1e-6, "normal mode changed after a slot call (%.3e)", d_ac);
            CHECK(B.cache_hit, "slot mode must report cache_hit");
            CHECK(B.shared_tokens == (size_t) slot_len, "slot mode shared_tokens %zu vs %d", B.shared_tokens, (int) slot_len);
            if (with_mass) {
                for (const auto & it : B.items) for (const auto & f : it.fields) {
                    CHECK(f.has_mass, "mass missing");
                    CHECK(f.log_mass <= 1e-4f, "log_mass must be <= 0, got %f", f.log_mass);
                }
            }
            CHECK(llama_memory_seq_pos_max(mem, 0) + 1 == slot_len, "slot KV length changed by a decision (%d)", (int) (llama_memory_seq_pos_max(mem, 0) + 1));
            for (int s = 2; s < n_slots + n_dec; ++s) {
                CHECK(llama_memory_seq_pos_max(mem, s) < 0, "branch sequence %d leaked", s);
            }
        }
    }
    // print one decision so the numbers are visible
    {
        options sl; sl.mass = true; sl.base_seq = 0; sl.base_len = slot_len;
        const auto B = eng.decide_batch("", { tails[0] }, cs.inputs, sl);
        const auto j = assemble(cs, B.items[0]);
        std::printf("sample decision (random weights, only the shape matters):\n%s\n", j.dump(1).substr(0, 900).c_str());
    }

    // the slot must still be a clean, usable sequence: continue it and compare with a fresh context
    {
        std::vector<llama_token> extra = common_tokenize(vocab, "next question please", false, true);
        llama_batch b = llama_batch_init(512, 0, 1);
        common_batch_clear(b);
        for (size_t i = 0; i < extra.size(); ++i) common_batch_add(b, extra[i], slot_len + (llama_pos) i, { 0 }, i + 1 == extra.size());
        CHECK(llama_decode(ctx, b) == 0, "continue slot decode");
        std::vector<float> a(llama_vocab_n_tokens(vocab));
        std::copy(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + a.size(), a.begin());
        llama_batch_free(b);

        // control 1: same procedure (prime 167 tokens, then 4 more) in a context that never ran a decision
        llama_context * c3 = llama_init_from_model(model, cp);
        {
            llama_batch bp = llama_batch_init(512, 0, 1);
            common_batch_clear(bp);
            for (size_t i = 0; i < head_tok.size(); ++i) common_batch_add(bp, head_tok[i], (llama_pos) i, { 0 }, false);
            CHECK(llama_decode(c3, bp) == 0, "control prime");
            common_batch_clear(bp);
            for (size_t i = 0; i < extra.size(); ++i) common_batch_add(bp, extra[i], slot_len + (llama_pos) i, { 0 }, i + 1 == extra.size());
            CHECK(llama_decode(c3, bp) == 0, "control continue");
            llama_batch_free(bp);
        }
        const float * l3 = llama_get_logits_ith(c3, -1);
        double d_ctrl = 0;
        for (size_t i = 0; i < a.size(); ++i) d_ctrl = std::max(d_ctrl, (double) std::fabs(a[i] - l3[i]));
        std::printf("slot continued after decisions vs same procedure without decisions: max logit diff %.2e\n", d_ctrl);
        CHECK(d_ctrl < 1e-4, "slot KV corrupted by decisions (logit diff vs control %.3e)", d_ctrl);

        // control 2 (informational): everything in one batch in a fresh context
        llama_context_params cp2 = llama_context_default_params();
        cp2.n_ctx = 4096; cp2.n_batch = 512; cp2.n_ubatch = 512; cp2.n_seq_max = 1;
        llama_context * c2 = llama_init_from_model(model, cp2);
        std::vector<llama_token> all = head_tok; all.insert(all.end(), extra.begin(), extra.end());
        llama_batch b2 = llama_batch_init(512, 0, 1);
        common_batch_clear(b2);
        for (size_t i = 0; i < all.size(); ++i) common_batch_add(b2, all[i], (llama_pos) i, { 0 }, i + 1 == all.size());
        CHECK(llama_decode(c2, b2) == 0, "fresh decode");
        const float * l2 = llama_get_logits_ith(c2, -1);
        double d_fresh = 0, d_fresh_ctrl = 0;
        for (size_t i = 0; i < a.size(); ++i) { d_fresh = std::max(d_fresh, (double) std::fabs(a[i] - l2[i])); d_fresh_ctrl = std::max(d_fresh_ctrl, (double) std::fabs(l3[i] - l2[i])); }
        std::printf("informational, one-batch fresh context: diff vs slot %.2e, diff vs control %.2e (same size means a batch-shape effect, not the decisions)\n", d_fresh, d_fresh_ctrl);
        llama_batch_free(b2);
        llama_free(c2);
        llama_free(c3);
    }

    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "FUNCTIONAL TEST PASSED" : "FUNCTIONAL TEST FAILED", g_fail);
    llama_free(ctx);
    llama_model_free(model);
    return g_fail == 0 ? 0 : 1;
}
