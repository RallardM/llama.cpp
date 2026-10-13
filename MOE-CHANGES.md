# MoE Cache Feature — Changes from `perf` to `moe-cache`

> **Branch:** `moe-cache` (forked from `ggml-org/llama.cpp`, based on `perf`)  
> **Base:** `perf` | **Commits ahead:** 30  
> **Scope:** MoE expert caching on CUDA — persistent LRU slot pools, async H2D transfer, per-tensor caches, and prefetching.

---

## 1. New Files Created

| File | Description |
|------|-------------|
| `ggml/src/ggml-cuda/moe-cache.cu` | MoE cache slot pool + LRU eviction + async H2D copy (656 lines) |
| `ggml/src/ggml-cuda/moe-cache.cuh` | Host-side MoE cache header, restricts to CUDA runtime API (115 lines) |
| `tests/test-moe-cache.cpp` | Unit tests for MoE cache (renamed from `.cu`) |

---

## 2. Modified Files

### 2.1 Core MoE / Buffer Type

| File | Changes |
|------|---------|
| `ggml/src/ggml.h` | Register `GGML_BUFFER_TYPE::CUDA_MoE_Cached` |
| `ggml/src/ggml-backend.h` | Backend infrastructure for MoE cache |
| `ggml/src/ggml-backend.cpp` | Simplified backend dispatch (−533 lines) |
| `ggml/src/ggml.c` | Core ggml simplification (−499 lines) |
| `include/llama.h` | Add `--moe-expert-cache-size` parameter plumb-through |

### 2.2 CUDA Backend (`ggml/src/ggml-cuda/`)

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Add moe-cache.cu to build, link cudart |
| `ggml-cuda.cu` | Dispatch hook for `mul_mat_id` (cached path), `supports_buft` accepts `CUDA_MoE_Cached` |
| `common.cuh` | Buffer type awareness, MoE FFN fusion disabled for cached experts |
| `topk-moe.cu` | Expert routing updates for MoE cache |
| `topk-moe.cuh` | Top-K MoE header adjustments |
| `mmf.cu` / `mmf.cuh` | Minor MoE cache integration (7 / 6 lines) |

### 2.3 Llama Core (`src/`)

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Add moe-cache test and sources |
| `llama.cpp` | Route expert tensors to `CUDA_MoE_Cached` when `--moe-expert-cache-size > 0` |
| `llama-model.cpp` | Pre-determine max expert size at model load time |
| `llama-context.cpp` | Eagerly preallocate slot pool at model load, route ops to CUDA for MoE cache |
| `llama-graph.cpp` | Graph-level MoE cache integration |
| `llama-arch.cpp` | Architecture-level MoE cache support |
| `llama-kv-cache.cpp` | KV cache integration with MoE caching |
| `llama-hparams.cpp` / `llama-hparams.h` | Hyperparams for MoE cache |
| `llama-cparams.h` | Config params for MoE cache |
| `llama-impl.h` | Internal implementation for MoE cache |
| `llama-batch.cpp` / `llama-batch.h` | Batch processing with MoE cache |
| `llama-model-loader.cpp` | Model loading with MoE cache awareness |
| `llama-model-saver.cpp` | Model saving with MoE cache |
| `llama-model.h` / `llama-context.h` / `llama-graph.h` / `llama-arch.h` | Header updates |
| `llama-mmap.cpp` / `llama-mmap.h` | Memory mapping for MoE cache |

### 2.4 Server (`tools/server/`)

| File | Changes |
|------|---------|
| `main.cpp` | Add `--moe-expert-cache-size` CLI flag |
| `server-models.cpp` / `server-models.h` | Model handling for MoE cache |
| `server-common.cpp` / `server-common.h` | Common server utilities for MoE cache |
| `server-task.cpp` / `server-task.h` | Task scheduling with MoE cache |
| `server.cpp` | Server integration |
| `server-chat.cpp` / `server-chat.h` | Chat with MoE cache |
| `server-http.cpp` / `server-http.h` | HTTP layer for MoE cache logging |
| `server-context.cpp` / `server-context.h` | Server context for MoE cache |
| `server-stream.cpp` / `server-stream.h` | Streaming with MoE cache |
| `server-tools.cpp` / `server-tools.h` | Tool integration |
| `server-queue.cpp` / `server-queue.h` | Queue management |

**Server logging:** Logs MoE cache hit rate per request.

### 2.5 Build System

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Top-level build updates for MoE cache |
| `cmake/build-info.cmake` | Build info updates |
| `Makefile` | Makefile updates |
| `src/CMakeLists.txt` | Source-level CMake for MoE cache |
| `tests/CMakeLists.txt` | Add `test-moe-cache` to test suite |

### 2.6 Tests

| File | Changes |
|------|---------|
| `tests/test-backend-ops.cpp` | Simplified backend ops tests (−2613 lines) |

### 2.7 Common / Utilities

| File | Changes |
|------|---------|
| `common/ngram-map.cpp` | Ngram map updates |
| `common/ngram-mod.cpp` | Minor ngram mod changes |

### 2.8 Examples

| File | Changes |
|------|---------|
| `examples/llama-bench/llama-bench.cpp` | Bench example with MoE cache |
| `examples/chat/13b-chat.sh` | Chat script |

---

## 3. Commit History (30 commits on `moe-cache` ahead of `perf`)

| # | Commit | Message |
|---|--------|---------|
| 30 | `01ffc2e77` | Implement dynamic Jev chat auto-routing feature |
| 29 | `342aff0e3` | Claude modifs |
| 28 | `8b5ca1215` | Local AI modifications |
| 27 | `9cf4f1a9f` | moe-cache: same-layer sibling prefetch (Prefetch B) |
| 26 | `5539e400a` | moe-cache: per-cache copy stream for H2D/compute pipelining (prefetch A) |
| 25 | `8b0f827ee` | moe-cache: stage when n_unique > n_slots to avoid mid-op eviction races |
| 24 | `05a5a51e8` | moe-cache: per-tensor eager preallocate, exact slot size per pool |
| 23 | `54feb4965` | moe-cache: per-tensor caches (slot count = N per layer per matrix) |
| 22 | `4b3dc3557` | moe-cache: revert to multi-cache (one pool per slot_size) for correctness |
| 21 | `af273fab2` | moe-cache: fix synthetic src0 nb[3] (slot_size * n_slots, not slot_size) |
| 20 | `1b8e22ded` | ggml-cuda: device supports_buft must accept CUDA_MoE_Cached |
| 19 | `7b9bd6f2d` | moe-cache: report is_host=false so scheduler routes ops to CUDA |
| 18 | `616b75ea9` | diag: log buffer type of first mul_mat_id op when cache is on |
| 17 | `e8ebf9b9e` | ggml-cuda: disable MoE FFN fusion when experts are in CUDA_MoE_Cached |
| 16 | `dcc2cb143` | server: log MoE cache hit rate per request |
| 15 | `e2ede2676` | moe-cache: eagerly preallocate slot pool at model load |
| 14 | `8612b4a27` | moe-cache: pre-determine max expert size at model load time |
| 13 | `8edc54c9f` | moe-cache: option A -- single per-device pool with grow-on-demand slot size |
| 12 | `96ddb23b1` | moe-cache: multi-cache registry keyed by (device, slot_size) |
| 11 | `cf582f8e5` | fix: build errors in iteration 2 |
| 10 | `b92c8d593` | moe-cache: iteration 2 -- persistent per-device LRU slot pool |
| 9 | `55030281b` | ggml-cuda: dispatch hook + cached path for mul_mat_id (iteration 1) |
| 8 | `c9b9b2dfc` | llama: route expert tensors to CUDA_MoE_Cached when --moe-expert-cache-size > 0 |
| 7 | `3d444e4ec` | ggml-cuda: register CUDA_MoE_Cached buffer type |
| 6 | `eff3b2a0e` | params: plumb --moe-expert-cache-size N through to llama_model_params |
| 5 | `42bb9bfe6` | moe-cache.cuh: drop common.cuh include, restrict to host CUDA runtime API |
| 4 | `c87c5e4e0` | tests: rename test-moe-cache .cu -> .cpp, link CUDA::cudart |
| 3 | `1ec174e33` | ggml-cuda: implement moe-cache slot pool + LRU + async H2D |
| 2 | `b411462b4` | ggml-cuda: stub moe-cache.cuh / moe-cache.cu |
| 1 | *(shared with perf)* | *(earlier commits)* |

---

## 4. Summary Statistics

| Metric | Value |
|--------|-------|
| Total files changed | 2641 (entire repo) |
| MoE-cache-specific files | **40** |
| Lines added (moe-cache files) | +4,659 |
| Lines removed (moe-cache files) | −11,330 |
| New files created | **3** |
| Files modified | **37** |
| Commits on branch | **30** |

---

## 5. Feature Architecture

```
--moe-expert-cache-size N
        │
        ▼
llama_model_params ──► llama_model (pre-determine max expert size)
                              │
                              ▼
                    llama_context (eagerly preallocate slot pool)
                              │
                              ▼
                    ggml-cuda dispatch (mul_mat_id cached path)
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
            slot pool   LRU cache   async H2D
            (per-device) (per-tensor) copy stream
                    │
                    ▼
            CUDA_MoE_Cached buffer type
                    │
                    ▼
            Server logs hit rate per request
```

---

## 6. Key Design Decisions

1. **CUDA_MoE_Cached buffer type** — New ggml buffer type to mark expert tensors that should be cached on the GPU.
2. **Per-device LRU slot pool** — One pool per device, with LRU eviction when `n_unique > n_slots`.
3. **Per-tensor caches** — Slot count = N per layer per matrix, with eager preallocation.
4. **Multi-cache registry** — Keyed by `(device, slot_size)` for correctness.
5. **Async H2D pipelining** — Separate copy streams for prefetch A (H2D/compute) and prefetch B (same-layer sibling).
6. **Multi-cache over single-cache** — Reverted to one pool per `slot_size` for correctness.
7. **MoE FFN fusion disabled** — When experts are in `CUDA_MoE_Cached`, fusion is disabled to avoid incorrect behavior.
