# parallel-decision functional test

Checks, on CPU with a tiny random model, that deciding on top of a slot (`base_seq`) gives the same
probabilities as the rendered-prefix path, that the slot's KV is left bit-identical, and that no branch
sequence leaks. Attention-only model only: hybrid, sliding-window and MoE models are not covered.

1. From the repo root: `python tools/parallel-decision/tests/make_tiny.py tiny.gguf`
2. Build libllama (CPU is enough): `cmake -S . -B build-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF` then `cmake --build build-cpu --target llama`
3. Compile `func_test.cpp` + `tools/parallel-decision/decision-engine.cpp` + `common/json.cpp` against `include`, `ggml/include`, `common`, `vendor`, `tools/parallel-decision`, define `DJ_NO_COMMON`, link llama, ggml, ggml-cpu, ggml-base (on Linux add `-rdynamic -Wl,--unresolved-symbols=ignore-all -Wl,-z,lazy`, because the chat-template symbols are not linked).
4. `func_test tiny.gguf` ends with `FUNCTIONAL TEST PASSED`.

Without `DJ_NO_COMMON` the test links against libcommon instead of its stand-ins.
