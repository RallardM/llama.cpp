# Writes a tiny random-weight llama GGUF that reuses the repo's llama-spm tokenizer. Run from the repo root:
#   python tools/parallel-decision/tests/make_tiny.py tiny.gguf
import sys
sys.path.insert(0, 'gguf-py')
import numpy as np
import gguf

src = gguf.GGUFReader('models/ggml-vocab-llama-spm.gguf')
n_vocab, n_embd, n_head, n_kv, n_layer, n_ff = 32000, 64, 4, 2, 2, 128
hd = n_embd // n_head

w = gguf.GGUFWriter(sys.argv[1] if len(sys.argv) > 1 else 'tiny.gguf', 'llama')
w.add_name('tiny-random')
w.add_context_length(2048)
w.add_embedding_length(n_embd)
w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_kv)
w.add_layer_norm_rms_eps(1e-5)
w.add_rope_dimension_count(hd)
w.add_vocab_size(n_vocab)
w.add_file_type(gguf.LlamaFileType.ALL_F32)

def arr(key):
    return src.fields[key].contents()

w.add_tokenizer_model(str(src.fields['tokenizer.ggml.model'].contents()))
w.add_token_list(arr('tokenizer.ggml.tokens'))
w.add_token_scores(arr('tokenizer.ggml.scores'))
w.add_token_types(arr('tokenizer.ggml.token_type'))
w.add_bos_token_id(int(src.fields['tokenizer.ggml.bos_token_id'].contents()))
w.add_eos_token_id(int(src.fields['tokenizer.ggml.eos_token_id'].contents()))
w.add_unk_token_id(int(src.fields['tokenizer.ggml.unknown_token_id'].contents()))
w.add_add_bos_token(True)

rng = np.random.default_rng(1234)
def r(*shape, s=0.12):
    return (rng.standard_normal(shape) * s).astype(np.float32)
def ones(n):
    return np.ones(n, dtype=np.float32)

w.add_tensor('token_embd.weight', r(n_vocab, n_embd, s=0.5))
w.add_tensor('output_norm.weight', ones(n_embd))
w.add_tensor('output.weight', r(n_vocab, n_embd, s=0.5))
for i in range(n_layer):
    p = f'blk.{i}.'
    w.add_tensor(p + 'attn_norm.weight', ones(n_embd))
    w.add_tensor(p + 'attn_q.weight', r(n_head * hd, n_embd))
    w.add_tensor(p + 'attn_k.weight', r(n_kv * hd, n_embd))
    w.add_tensor(p + 'attn_v.weight', r(n_kv * hd, n_embd))
    w.add_tensor(p + 'attn_output.weight', r(n_embd, n_head * hd))
    w.add_tensor(p + 'ffn_norm.weight', ones(n_embd))
    w.add_tensor(p + 'ffn_gate.weight', r(n_ff, n_embd))
    w.add_tensor(p + 'ffn_up.weight', r(n_ff, n_embd))
    w.add_tensor(p + 'ffn_down.weight', r(n_embd, n_ff))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print('wrote', sys.argv[1] if len(sys.argv) > 1 else 'tiny.gguf')
