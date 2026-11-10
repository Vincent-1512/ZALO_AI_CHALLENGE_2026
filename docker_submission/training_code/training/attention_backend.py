"""Scoped workaround for native GQA selecting quadratic math SDPA on T4.

Transformers is pinned to 4.57.1. Its existing SDPA function repeats KV heads
when use_gqa_in_sdpa returns False; masks, scaling and dropout remain upstream.
"""
from contextlib import contextmanager
from types import SimpleNamespace


def _repeat_kv_instead_of_native_gqa(attention_mask, key):
    return False


@contextmanager
def training_attention_context():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        yield
        return
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers.integrations import sdpa_attention
    original = sdpa_attention.use_gqa_in_sdpa
    try:
        sdpa_attention.use_gqa_in_sdpa = _repeat_kv_instead_of_native_gqa
        # No math fallback: it can allocate a multi-GiB sequence^2 matrix on T4.
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            yield
    finally:
        sdpa_attention.use_gqa_in_sdpa = original


def check_t4_attention(sequence_length, config):
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        return
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    width = getattr(config, 'head_dim', None) or config.hidden_size // heads
    tensors = [torch.randn(1, n, sequence_length, width, device='cuda',
                           dtype=torch.float16, requires_grad=True) for n in (heads, kv_heads, kv_heads)]
    module = SimpleNamespace(num_key_value_groups=heads // kv_heads, is_causal=True)
    print(f'T4 attention probe: seq={sequence_length}, Q heads={heads}, KV heads={kv_heads}, '
          f'head_dim={width}; efficient-only SDPA, no native GQA/math fallback', flush=True)
    with training_attention_context():
        output, _ = sdpa_attention_forward(module, *tensors, attention_mask=None, scaling=width ** -.5)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
    if not all(t.grad is not None and torch.isfinite(t.grad).all() for t in tensors):
        raise RuntimeError('T4 attention probe produced invalid gradients')
    print('T4 efficient attention forward/backward probe passed (before model allocation).', flush=True)
