import contextlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parents[1] / 'training'))


class AttentionBackendTests(unittest.TestCase):
    def test_t4_policy_preserves_causal_mask_outputs_gradients_and_restores_helper(self):
        import torch
        from types import SimpleNamespace
        from transformers.integrations import sdpa_attention as hf
        from attention_backend import training_attention_context
        original = hf.use_gqa_in_sdpa
        module = SimpleNamespace(num_key_value_groups=4, is_causal=True)
        for masked in (False, True):
            torch.manual_seed(2026)
            tensors = [torch.randn(1, heads, 12, 8, requires_grad=True) for heads in (8, 2, 2)]
            mask = torch.ones(1, 1, 12, 12, dtype=torch.bool).tril() if masked else None
            reference, _ = hf.sdpa_attention_forward(module, *tensors, mask, scaling=.25)
            reference.square().sum().backward()
            gradients = [t.grad.clone() for t in tensors]
            for t in tensors:
                t.grad = None
            # CPU numerical test; only the CUDA kernel context is replaced with a no-op.
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.get_device_capability', return_value=(7, 5)), \
                 patch('torch.nn.attention.sdpa_kernel', return_value=contextlib.nullcontext()) as kernel:
                with training_attention_context():
                    self.assertFalse(hf.use_gqa_in_sdpa(None, tensors[1]))
                    actual, _ = hf.sdpa_attention_forward(module, *tensors, mask, scaling=.25)
                    actual.square().sum().backward()
                kernel.assert_called_once_with([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION])
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
            for t, grad in zip(tensors, gradients):
                torch.testing.assert_close(t.grad, grad, rtol=1e-4, atol=1e-5)
            self.assertIs(hf.use_gqa_in_sdpa, original)

    def test_policy_restores_helper_after_training_error(self):
        import torch
        from transformers.integrations import sdpa_attention as hf
        from attention_backend import training_attention_context
        original = hf.use_gqa_in_sdpa
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_capability', return_value=(7, 5)), \
             patch('torch.nn.attention.sdpa_kernel', return_value=contextlib.nullcontext()):
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                with training_attention_context():
                    raise RuntimeError('training failed')
        self.assertIs(hf.use_gqa_in_sdpa, original)

    def test_ampere_policy_is_unchanged(self):
        from attention_backend import training_attention_context
        from transformers.integrations import sdpa_attention as hf
        original = hf.use_gqa_in_sdpa
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_capability', return_value=(8, 0)), \
             patch('torch.nn.attention.sdpa_kernel') as kernel:
            with training_attention_context():
                self.assertIs(hf.use_gqa_in_sdpa, original)
            kernel.assert_not_called()
