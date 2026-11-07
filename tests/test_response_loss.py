"""Compare optimized loss and gradients against the actual Qwen causal LM loss."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).parents[1] / 'training'))


class ResponseLossTests(unittest.TestCase):
    def test_matches_qwen_loss_and_gradients_with_masked_prompt_and_padding(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from response_loss import response_only_loss
        torch.manual_seed(2026)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).eval()
        ids = torch.randint(0, 64, (2, 16))
        labels = ids.clone()
        labels[:, :10] = -100
        labels[1, 13:] = -100
        inputs = {'input_ids': ids, 'labels': labels, 'attention_mask': torch.ones_like(ids)}
        reference = model(**inputs, use_cache=False).loss
        reference.backward()
        grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        loss, outputs = response_only_loss(model, inputs)
        torch.testing.assert_close(loss, reference)
        self.assertLess(outputs.logits.shape[1], ids.shape[1])
        loss.backward()
        for name, parameter in model.named_parameters():
            if name in grads:
                torch.testing.assert_close(parameter.grad, grads[name], rtol=1e-4, atol=1e-6)
        self.assertTrue(torch.equal(inputs['labels'], labels))

    def test_peft_trainer_one_optimizer_step_and_evaluation(self):
        import tempfile
        import math
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models
        from response_loss import ResponseOnlyTrainer
        from train_lora import CausalPaddingCollator
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8)
        config._attn_implementation = 'sdpa'
        model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(
            r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
            models.WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0")),
            pad_token="t0", unk_token="t0")
        ids = list(range(1, 17))
        records = [{'input_ids': ids, 'attention_mask': [1]*16,
                    'labels': [-100]*10 + ids[10:]} for _ in range(2)]
        with tempfile.TemporaryDirectory() as folder:
            args = TrainingArguments(output_dir=folder, use_cpu=True,
                max_steps=1, gradient_accumulation_steps=2,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                eval_strategy='steps', eval_steps=1, save_strategy='steps', save_steps=1,
                report_to='none', remove_unused_columns=False, disable_tqdm=True)
            trainer = ResponseOnlyTrainer(model=model, args=args, longest_first=True,
                train_dataset=records, eval_dataset=records,
                data_collator=CausalPaddingCollator(tokenizer))
            self.assertEqual(list(trainer._get_train_sampler()), [0, 1])
            trainer.train()
            metrics = trainer.evaluate()
            self.assertEqual(trainer.state.global_step, 1)
            self.assertTrue(math.isfinite(metrics['eval_loss']))
            sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
            from colab_helpers import latest_complete_checkpoint
            checkpoint = latest_complete_checkpoint(folder)
            self.assertEqual(checkpoint.name, 'checkpoint-1')
            args.max_steps = 2
            resumed = ResponseOnlyTrainer(model=model, args=args, longest_first=True,
                train_dataset=records, eval_dataset=records,
                data_collator=CausalPaddingCollator(tokenizer))
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 2)
            self.assertTrue(any(torch.count_nonzero(p).item() > 0
                for name, p in model.named_parameters() if 'lora_B' in name))
