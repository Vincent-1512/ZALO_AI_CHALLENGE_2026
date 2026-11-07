import copy
import importlib.util
import json
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('train_lora', Path(__file__).parents[1] / 'training/train_lora.py')
train = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = train
SPEC.loader.exec_module(train)


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = ''.join(f'<{m["role"]}>{m["content"]}</{m["role"]}>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return list(text.encode())


class TrainingEncodingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.record = {
            'sample_id': 'example',
            'messages': [
                {'role': 'system', 'content': 'system'},
                {'role': 'user', 'content': json.dumps({
                    'request': 'request', 'available_tools': [{'name': 'app.tool'}],
                    'steps_this_request': [], 'history': [{'text': 'h' * 100}],
                }, separators=(',', ':'))},
            ],
            'target': 't' * 250,
        }

    def test_target_budget_removes_history_and_preserves_labels(self):
        original = copy.deepcopy(self.record)
        no_history = copy.deepcopy(self.record)
        payload = json.loads(no_history['messages'][1]['content'])
        payload['history'] = []
        no_history['messages'][1]['content'] = json.dumps(payload, separators=(',', ':'))
        full = self.tokenizer.apply_chat_template(no_history['messages'] + [
            {'role': 'assistant', 'content': self.record['target']}])
        encoded = train.encode_record(self.record, self.tokenizer, len(full))
        self.assertEqual(encoded['input_ids'], full)
        prefix = self.tokenizer.apply_chat_template(no_history['messages'], add_generation_prompt=True)
        self.assertEqual(encoded['labels'][:len(prefix)], [-100] * len(prefix))
        self.assertEqual(encoded['labels'][len(prefix):], full[len(prefix):])
        self.assertEqual(original, self.record)

    def test_preflight_selects_context_and_keeps_all_records(self):
        from types import SimpleNamespace
        import tempfile
        import contextlib
        import io
        record = copy.deepcopy(self.record)
        record['target'] = 't' * 2500
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ('train', 'dev'):
                (root / f'{split}.jsonl').write_text(json.dumps(record) + '\n')
            args = SimpleNamespace(train_jsonl=root/'train.jsonl', dev_jsonl=root/'dev.jsonl',
                limit_train_samples=0, limit_dev_samples=0, max_sequence_length=8192,
                auto_context=True, longest_first=False, preflight_report=root/'report.json',
                output_dir=root)
            with contextlib.redirect_stdout(io.StringIO()):
                train_rows, dev_rows = train.load_encoded_data(args, self.tokenizer)
            self.assertEqual(args.max_sequence_length, 4096)
            self.assertEqual((len(train_rows), len(dev_rows)), (1, 1))
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report['splits']['train']['records'], 1)

    def test_oversize_schema_and_target_are_not_silently_cut(self):
        with self.assertRaisesRegex(ValueError, 'example'):
            train.encode_record(self.record, self.tokenizer, 50)

    def test_context_limit_rejects_invalid_sizes(self):
        for length in (0, -1, 16385):
            with self.subTest(length=length), self.assertRaises(ValueError):
                train.encode_record(self.record, self.tokenizer, length)


if __name__ == '__main__':
    unittest.main()
