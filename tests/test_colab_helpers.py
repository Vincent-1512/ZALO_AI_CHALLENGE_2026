import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('colab_helpers', Path(__file__).parents[1] / 'scripts/colab_helpers.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class LoggingTests(unittest.TestCase):
    def test_child_traceback_is_persisted_and_surfaced(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            log = Path(folder) / 'run.log'
            with self.assertRaisesRegex(RuntimeError, 'deliberate failure'):
                helpers.run_logged([sys.executable, '-u', '-c',
                    'print("started", flush=True); raise ValueError("deliberate failure")'], folder, log)
            self.assertIn('started', log.read_text())
            self.assertIn('Traceback', log.read_text())

    def test_successful_child_returns_log_path(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            log = Path(folder) / 'run.log'
            self.assertEqual(helpers.run_logged([sys.executable, '-c', 'print("ok")'], folder, log), log)
            self.assertIn('ok', log.read_text())

    def test_resume_skips_incomplete_latest_checkpoint(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            complete = Path(folder)/'checkpoint-25'
            complete.mkdir()
            for name in ('optimizer.pt','scheduler.pt','rng_state.pth','scaler.pt',
                         'adapter_config.json','adapter_model.safetensors'):
                (complete/name).write_text('nonempty')
            (complete/'trainer_state.json').write_text(json.dumps({'global_step':25}))
            (Path(folder)/'checkpoint-50').mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(helpers.latest_complete_checkpoint(folder, require_scaler=True), complete)
            (complete/'scaler.pt').unlink()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
                helpers.latest_complete_checkpoint(folder, require_scaler=True)
