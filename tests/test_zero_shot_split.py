import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "training")]
from prepare_zero_shot_split import split_zero_shot, strip_apps


def row(i, calls, hist=()):
    return {"id": f"r{i}", "audio_text": f"yêu cầu số {i}", "label": {"tool_calls": [{"name": n, "arguments": {}} for n in calls]},
            "history": [{"role": "assistant", "tool_calls": [{"name": n, "arguments": {}} for n in hist]}] if hist else [],
            "context": {"installed_apps": {"a": {}, "b": {}, "c": {}}}}


class ZeroShotSplitTests(unittest.TestCase):
    def test_partitions_by_app_without_leak(self):
        rows = [row(0, ["c.x"]), row(1, ["a.x"], hist=["c.y"]), row(2, ["a.x"]), row(3, ["b.x", "sys.ask_missing_info"])] + \
               [row(10 + i, ["a.x"]) for i in range(20)]
        train, dev_id, dev_zs, dropped = split_zero_shot(rows, {"c"}, 7)
        self.assertEqual([r["id"] for r in dev_zs], ["r0"])
        self.assertEqual(dropped, 1)
        self.assertTrue(all("c" not in r["context"]["installed_apps"] for r in train))
        self.assertTrue(all("c" in r["context"]["installed_apps"] for r in dev_zs))
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in dev_id})

    def test_strip_does_not_mutate_input(self):
        r = row(0, ["a.x"])
        strip_apps(r, {"a"})
        self.assertIn("a", r["context"]["installed_apps"])


if __name__ == "__main__":
    unittest.main()
