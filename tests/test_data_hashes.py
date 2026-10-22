"""数据指纹和续训配置校验；只使用临时文件，不启动训练。"""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qwen_math.common.checkpoint import (
    compute_data_hashes,
    file_sha256,
    save_run_config,
    validate_resume_config,
)


class DataHashTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory(prefix="qwen_hash_test_", dir="/tmp")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.train = self.root / "train.jsonl"
        self.val = self.root / "val.jsonl"
        self.train.write_text('{"question":"训练题"}\n', encoding="utf-8")
        self.val.write_text('{"question":"验证题"}\n', encoding="utf-8")
        self.config = {
            "train_path": str(self.train), "val_path": str(self.val),
            "batch_size": 1, "resume_from": None,
        }
        self.checkpoint = self.root / "checkpoint"

    def test_streamed_hash_matches_reference(self):
        self.train.write_bytes(b"abc" * 800000)
        expected = hashlib.sha256(self.train.read_bytes()).hexdigest()
        self.assertEqual(file_sha256(self.train), expected)

    def test_unchanged_data_and_resume_path_pass(self):
        original = dict(self.config)
        save_run_config(self.config, self.checkpoint)
        self.assertEqual(self.config, original)
        resumed = dict(self.config, resume_from=str(self.checkpoint))
        validate_resume_config(resumed, self.checkpoint)
        snapshot = json.loads((self.checkpoint / "run_config.json").read_text())
        self.assertEqual(snapshot["data_sha256"], compute_data_hashes(self.config))

    def test_either_file_changed_in_place_is_rejected(self):
        for field in ("train_path", "val_path"):
            with self.subTest(field=field):
                save_run_config(self.config, self.checkpoint)
                path = Path(self.config[field])
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, field):
                    validate_resume_config(self.config, self.checkpoint)
                path.write_bytes(original)

    def test_regular_config_change_is_rejected(self):
        save_run_config(self.config, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "续训配置"):
            validate_resume_config(dict(self.config, batch_size=2), self.checkpoint)

    def test_old_snapshot_without_hash_is_rejected(self):
        self.checkpoint.mkdir()
        (self.checkpoint / "run_config.json").write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "缺少完整数据哈希"):
            validate_resume_config(self.config, self.checkpoint)

    def test_periodic_snapshot_keeps_startup_fingerprints(self):
        initial = compute_data_hashes(self.config)
        self.train.write_bytes(self.train.read_bytes() + b"\n")
        save_run_config(self.config, self.checkpoint, data_hashes=initial)
        snapshot = json.loads((self.checkpoint / "run_config.json").read_text())
        self.assertEqual(snapshot["data_sha256"], initial)
        with self.assertRaisesRegex(ValueError, "train_path"):
            validate_resume_config(self.config, self.checkpoint)


if __name__ == "__main__":
    unittest.main()
