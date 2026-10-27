"""使用随机小模型检查 PPO 入口、保存和确定性续训，无需下载。"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file


def import_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_save_and_resume(tmp_path):
    entry = import_script("train_ppo")
    fixture = import_script("check_ppo_iteration")
    model_id, adapter = fixture.save_tiny_fixture(tmp_path / "fixture")
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({
        "question": "What is 2+3?", "solution": "Final answer: 5", "answer": "5",
    }) + "\n", encoding="utf-8")
    config = dict(entry.CONFIG, model_id=model_id, sft_adapter_path=adapter,
                  train_path=str(data), device="cpu", dtype="float32",
                  max_prompt_tokens=32, max_new_tokens=8, max_iterations=2,
                  update_epochs=2, save_steps=1)

    uninterrupted = dict(config, output_dir=str(tmp_path / "continuous"))
    entry.run_training(uninterrupted)
    resumed = dict(config, output_dir=str(tmp_path / "resumed"), max_iterations=1)
    entry.run_training(resumed)
    checkpoint = Path(resumed["output_dir"]) / "checkpoints/iteration-000001"
    resumed["max_iterations"] = 2
    entry.run_training(resumed, resume_from=str(checkpoint))

    roots = [Path(c["output_dir"]) for c in (uninterrupted, resumed)]
    final = [root / "final-iteration-000002" for root in roots]
    for directory in final:
        for name in ("adapter_model.safetensors", "value_head.pt", "training_state.pt", "run_config.json", "tokenizer_config.json"):
            assert (directory / name).is_file()
    for name in ("adapter_model.safetensors", "value_head.pt"):
        states = [load_file(str(d / name)) if name.endswith("safetensors") else
                  torch.load(d / name, weights_only=True) for d in final]
        assert states[0].keys() == states[1].keys()
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    logs = [[json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()] for root in roots]
    assert [row["iteration"] for row in logs[1]] == [1, 2]
    for left, right in zip(*logs):
        assert {k: v for k, v in left.items() if k != "seconds"} == {
            k: v for k, v in right.items() if k != "seconds"
        }
    with pytest.raises(FileExistsError, match="输出目录非空"):
        entry.run_training(resumed)
    data.write_text(data.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="已改变"):
        entry.run_training(resumed, resume_from=str(checkpoint))


def test_eos_zero_is_valid():
    from qwen_math.rollout.sampler import trim_response

    trimmed, terminated = trim_response(torch.tensor([7, 0, 9]), eos_token_id=0)
    assert trimmed.tolist() == [7, 0]
    assert terminated is True
