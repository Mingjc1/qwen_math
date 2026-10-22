"""不依赖训练库的配置加载与校验；配置是起步方案，不是已验证的实验配方。"""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
ALGORITHMS = {"sft", "dpo", "ppo", "grpo"}


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str
    model_id: str
    model_revision: str | None
    seed: int
    dtype: str
    train_data: str
    output_dir: str
    sft_adapter_path: str | None
    max_prompt_tokens: int
    max_response_tokens: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_updates: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    gradient_checkpointing: bool
    reference_policy: str
    cache_reference_logps: bool
    reward: str
    group_size: int
    generation_micro_batch_size: int
    rollout_prompts: int
    temperature: float
    top_p: float
    top_k: int
    dpo_beta: float
    kl_coef: float
    clip_epsilon: float
    gamma: float
    gae_lambda: float
    update_epochs: int
    value_head: str

    def validate(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm 必须属于 {sorted(ALGORITHMS)}")
        if self.model_id != MODEL_ID:
            raise ValueError(f"当前主实验模型固定为 {MODEL_ID}")
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str) or not self.model_revision.strip()
        ):
            raise ValueError("model_revision 必须是非空版本字符串或 null")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("不支持的 dtype")
        for name in (
            "max_prompt_tokens", "max_response_tokens", "micro_batch_size",
            "gradient_accumulation_steps", "max_updates", "lora_r", "lora_alpha",
            "group_size", "generation_micro_batch_size", "rollout_prompts", "update_epochs",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed 必须是非负整数")
        if not self.target_modules or any(
            not isinstance(name, str) or not name.strip() for name in self.target_modules
        ):
            raise ValueError("target_modules 必须是非空层名列表")
        for name in ("train_data", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} 必须是非空路径")
        for name in ("gradient_checkpointing", "cache_reference_logps"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} 必须是布尔值")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout 必须在 [0,1) 范围")
        if not self.learning_rate > 0 or not self.dpo_beta > 0 or not self.kl_coef >= 0:
            raise ValueError("learning_rate/dpo_beta 必须为正，kl_coef 必须非负")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon 必须在 (0,1) 范围")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma/gae_lambda 必须在 [0,1] 范围")
        if self.temperature != 1.0 or self.top_p != 1.0 or self.top_k != 0:
            raise ValueError("第一版固定原始策略采样：temperature=1, top_p=1, top_k=0")
        if self.algorithm == "sft":
            if self.sft_adapter_path is not None or self.reference_policy != "none":
                raise ValueError("SFT 从基座开始，不加载 SFT adapter 或 reference")
        else:
            if not isinstance(self.sft_adapter_path, str) or not self.sft_adapter_path.strip():
                raise ValueError("后训练分支必须指定同一份 SFT adapter 快照")
            if self.reference_policy != "frozen_sft":
                raise ValueError("后训练参考策略必须是 frozen_sft")
        if self.algorithm in {"ppo", "grpo"} and self.reward != "answer_correctness":
            raise ValueError("第一版在线 RL 使用 answer_correctness 规则奖励")
        if self.algorithm == "grpo" and self.group_size < 2:
            raise ValueError("GRPO group_size 必须 >= 2")
        expected_head = "shared_backbone" if self.algorithm == "ppo" else "none"
        if self.value_head != expected_head:
            raise ValueError(f"{self.algorithm} 的 value_head 应为 {expected_head}")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> TrainConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("配置顶层必须是 JSON 对象")
    expected = {field.name for field in fields(TrainConfig)}
    unknown = raw.keys() - expected
    missing = expected - raw.keys()
    if unknown or missing:
        raise ValueError(f"配置字段不匹配：未知={sorted(unknown)}，缺少={sorted(missing)}")
    config = TrainConfig(**raw)
    config.validate()
    return config
