# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    max_pixels: int = 4194304
    min_pixels: int = 262144
    filter_overlong_prompts: bool = True

    def post_init(self):
        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                self.format_prompt = None


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "grpo"
    disable_kl: bool = False
    use_kl_loss: bool = False
    kl_penalty: str = "kl"
    kl_coef: float = 1e-3
    kl_type: str = "fixed"
    kl_horizon: float = 0.0
    kl_target: float = 0.0
    keep_neg_ratio: float = 1.0  # GRPO: ratio of negative samples to keep (1.0 = keep all, keep worst samples first)
    keep_pos_ratio: float = 1.0  # GRPO: ratio of positive samples to keep (1.0 = keep all, keep best samples first)
    compute_new_adv: bool = False  # GRPO: recompute advantages using only kept samples after filtering


@dataclass
class TrainerConfig:
    total_episodes: int = 10
    max_steps: Optional[int] = None
    project_name: str = "easy_r1"
    experiment_name: str = "demo"
    logger: Tuple[str] = ("console", "wandb")
    nnodes: int = 1
    n_gpus_per_node: int = 8
    critic_warmup: int = 0
    val_freq: int = -1
    val_before_train: bool = True
    val_only: bool = False
    val_generations_to_log: int = 0
    train_generations_to_log: int = 0  # Number of training rollout samples to log
    train_log_freq: int = 10  # Log training generations every N steps
    save_freq: int = -1
    save_limit: int = -1
    save_checkpoint_path: Optional[str] = None
    load_checkpoint_path: Optional[str] = None

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)
        if self.load_checkpoint_path is not None:
            self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        
        # Validate keep_neg_ratio and keep_pos_ratio range [0, 1]
        if not (0.0 <= self.algorithm.keep_neg_ratio <= 1.0):
            raise ValueError(
                f"keep_neg_ratio must be in the range [0.0, 1.0]. "
                f"Got keep_neg_ratio={self.algorithm.keep_neg_ratio}. "
                f"0.0 means discard all negative samples, 1.0 means keep all negative samples."
            )
        
        if not (0.0 <= self.algorithm.keep_pos_ratio <= 1.0):
            raise ValueError(
                f"keep_pos_ratio must be in the range [0.0, 1.0]. "
                f"Got keep_pos_ratio={self.algorithm.keep_pos_ratio}. "
                f"0.0 means discard all positive samples, 1.0 means keep all positive samples."
            )
        
        # Validate keep_neg_ratio and keep_pos_ratio: only supported for grpo
        if self.algorithm.adv_estimator != "grpo" and (self.algorithm.keep_neg_ratio != 1.0 or self.algorithm.keep_pos_ratio != 1.0):
            raise ValueError(
                f"keep_neg_ratio and keep_pos_ratio are currently only supported for grpo algorithm. "
                f"Got adv_estimator='{self.algorithm.adv_estimator}' with keep_neg_ratio={self.algorithm.keep_neg_ratio}, keep_pos_ratio={self.algorithm.keep_pos_ratio}. "
                f"Please either set adv_estimator='grpo', or set both keep_neg_ratio=1.0 and keep_pos_ratio=1.0."
            )
        
        # Validate compute_new_adv: only supported for grpo
        if self.algorithm.adv_estimator != "grpo" and self.algorithm.compute_new_adv:
            raise ValueError(
                f"compute_new_adv is currently only supported for grpo algorithm. "
                f"Got adv_estimator='{self.algorithm.adv_estimator}' with compute_new_adv={self.algorithm.compute_new_adv}. "
                f"Please either set adv_estimator='grpo', or set compute_new_adv=False."
            )
        
        # Validate compute_new_adv: only useful when filtering is enabled
        if self.algorithm.compute_new_adv and self.algorithm.keep_neg_ratio == 1.0 and self.algorithm.keep_pos_ratio == 1.0:
            print(
                f"⚠️ Warning: compute_new_adv=True but keep_neg_ratio=1.0 and keep_pos_ratio=1.0. "
                f"compute_new_adv is only effective when sample filtering is enabled (keep_neg_ratio < 1.0 or keep_pos_ratio < 1.0). "
                f"The computed advantages will be the same as the original GRPO advantages."
            )

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
