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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, quality_assessment_metrics


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def extract_answer_from_response(response: str) -> Optional[float]:
    """
    Extract numeric answer from model response.
    Looks for content between <answer> and </answer> tags.
    
    Args:
        response: Model generated response string
        
    Returns:
        Extracted float value or None if extraction fails
    """
    try:
        answer_start = response.find("<answer>")
        answer_end = response.find("</answer>", answer_start + len("<answer>"))
        
        if answer_end == -1:
            if answer_start == -1:
                # No tags found, try to parse the whole response
                answer_text = response
            else:
                # Only start tag found, take everything after it
                answer_text = response[answer_start + len("<answer>"):]
        else:
            # Both tags found, extract content between them
            answer_text = response[answer_start + len("<answer>"):answer_end]
        
        return float(answer_text.strip())
    except Exception:
        return None


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0, keep_neg_ratio: float = 1.0, keep_pos_ratio: float = 1.0, compute_new_adv: bool = False):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # GRPO: compute advantages with optional positive and negative sample filtering
        # Support keep_neg_ratio: keep a portion of negative samples based on advantage ranking (worst first)
        # Support keep_pos_ratio: keep a portion of positive samples based on advantage ranking (best first)
        # Support compute_new_adv: recompute advantages using only kept samples (after filtering)
        advantages, returns, advantages_original, winner_mask = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards, response_mask, index, keep_neg_ratio=keep_neg_ratio, keep_pos_ratio=keep_pos_ratio, compute_new_adv=compute_new_adv
        )
        # Store original advantages for metrics (especially when keep_neg_ratio < 1.0 or keep_pos_ratio < 1.0)
        data.batch["advantages_original"] = advantages_original
        # Store winner mask for filtering loss computation (if keep_neg_ratio < 1.0 or keep_pos_ratio < 1.0)
        data.batch["winner_mask"] = winner_mask
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
        val_reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_episodes

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")
        
        # Initialize best main_score for checkpoint tracking
        self.best_main_score = -float('inf')
        
        # Initialize train generation table for wandb (累积式记录)
        self.train_generation_table = None

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)
    
    def _maybe_log_train_generations(
        self, batch: DataProto, reward_scores: torch.Tensor
    ) -> None:
        """Log training rollout samples with multiple outputs per prompt (n > 1)"""
        if self.config.trainer.train_generations_to_log <= 0:
            return
        
        # Only log every train_log_freq steps
        if self.global_step % self.config.trainer.train_log_freq != 0:
            return
        
        rollout_n = self.config.worker.rollout.n
        
        # Decode inputs and outputs
        input_ids = batch.batch["prompts"]  # Shape: (bs * n, prompt_length)
        response_ids = batch.batch["responses"]  # Shape: (bs * n, response_length)
        ground_truth = batch.non_tensor_batch["ground_truth"]  # Shape: (bs * n,)
        scores = reward_scores.cpu().tolist()  # Shape: (bs * n,)
        
        # Decode text
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in response_ids]
        
        # Group by prompt (every n consecutive samples belong to the same prompt)
        num_prompts = len(input_texts) // rollout_n
        grouped_samples = []
        
        for prompt_idx in range(num_prompts):
            start_idx = prompt_idx * rollout_n
            end_idx = start_idx + rollout_n
            
            # Get data for this prompt's rollouts
            prompt_input = input_texts[start_idx]  # Same for all n rollouts
            prompt_label = str(ground_truth[start_idx])  # Same for all n rollouts
            rollout_outputs = output_texts[start_idx:end_idx]
            rollout_scores = scores[start_idx:end_idx]
            
            # Create a row with: (input, label, output_1, score_1, output_2, score_2, ...)
            row = [prompt_input, prompt_label]
            for i, (output, score) in enumerate(zip(rollout_outputs, rollout_scores), 1):
                row.extend([output, score])
            
            grouped_samples.append(tuple(row))
        
        # Sort and sample
        grouped_samples.sort(key=lambda x: x[0])  # Sort by input text
        rng = np.random.RandomState(42 + self.global_step)  # Different seed each time
        rng.shuffle(grouped_samples)
        grouped_samples = grouped_samples[: self.config.trainer.train_generations_to_log]
        
        # Log to console/wandb with custom format
        self._log_train_generations_to_backends(grouped_samples, rollout_n, self.global_step)
    
    def _log_train_generations_to_backends(
        self, samples: List[tuple], rollout_n: int, step: int
    ) -> None:
        """Log training generations to console/wandb/swanlab"""
        # Log to console
        if "console" in self.config.trainer.logger:
            print(f"\n{'='*80}")
            print(f"[Train Rollout Generations at Step {step}]")
            print(f"{'='*80}")
            for i, sample in enumerate(samples, 1):
                prompt_input = sample[0]
                prompt_label = sample[1]
                print(f"\n--- Sample {i} ---")
                print(f"[Input] {prompt_input[:200]}..." if len(prompt_input) > 200 else f"[Input] {prompt_input}")
                print(f"[Label] {prompt_label}")
                for rollout_idx in range(rollout_n):
                    output = sample[2 + rollout_idx * 2]
                    score = sample[3 + rollout_idx * 2]
                    print(f"[Rollout {rollout_idx+1}] Score: {score:.4f}")
                    print(f"  Output: {output[:200]}..." if len(output) > 200 else f"  Output: {output}")
            print(f"{'='*80}\n")
        
        # Log to wandb (使用累积式 table，避免覆盖问题)
        if "wandb" in self.config.trainer.logger:
            try:
                import wandb
                # Create table with dynamic columns
                columns = ["step", "input", "label"]
                for i in range(1, rollout_n + 1):
                    columns.extend([f"output_rollout_{i}", f"score_rollout_{i}"])
                
                # Initialize table on first call
                if self.train_generation_table is None:
                    self.train_generation_table = wandb.Table(columns=columns)
                
                # Create a new table with same columns and existing data (workaround for wandb issue)
                # See: https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                new_table = wandb.Table(columns=columns, data=self.train_generation_table.data)
                
                # Add new rows
                for sample in samples:
                    row = [step, sample[0], sample[1]]
                    for rollout_idx in range(rollout_n):
                        row.append(sample[2 + rollout_idx * 2])  # output
                        row.append(sample[3 + rollout_idx * 2])  # score
                    new_table.add_data(*row)
                
                wandb.log({"train/rollout_generations": new_table}, step=step)
                self.train_generation_table = new_table
            except Exception as e:
                print(f"Warning: Failed to log train generations to wandb: {e}")
        
        # Log to swanlab
        if "swanlab" in self.config.trainer.logger:
            try:
                import swanlab
                text_list = []
                for i, sample in enumerate(samples, 1):
                    parts = [f"Input: {sample[0]}", f"Label: {sample[1]}"]
                    for rollout_idx in range(rollout_n):
                        output = sample[2 + rollout_idx * 2]
                        score = sample[3 + rollout_idx * 2]
                        parts.append(f"Rollout {rollout_idx+1} (score={score:.4f}): {output}")
                    row_text = "\n\n".join(parts)
                    text_list.append(swanlab.Text(row_text, caption=f"train_sample_{i}"))
                swanlab.log({"train/rollout_generations": text_list}, step=step)
            except Exception as e:
                print(f"Warning: Failed to log train generations to swanlab: {e}")

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        # Lists to collect all predictions and ground truth for PLCC/SRCC calculation
        y_out, y_label = [], []
        error_count = 0
        
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            
            # Get ground truth labels
            ground_truth_batch = test_batch.non_tensor_batch["ground_truth"].tolist()
            sample_labels.extend(ground_truth_batch)
            
            # Extract predictions from output texts for PLCC/SRCC calculation
            # Note: extract_answer_from_response doesn't require <think> tag, just <answer> tag
            for output_text, gt in zip(output_texts, ground_truth_batch):
                pred = extract_answer_from_response(output_text)
                if pred is not None:
                    y_out.append(pred)
                    y_label.append(float(gt))
                else:
                    error_count += 1
            
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        
        # Calculate AGIQA-3K metrics (PLCC, SRCC) if we have valid predictions
        agiqa_metrics = {
            "val/extraction_answer_error": error_count,
        }
        if len(y_out) > 0 and len(y_label) > 0:
            print(f"Computing AGIQA-3K metrics on {len(y_out)} valid samples (errors: {error_count})")
            
            # Compute quality assessment metrics
            plcc, srcc, main_score, invalid_stats = quality_assessment_metrics(y_out, y_label)
            agiqa_metrics.update({
                "val/plcc": plcc,
                "val/srcc": srcc,
                "val/main_score": main_score,
                "val/quality_total_samples": invalid_stats['total_samples'],
                "val/quality_valid_samples": invalid_stats['valid_samples'],
                "val/quality_invalid_samples": invalid_stats['invalid_samples'],
                "val/quality_nan_samples": invalid_stats['nan_samples'],
                "val/quality_inf_samples": invalid_stats['inf_samples'],
            })
            
            print(f"AGIQA-3K Metrics: PLCC={plcc:.4f}, SRCC={srcc:.4f}, MainScore={main_score:.4f}")
            print(f"Quality Stats: Valid={invalid_stats['valid_samples']}/{invalid_stats['total_samples']}, "
                  f"NaN={invalid_stats['nan_samples']}, Inf={invalid_stats['inf_samples']}")
        else:
            print(f"Warning: No valid predictions extracted for AGIQA-3K metrics (errors: {error_count})")
        
        return {"val/reward_score": reward_score, **val_reward_metrics, **agiqa_metrics}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _save_best_checkpoint(self, main_score: float) -> None:
        """
        Save best checkpoint based on main_score.
        The best checkpoint will be saved to {save_checkpoint_path}/best_ckpt/
        and will be overwritten when a better score is achieved.
        
        Args:
            main_score: The main evaluation score to track
        """
        if main_score <= self.best_main_score:
            return
        
        # Update best score
        old_best = self.best_main_score
        self.best_main_score = main_score
        print(f"🎉 New best main_score: {main_score:.4f} (previous: {old_best:.4f}) at step {self.global_step}")
        
        # Save to best_ckpt directory
        best_folder_path = os.path.join(self.config.trainer.save_checkpoint_path, "best_ckpt")
        actor_path = os.path.join(best_folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)
        
        if self.use_critic:
            critic_path = os.path.join(best_folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)
        
        dataloader_path = os.path.join(best_folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)
        
        # Save metadata about the best checkpoint
        metadata_path = os.path.join(best_folder_path, "best_metadata.json")
        metadata = {
            "global_step": self.global_step,
            "main_score": main_score,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Best checkpoint saved to {best_folder_path}")

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        for _ in tqdm(range(self.config.trainer.total_episodes), desc="Episode", position=0):
            for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=1):
                self.global_step += 1
                if self.global_step > self.training_steps:
                    break

                metrics, timing_raw = {}, {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if "multi_modal_data" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):  # wg: worker group
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == "remax":
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["temperature"] = 0
                            gen_baseline_batch.meta_info["n"] = 1
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor, _ = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    batch.non_tensor_batch.pop("multi_modal_data", None)

                    # compute reward
                    with _timer("reward", timing_raw):
                        if self.use_reward_model:
                            raise NotImplementedError("Reward model is not supported yet.")

                        # we combine with rule-based rm
                        reward_tensor, reward_metrics = self.reward_fn(batch)
                        
                        # Check for NaN in reward_tensor
                        if torch.isnan(reward_tensor).any():
                            nan_count = torch.isnan(reward_tensor).sum().item()
                            total_elements = reward_tensor.numel()
                            print(f"⚠️ Warning: Found {nan_count}/{total_elements} NaN values in reward_tensor at step {self.global_step}")
                            print(f"   Replacing NaN with 0.0 to prevent training crash")
                            reward_tensor = torch.nan_to_num(reward_tensor, nan=0.0)
                        
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {
                            f"reward/{key}": value for key, value in reduce_metrics(reward_metrics).items()
                        }
                        metrics.update(reward_metrics)
                    
                    # Log training rollout generations (before balance to preserve order)
                    self._maybe_log_train_generations(batch, reward_tensor.sum(dim=-1))

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            keep_neg_ratio=self.config.algorithm.keep_neg_ratio,
                            keep_pos_ratio=self.config.algorithm.keep_pos_ratio,
                            compute_new_adv=self.config.algorithm.compute_new_adv,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_step:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        with _timer("validation", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)
                        
                        # Save best checkpoint if main_score improved
                        if "val/main_score" in val_metrics:
                            self._save_best_checkpoint(val_metrics["val/main_score"])

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                num_gpus = self.resource_pool_manager.get_num_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

                self.logger.log(data=metrics, step=self.global_step)

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)
                
                # Save best checkpoint if main_score improved
                if "val/main_score" in val_metrics:
                    self._save_best_checkpoint(val_metrics["val/main_score"])

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")
            print(f"Best main_score achieved: {self.best_main_score:.4f}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
