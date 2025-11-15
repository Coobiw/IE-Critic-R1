# Copyright 2022 The HuggingFace Team
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int) -> None:
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        pass


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    index: torch.Tensor, 
    eps: float = 1e-6,
    keep_neg_ratio: float = 1.0,
    keep_pos_ratio: float = 1.0,
    compute_new_adv: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    
    Strategy:
    - Compute standard GRPO advantages (normalized scores)
    - If keep_pos_ratio < 1.0:
      Keep a portion of positive samples (advantage > 0) based on keep_pos_ratio (keep best samples first - highest advantages)
    - If keep_neg_ratio < 1.0:
      Keep a portion of negative samples (advantage < 0) based on keep_neg_ratio (keep worst samples first - lowest advantages)
    - If compute_new_adv = True:
      Recompute advantages using only kept samples (recalculate mean and std from kept samples)

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            Group index for each sample
        eps: `(float)`
            Small value for numerical stability
        keep_neg_ratio: `(float)`
            Ratio of negative samples to keep (0.0 to 1.0)
            1.0 = keep all, 0.5 = keep worst 50% of negative samples (lowest advantages)
        keep_pos_ratio: `(float)`
            Ratio of positive samples to keep (0.0 to 1.0)
            1.0 = keep all, 0.5 = keep best 50% of positive samples (highest advantages)
        compute_new_adv: `(bool)`
            If True, recompute advantages using only kept samples (after filtering)
            If False, use original advantages for kept samples

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length) - GRPO advantages (filtered samples have 0 if keep_neg_ratio or keep_pos_ratio < 1.0)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages_original: `(torch.Tensor)`
            shape: (bs, response_length) - original GRPO advantages for metrics (before filtering)
        winner_mask: `(torch.Tensor)`
            shape: (bs, response_length) - mask indicating kept samples (1=kept, 0=filtered)

    """
    # Validate keep_neg_ratio and keep_pos_ratio are in [0, 1]
    assert 0.0 <= keep_neg_ratio <= 1.0, f"keep_neg_ratio must be in [0, 1], got {keep_neg_ratio}"
    assert 0.0 <= keep_pos_ratio <= 1.0, f"keep_pos_ratio must be in [0, 1], got {keep_pos_ratio}"
    
    scores = token_level_rewards.sum(dim=-1)
    
    # Check for NaN in input rewards
    if torch.isnan(scores).any():
        nan_count = torch.isnan(scores).sum().item()
        print(f"⚠️ Warning: Found {nan_count} NaN values in GRPO scores (input rewards)")
        print(f"   NaN indices: {torch.where(torch.isnan(scores))[0].tolist()}")
        scores = torch.nan_to_num(scores, nan=0.0)
    
    id2score = defaultdict(list)
    id2idx_list = defaultdict(list)  # Track indices for each group
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
        id2idx_list[index[i]].append(i)

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    # Compute GRPO advantages (normalized scores)
    original_scores = scores.clone()
    for i in range(bsz):
        idx = index[i]
        # If std is 0 (all samples in group have same score), set advantage to 0
        if id2std[idx] < eps:
            original_scores[i] = 0.0
        else:
            original_scores[i] = (scores[i] - id2mean[idx]) / id2std[idx]

    advantages_original = original_scores.unsqueeze(-1) * response_mask
    
    # If both keep_neg_ratio == 1.0 and keep_pos_ratio == 1.0, keep all samples (standard GRPO)
    if keep_neg_ratio >= 1.0 and keep_pos_ratio >= 1.0:
        returns = original_scores.unsqueeze(-1) * response_mask
        winner_mask = torch.ones_like(original_scores).unsqueeze(-1) * response_mask
        return returns, returns, advantages_original, winner_mask
    
    # Filter positive and negative samples based on keep_pos_ratio and keep_neg_ratio
    filtered_advantages = torch.zeros_like(original_scores)
    sample_mask = torch.zeros_like(original_scores)  # 1 for kept samples, 0 for filtered
    
    for idx in id2score:
        group_indices = id2idx_list[idx]
        group_advantages = torch.tensor([original_scores[i] for i in group_indices])
        
        # Separate positive and negative samples
        positive_samples = []  # (local_idx, global_idx, advantage)
        negative_samples = []  # (local_idx, global_idx, advantage)
        
        for local_idx, global_idx in enumerate(group_indices):
            adv = group_advantages[local_idx]
            if adv > 0:
                positive_samples.append((local_idx, global_idx, adv))
            elif adv < 0:
                negative_samples.append((local_idx, global_idx, adv))
        
        # Keep zero-advantage samples (neutral samples)
        for local_idx, global_idx in enumerate(group_indices):
            if group_advantages[local_idx] == 0.0:
                filtered_advantages[global_idx] = 0.0
                sample_mask[global_idx] = 1.0
        
        # Keep a portion of positive samples based on keep_pos_ratio
        if len(positive_samples) > 0 and keep_pos_ratio > 0.0:
            # Sort positive samples by advantage (descending order - best first, highest advantages)
            positive_samples.sort(key=lambda x: x[2], reverse=True)
            
            # Calculate how many positive samples to keep
            num_pos_to_keep = max(1, int(len(positive_samples) * keep_pos_ratio)) if keep_pos_ratio < 1.0 else len(positive_samples)
            
            # Keep the best positive samples
            for i in range(num_pos_to_keep):
                _, global_idx, _ = positive_samples[i]
                filtered_advantages[global_idx] = original_scores[global_idx]
                sample_mask[global_idx] = 1.0
        
        # Keep a portion of negative samples based on keep_neg_ratio
        if len(negative_samples) > 0 and keep_neg_ratio > 0.0:
            # Sort negative samples by advantage (ascending order - worst first, lowest advantages)
            negative_samples.sort(key=lambda x: x[2])
            
            # Calculate how many negative samples to keep
            num_neg_to_keep = max(1, int(len(negative_samples) * keep_neg_ratio)) if keep_neg_ratio < 1.0 else len(negative_samples)
            
            # Keep the worst negative samples
            for i in range(num_neg_to_keep):
                _, global_idx, _ = negative_samples[i]
                filtered_advantages[global_idx] = original_scores[global_idx]
                sample_mask[global_idx] = 1.0

    returns = filtered_advantages.unsqueeze(-1) * response_mask
    winner_mask = sample_mask.unsqueeze(-1) * response_mask  # Broadcast to token level
    
    # If compute_new_adv=True, recompute advantages using only kept samples
    if compute_new_adv and (keep_neg_ratio < 1.0 or keep_pos_ratio < 1.0):
        # Recompute advantages for kept samples using their own mean and std
        new_advantages = torch.zeros_like(original_scores)
        
        for idx in id2score:
            group_indices = id2idx_list[idx]
            
            # Find kept samples in this group
            kept_indices = [i for i in group_indices if sample_mask[i] == 1.0]
            
            if len(kept_indices) < 2:
                # Not enough samples to compute std, use original advantages
                # This should rarely happen if keep_pos_ratio and keep_neg_ratio are reasonable
                for i in kept_indices:
                    new_advantages[i] = original_scores[i]
            else:
                # Recompute mean and std using only kept samples
                kept_scores = torch.tensor([scores[i] for i in kept_indices])
                kept_mean = torch.mean(kept_scores)
                kept_std = torch.std(kept_scores)
                
                # Renormalize kept samples
                for i in kept_indices:
                    if kept_std < eps:
                        # If std is 0 (all kept samples have same score), set advantage to 0
                        new_advantages[i] = 0.0
                    else:
                        new_advantages[i] = (scores[i] - kept_mean) / kept_std
        
        # Use new advantages for training
        final_advantages = new_advantages.unsqueeze(-1) * response_mask
        returns = new_advantages.unsqueeze(-1) * response_mask
        
        return final_advantages, returns, advantages_original, winner_mask
    
    return returns, returns, advantages_original, winner_mask


@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = VF.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the policy loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy

    """
    negative_approx_kl = log_probs - old_log_probs
    # clamp the ratio before exp to avoid nan
    # see: https://github.com/pytorch/pytorch/issues/10729
    ratio = torch.exp(negative_approx_kl)
    clipped_ratio = torch.exp(
        torch.clamp(negative_approx_kl, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    pg_loss = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss3 = -advantages * clip_ratio_dual

    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
    pg_clipfrac_higher = (pg_loss < pg_loss2).float()
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
    pg_clipfrac_lower = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    pg_clipfrac_higher = VF.masked_mean(pg_clipfrac_higher, response_mask)
    pg_clipfrac_lower = VF.masked_mean(pg_clipfrac_lower, response_mask)
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)
    
    final_pg_losses = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    final_pg_loss = agg_loss(loss_mat=final_pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    return final_pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    cliprange_value: float,
) -> Tuple[torch.Tensor, float]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        action_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    vf_loss = 0.5 * VF.masked_mean(torch.max(vf_loss1, vf_loss2), action_mask)  # clip if vf_loss1 < vf_loss2
    vf_clipfrac = VF.masked_mean((vf_loss1 < vf_loss2).float(), action_mask)
    return vf_loss, vf_clipfrac


def compute_kl(log_probs: torch.FloatTensor, ref_log_probs: torch.FloatTensor, kl_penalty: str) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        kl = ref_log_probs - log_probs
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
