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

from typing import Any, Dict, List

import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit

from ..protocol import DataProto

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    """
    Fit a logistic function to the data, handling NaN and inf values.
    Returns the fitted output and a dictionary of statistics.
    """
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def performance_fit(y_label, y_output, func_fit=True):
    """
    Calculate PLCC, SRCC, and main score for quality assessment.
    Filters out NaN and inf values before computation.
    
    Returns:
        tuple: (PLCC, SRCC, main_score, invalid_count_dict)
    """
    # Convert to numpy arrays if needed
    y_label = np.asarray(y_label, dtype=np.float64)
    y_output = np.asarray(y_output, dtype=np.float64)
    
    # Find valid (finite) values
    valid_mask = np.isfinite(y_label) & np.isfinite(y_output)
    num_invalid = np.sum(~valid_mask)
    num_total = len(y_label)
    
    # Count NaN and inf separately for detailed logging
    num_nan = np.sum(np.isnan(y_label) | np.isnan(y_output))
    num_inf = np.sum(np.isinf(y_label) | np.isinf(y_output))
    
    invalid_stats = {
        'total_samples': num_total,
        'invalid_samples': int(num_invalid),
        'nan_samples': int(num_nan),
        'inf_samples': int(num_inf),
        'valid_samples': int(np.sum(valid_mask))
    }
    
    # Filter to valid values only
    y_label_valid = y_label[valid_mask]
    y_output_valid = y_output[valid_mask]
    
    # Check if we have enough valid samples for curve fitting
    if len(y_label_valid) < 4:
        # Not enough samples for curve fitting (need at least 4 for 4 parameters)
        print(f"Warning: Only {len(y_label_valid)} valid samples out of {num_total} for quality assessment. "
              f"Skipping curve fitting. NaN: {num_nan}, Inf: {num_inf}")
        return 0.0, 0.0, 0.0, invalid_stats
    
    try:
        if func_fit:
            y_output_logistic = fit_function(y_label_valid, y_output_valid)
        else:
            y_output_logistic = y_output_valid
            
        PLCC = stats.pearsonr(y_output_logistic, y_label_valid)[0]
        SRCC = stats.spearmanr(y_output_valid, y_label_valid)[0]
        
        # Log warning if we filtered out samples
        if num_invalid > 0:
            print(f"Quality assessment: Filtered {num_invalid}/{num_total} invalid samples "
                  f"(NaN: {num_nan}, Inf: {num_inf})")
        
        return PLCC, SRCC, (PLCC+SRCC) / 2, invalid_stats
    except Exception as e:
        print(f"Warning: Failed to compute quality assessment metrics: {e}")
        print(f"Valid samples: {len(y_label_valid)}/{num_total}, NaN: {num_nan}, Inf: {num_inf}")
        return 0.0, 0.0, 0.0, invalid_stats


def quality_assessment_metrics(y_out: List[str], y_label: List[str]):
    """
    Compute quality assessment metrics (PLCC, SRCC, main_score).
    
    Returns:
        tuple: (plcc, srcc, main_score, invalid_stats_dict)
    """
    plcc, srcc, main_score, invalid_stats = performance_fit(y_label, y_out, func_fit=True)
    return plcc, srcc, main_score, invalid_stats
    
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def compute_non_cutoff_response_length_metrics(
    response_length: torch.Tensor, max_response_length: int
) -> Dict[str, Any]:
    """
    Compute response length metrics excluding samples that hit the cutoff (max_response_length).
    This is useful for tracking the evolution of response length without being skewed by cutoff samples.
    
    Args:
        response_length: Tensor of response lengths, shape (batch_size,)
        max_response_length: Maximum allowed response length
    
    Returns:
        Dictionary with metrics for non-cutoff samples
    """
    # Find samples that are NOT cutoff (response_length < max_response_length)
    non_cutoff_mask = response_length < max_response_length
    non_cutoff_lengths = response_length[non_cutoff_mask]
    
    # If no non-cutoff samples, return zeros
    if non_cutoff_lengths.numel() == 0:
        return {
            "response_length_no_cutoff/mean": 0.0,
            "response_length_no_cutoff/max": 0.0,
            "response_length_no_cutoff/min": 0.0,
            "response_length_no_cutoff/count": 0,
        }
    
    return {
        "response_length_no_cutoff/mean": torch.mean(non_cutoff_lengths.float()).detach().item(),
        "response_length_no_cutoff/max": torch.max(non_cutoff_lengths).detach().item(),
        "response_length_no_cutoff/min": torch.min(non_cutoff_lengths).detach().item(),
        "response_length_no_cutoff/count": non_cutoff_lengths.numel(),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> Dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]
    
    # For GRPO (with keep_neg_ratio < 1.0 or keep_pos_ratio < 1.0), use original advantages for metrics logging
    # The actual training uses filtered advantages stored in batch["advantages"]
    advantages_for_metrics = batch.batch.get("advantages_original", advantages)

    max_response_length = batch.batch["responses"].size(-1)

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    valid_adv = torch.masked_select(advantages_for_metrics, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)
    
    # Compute sample-level advantage (average over response tokens) for classification
    # Shape: (batch_size,)
    sample_advantages = (advantages_for_metrics * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
    
    # Classify samples as positive (adv > 0) or negative (adv < 0)
    positive_mask = sample_advantages > 0
    negative_mask = sample_advantages < 0
    
    num_positive = positive_mask.sum().item()
    num_negative = negative_mask.sum().item()
    
    # Compute response length statistics for positive and negative samples
    pos_neg_metrics = {}
    
    if num_positive > 0:
        positive_response_length = response_length[positive_mask]
        pos_neg_metrics.update({
            "sample_distribution/num_positive": num_positive,
            "response_length_positive/mean": torch.mean(positive_response_length).detach().item(),
            "response_length_positive/max": torch.max(positive_response_length).detach().item(),
            "response_length_positive/min": torch.min(positive_response_length).detach().item(),
        })
    else:
        pos_neg_metrics.update({
            "sample_distribution/num_positive": 0,
            "response_length_positive/mean": 0.0,
            "response_length_positive/max": 0.0,
            "response_length_positive/min": 0.0,
        })
    
    if num_negative > 0:
        negative_response_length = response_length[negative_mask]
        pos_neg_metrics.update({
            "sample_distribution/num_negative": num_negative,
            "response_length_negative/mean": torch.mean(negative_response_length).detach().item(),
            "response_length_negative/max": torch.max(negative_response_length).detach().item(),
            "response_length_negative/min": torch.min(negative_response_length).detach().item(),
        })
    else:
        pos_neg_metrics.update({
            "sample_distribution/num_negative": 0,
            "response_length_negative/mean": 0.0,
            "response_length_negative/max": 0.0,
            "response_length_negative/min": 0.0,
        })
    
    # Add total number of samples for reference
    pos_neg_metrics["sample_distribution/total"] = num_positive + num_negative
    
    # If using keep_neg_ratio (i.e., some samples are filtered), also compute distribution for kept samples
    if "winner_mask" in batch.batch:
        # winner_mask indicates which samples are actually used for training (not filtered)
        # Compute sample-level mask: a sample is "kept" if any of its tokens has winner_mask=1
        sample_kept_mask = (batch.batch["winner_mask"] * response_mask).sum(dim=-1) > 0
        
        # Compute kept sample distribution
        kept_positive_mask = positive_mask & sample_kept_mask
        kept_negative_mask = negative_mask & sample_kept_mask
        
        num_kept_positive = kept_positive_mask.sum().item()
        num_kept_negative = kept_negative_mask.sum().item()
        num_filtered = (~sample_kept_mask).sum().item()
        
        pos_neg_metrics.update({
            "sample_distribution/num_kept_positive": num_kept_positive,
            "sample_distribution/num_kept_negative": num_kept_negative,
            "sample_distribution/num_filtered": num_filtered,
            "sample_distribution/kept_total": num_kept_positive + num_kept_negative,
        })
    
    # For GRPO (with keep_neg_ratio < 1.0 or keep_pos_ratio < 1.0), also compute metrics for their filtered advantages
    additional_metrics = {}
    if "advantages_original" in batch.batch:
        # This means we're using GRPO with sample filtering (keep_neg_ratio < 1.0 or keep_pos_ratio < 1.0)
        # batch["advantages"] contains filtered advantages (used for training)
        # If compute_new_adv=True, these are recomputed advantages; otherwise they are original advantages
        valid_adv_specific = torch.masked_select(advantages, response_mask)
        
        # Filter out zeros (filtered samples) for more meaningful statistics
        non_zero_adv = valid_adv_specific[valid_adv_specific != 0.0]
        
        if non_zero_adv.numel() > 0:
            # Compute metrics for non-zero advantages (kept samples)
            additional_metrics = {
                "critic/advantages_processed/mean": torch.mean(non_zero_adv).detach().item(),
                "critic/advantages_processed/max": torch.max(non_zero_adv).detach().item(),
                "critic/advantages_processed/min": torch.min(non_zero_adv).detach().item(),
                "critic/advantages_processed/std": torch.std(non_zero_adv).detach().item(),
            }
        else:
            # All samples filtered (should not happen in practice)
            additional_metrics = {
                "critic/advantages_processed/mean": 0.0,
                "critic/advantages_processed/max": 0.0,
                "critic/advantages_processed/min": 0.0,
                "critic/advantages_processed/std": 0.0,
            }

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length (all samples)
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (excluding cutoff samples) - for plotting evolution curve
        **compute_non_cutoff_response_length_metrics(response_length, max_response_length),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # Positive/Negative sample distribution and response length statistics
        **pos_neg_metrics,
        # GRPO filtered advantages metrics (when keep_neg_ratio < 1.0 or keep_pos_ratio < 1.0)
        **additional_metrics,
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], num_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }
