import re
from typing import Dict
import math

def grade_answer_gaussian(
    pred,
    gt,
    r_min: float = 0.05,     # 在 diff=1 处的目标最小奖励
    diff_at_rmin: float = 1.0,  # "相差多少分"时衰减到 r_min（原始分值尺度）
    use_floor: bool = True,
) -> float:
    try:
        # 转换为浮点数并检查有效性
        pred_val = float(pred)
        gt_val = float(gt)
        
        # 检查输入是否为 NaN 或 inf
        if not math.isfinite(pred_val) or not math.isfinite(gt_val):
            print(f"Warning: Invalid input in grade_answer_gaussian: pred={pred}, gt={gt}")
            return 0.0
        
        # 检查预测范围
        if pred_val < 1.0 or pred_val > 5.0:
            return 0.0
        
        # 归一误差到[0,1]，gt/pred在[1,5]
        d = abs(pred_val - gt_val) / 4.0
        
        # 保护 diff_at_rmin 不为 0
        diff_at_rmin_safe = max(abs(float(diff_at_rmin)), 1e-6)
        d0 = diff_at_rmin_safe / 4.0

        # 数值保护
        r_min_clamped = max(min(float(r_min), 0.999999), 1e-6)

        # 由 r(d0)=r_min 反推 sigma（高斯）
        sigma = d0 / math.sqrt(2.0 * math.log(1.0 / r_min_clamped))

        # 高斯衰减（标量）
        r = math.exp(- (d ** 2) / (2.0 * sigma ** 2))

        # 地板（避免稀疏）
        if use_floor:
            r = max(r, r_min_clamped)
        
        # 确保在 [0, 1] 范围内
        r = max(0.0, min(r, 1.0))
        
        # 最终检查结果是否有效
        if not math.isfinite(r):
            print(f"Warning: Invalid result in grade_answer_gaussian: r={r}, pred={pred}, gt={gt}, d={d}, sigma={sigma}")
            return 0.0

        return float(r)
    
    except (ValueError, TypeError, ZeroDivisionError) as e:
        print(f"Warning: Exception in grade_answer_gaussian: {e}, pred={pred}, gt={gt}")
        return 0.0

def grade_answer_laplace(
    pred,
    gt,
    r_min: float = 0.05,      # 在 diff=1 处的目标最小奖励
    diff_at_rmin: float = 1.0,  # "相差多少分"时衰减到 r_min（原始分值尺度）
    use_floor: bool = True,
) -> float:
    """Compute Laplace-shaped point-wise reward for regression QA tasks."""
    try:
        # 转换为浮点数并检查有效性
        pred_val = float(pred)
        gt_val = float(gt)
        
        # 检查输入是否为 NaN 或 inf
        if not math.isfinite(pred_val) or not math.isfinite(gt_val):
            print(f"Warning: Invalid input in grade_answer_laplace: pred={pred}, gt={gt}")
            return 0.0
        
        # 检查预测范围
        if pred_val < 1.0 or pred_val > 5.0:
            return 0.0

        # 归一误差到 [0,1]，gt/pred 在 [1,5]
        d = abs(pred_val - gt_val) / 4.0
        
        # 保护 diff_at_rmin 不为 0
        diff_at_rmin_safe = max(abs(float(diff_at_rmin)), 1e-6)
        d0 = diff_at_rmin_safe / 4.0

        # 数值保护
        r_min_clamped = max(min(float(r_min), 0.999999), 1e-6)

        # 由 r(d0)=r_min 反推 τ（拉普拉斯）
        tau = d0 / math.log(1.0 / r_min_clamped)

        # 拉普拉斯衰减（标量）
        r = math.exp(- d / tau)

        # 地板（避免稀疏）
        if use_floor:
            r = max(r, r_min_clamped)
        
        # 确保在 [0, 1] 范围内
        r = max(0.0, min(r, 1.0))
        
        # 最终检查结果是否有效
        if not math.isfinite(r):
            print(f"Warning: Invalid result in grade_answer_laplace: r={r}, pred={pred}, gt={gt}, d={d}, tau={tau}")
            return 0.0

        return float(r)
    
    except (ValueError, TypeError, ZeroDivisionError) as e:
        print(f"Warning: Exception in grade_answer_laplace: {e}, pred={pred}, gt={gt}")
        return 0.0

def grade_answer_l1(
    pred,
    gt,
    r_min: float = 0.05,      # 在 diff=diff_at_rmin 处的目标最小奖励
    diff_at_rmin: float = 1.0,  # "相差多少分"时衰减到 r_min（原始分值尺度）
    use_floor: bool = True,
) -> float:
    """Compute L1-based point-wise reward: r = 1 - coeff * |pred - gt|"""
    try:
        # 转换为浮点数并检查有效性
        pred_val = float(pred)
        gt_val = float(gt)
        
        # 检查输入是否为 NaN 或 inf
        if not math.isfinite(pred_val) or not math.isfinite(gt_val):
            print(f"Warning: Invalid input in grade_answer_l1: pred={pred}, gt={gt}")
            return 0.0
        
        # 检查预测范围
        if pred_val < 1.0 or pred_val > 5.0:
            return 0.0

        # 计算实际误差
        diff = abs(pred_val - gt_val)

        # 数值保护
        r_min_clamped = max(min(float(r_min), 0.999999), 1e-6)
        
        # 保护 diff_at_rmin 不为 0
        diff_at_rmin_safe = max(abs(float(diff_at_rmin)), 1e-6)

        # 由 r(diff_at_rmin) = r_min 反推 coeff
        # r_min = 1 - coeff * diff_at_rmin
        # coeff = (1 - r_min) / diff_at_rmin
        coeff = (1.0 - r_min_clamped) / diff_at_rmin_safe

        # L1 线性衰减
        r = 1.0 - coeff * diff

        # 地板（避免稀疏）
        if use_floor:
            r = max(r, r_min_clamped)
        else:
            # 确保非负
            r = max(r, 0.0)

        # 确保在 [0, 1] 范围内
        r = max(0.0, min(r, 1.0))
        
        # 最终检查结果是否有效
        if not math.isfinite(r):
            print(f"Warning: Invalid result in grade_answer_l1: r={r}, pred={pred}, gt={gt}, diff={diff}, coeff={coeff}")
            return 0.0

        return float(r)
    
    except (ValueError, TypeError) as e:
        print(f"Warning: Exception in grade_answer_l1: {e}, pred={pred}, gt={gt}")
        return 0.0

def grade_answer_l2(
    pred,
    gt,
    r_min: float = 0.05,      # 在 diff=diff_at_rmin 处的目标最小奖励
    diff_at_rmin: float = 1.0,  # "相差多少分"时衰减到 r_min（原始分值尺度）
    use_floor: bool = True,
) -> float:
    """Compute L2-based point-wise reward: r = 1 - coeff * (pred - gt)^2"""
    try:
        # 转换为浮点数并检查有效性
        pred_val = float(pred)
        gt_val = float(gt)
        
        # 检查输入是否为 NaN 或 inf
        if not math.isfinite(pred_val) or not math.isfinite(gt_val):
            print(f"Warning: Invalid input in grade_answer_l2: pred={pred}, gt={gt}")
            return 0.0
        
        # 检查预测范围
        if pred_val < 1.0 or pred_val > 5.0:
            return 0.0

        # 计算实际误差
        diff = abs(pred_val - gt_val)

        # 数值保护
        r_min_clamped = max(min(float(r_min), 0.999999), 1e-6)
        
        # 保护 diff_at_rmin 不为 0
        diff_at_rmin_safe = max(abs(float(diff_at_rmin)), 1e-6)

        # 由 r(diff_at_rmin) = r_min 反推 coeff
        # r_min = 1 - coeff * (diff_at_rmin)^2
        # coeff = (1 - r_min) / (diff_at_rmin)^2
        coeff = (1.0 - r_min_clamped) / (diff_at_rmin_safe ** 2)

        # L2 二次衰减
        r = 1.0 - coeff * (diff ** 2)

        # 地板（避免稀疏）
        if use_floor:
            r = max(r, r_min_clamped)
        else:
            # 确保非负
            r = max(r, 0.0)

        # 确保在 [0, 1] 范围内
        r = max(0.0, min(r, 1.0))
        
        # 最终检查结果是否有效
        if not math.isfinite(r):
            print(f"Warning: Invalid result in grade_answer_l2: r={r}, pred={pred}, gt={gt}, diff={diff}, coeff={coeff}")
            return 0.0

        return float(r)
    
    except (ValueError, TypeError) as e:
        print(f"Warning: Exception in grade_answer_l2: {e}, pred={pred}, gt={gt}")
        return 0.0


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def accuracy_reward_gaussian(
    predict_str: str,
    ground_truth: str,
    r_min=0.05,  # 奖励地板；“diff=1”处的目标值也就是 r_min
    diff_at_rmin=1.0,   # 在“相差多少分”时把奖励压到 r_min；默认 1 分
    use_floor=True,     # 是否启用地板裁剪
) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        return grade_answer_gaussian(given_answer, ground_truth.strip(), r_min, diff_at_rmin, use_floor)

    except Exception as e:
        # print(e)
        pass

    return 0.0

def accuracy_reward_laplace(
    predict_str: str,
    ground_truth: str,
    r_min=0.05,  # 奖励地板；"diff=1"处的目标值也就是 r_min
    diff_at_rmin=1.0,   # 在"相差多少分"时把奖励压到 r_min；默认 1 分
    use_floor=True,     # 是否启用地板裁剪
) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        return grade_answer_laplace(given_answer, ground_truth.strip(), r_min, diff_at_rmin, use_floor)

    except Exception as e:
        # print(e)
        pass

    return 0.0

def accuracy_reward_l1(
    predict_str: str,
    ground_truth: str,
    r_min=0.05,  # 奖励地板；"diff=diff_at_rmin"处的目标值也就是 r_min
    diff_at_rmin=1.0,   # 在"相差多少分"时把奖励压到 r_min；默认 1 分
    use_floor=True,     # 是否启用地板裁剪
) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        return grade_answer_l1(given_answer, ground_truth.strip(), r_min, diff_at_rmin, use_floor)

    except Exception as e:
        # print(e)
        pass

    return 0.0

def accuracy_reward_l2(
    predict_str: str,
    ground_truth: str,
    r_min=0.05,  # 奖励地板；"diff=diff_at_rmin"处的目标值也就是 r_min
    diff_at_rmin=1.0,   # 在"相差多少分"时把奖励压到 r_min；默认 1 分
    use_floor=True,     # 是否启用地板裁剪
) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        return grade_answer_l2(given_answer, ground_truth.strip(), r_min, diff_at_rmin, use_floor)

    except Exception as e:
        # print(e)
        pass

    return 0.0

def compute_score_laplace(
    predict_str: str,
    ground_truth: str,
    format_weight: float = 0.5,
    strict_format: bool = True,
    r_min=0.05,  # 奖励地板；“diff=1”处的目标值也就是 r_min
    diff_at_rmin=1.0,   # 在“相差多少分”时把奖励压到 r_min；默认 1 分
    use_floor=True,     # 是否启用地板裁剪
)-> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward_laplace(predict_str, ground_truth, r_min, diff_at_rmin, use_floor)
    overall_score = accuracy_score + format_weight * format_score
    if strict_format:
        final_overall = overall_score if format_score == 1.0 else 0.0
    else:
        final_overall = overall_score
    return {
        "overall": final_overall,
        "format": format_score,
        "accuracy": accuracy_score,
    }

def compute_score_gaussian(
    predict_str: str,
    ground_truth: str,
    format_weight: float = 0.5,
    strict_format: bool = True,
    r_min=0.05,  # 奖励地板；"diff=1"处的目标值也就是 r_min
    diff_at_rmin=1.0,   # 在"相差多少分"时把奖励压到 r_min；默认 1 分
    use_floor=True,     # 是否启用地板裁剪
)-> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward_gaussian(predict_str, ground_truth, r_min, diff_at_rmin, use_floor)
    overall_score = accuracy_score + format_weight * format_score
    if strict_format:
        final_overall = overall_score if format_score == 1.0 else 0.0
    else:
        final_overall = overall_score
    return {
        "overall": final_overall,
        "format": format_score,
        "accuracy": accuracy_score,
    }

def compute_score_l1(
    predict_str: str,
    ground_truth: str,
    format_weight: float = 0.5,
    strict_format: bool = True,
    r_min=0.05,  # 奖励地板；"diff=diff_at_rmin"处的目标值也就是 r_min
    diff_at_rmin=1.0,   # 在"相差多少分"时把奖励压到 r_min；默认 1 分
    use_floor=True,     # 是否启用地板裁剪
)-> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward_l1(predict_str, ground_truth, r_min, diff_at_rmin, use_floor)
    overall_score = accuracy_score + format_weight * format_score
    if strict_format:
        final_overall = overall_score if format_score == 1.0 else 0.0
    else:
        final_overall = overall_score
    return {
        "overall": final_overall,
        "format": format_score,
        "accuracy": accuracy_score,
    }

def compute_score_l2(
    predict_str: str,
    ground_truth: str,
    format_weight: float = 0.5,
    strict_format: bool = True,
    r_min=0.05,  # 奖励地板；"diff=diff_at_rmin"处的目标值也就是 r_min
    diff_at_rmin=1.0,   # 在"相差多少分"时把奖励压到 r_min；默认 1 分
    use_floor=True,     # 是否启用地板裁剪
)-> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward_l2(predict_str, ground_truth, r_min, diff_at_rmin, use_floor)
    overall_score = accuracy_score + format_weight * format_score
    if strict_format:
        final_overall = overall_score if format_score == 1.0 else 0.0
    else:
        final_overall = overall_score
    return {
        "overall": final_overall,
        "format": format_score,
        "accuracy": accuracy_score,
    }