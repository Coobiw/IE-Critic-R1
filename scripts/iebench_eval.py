import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
from torch.utils.data import Dataset

import random
import numpy as np

from scipy import stats
import numpy as np
from scipy.optimize import curve_fit

from datasets import load_dataset

CoT_PROMPT = """\
<image><image> Edit Instruction: {edit_instruction}

Given the source image and the edited image, along with the editing instruction, please evaluate the overall quality of the edited image.
Consider the following aspects:
1. Text alignment: How well does the edit follow the instruction?
2. Fidelity: How well does the edited image preserve the unedited parts of the source image?
3. Quality: What is the perceptual quality of the edited image?

The overall rating should be a float between 1 and 5, with 1 representing very poor quality and 5 representing excellent quality.
Return your final answer as a number rounded to two decimal places. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>\
"""

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def performance_fit(y_label, y_output, func_fit=True):
    if func_fit:
        y_output_logistic = fit_function(y_label, y_output)
    else:
        y_output_logistic = y_output
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]

    return PLCC, SRCC, (PLCC+SRCC) / 2

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

class IE4k(Dataset):

    def __init__(self):
        super().__init__()
        self.ie_dataset = load_dataset("Coobiw/IE-Bench-4k", split="test")

    def __len__(self):
        return len(self.ie_dataset)

    def __getitem__(self,idx):
        item = self.ie_dataset[idx]
        gt = item['answer']
        edit_inst = item['edit_inst']
        query_template = CoT_PROMPT

        msgs = [
            {
                "role": "user", "content": [
                    {"type": "image", "image": item['images'][0]},
                    {"type": "image", "image": item['images'][1]},
                    {"type": "text", "text": query_template.format(edit_instruction=edit_inst)}
                ]
            },
            {
                "role": "assistant",
                "content": "<think>"
            }
        ]
        return {
            "messages": msgs,
            "mos_perception": float(gt),
        }

def model_gen(model, processor, messages):
    # Preparation for inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
    ]
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Batch Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False, temperature=0.)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts

def extract_answer_from_response(response: str):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=bool, action="store_true", default=False)
    parser.add_argument("--output_fname", type=str, default="results/ie_critic_r1_ie4k.json")
    args = parser.parse_args()
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Coobiw/IE-Critic-R1-7B", 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    
    max_pixels = 1048576 # 1024 x 1024
    min_pixels = 262144 # 512 x 512
    processor = AutoProcessor.from_pretrained("Coobiw/IE-Critic-R1-7B", min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side  = 'left'

    ie_4k = IE4k()
    output = []
    output_fname = args.output_fname

    eval_bs = 32
    indices = list(range(len(ie_4k)))[::eval_bs]
    l = len(ie_4k)
    for start_idx in tqdm(indices):
        if start_idx + eval_bs > l:
            items = [ie_4k[idx] for idx in range(start_idx, l)]
        else:
            items = [ie_4k[idx] for idx in range(start_idx, start_idx + eval_bs)]

        batch_messages = [item['messages'] for item in items]
        model_responses = model_gen(model, processor, batch_messages)

        for response_idx, model_response in enumerate(model_responses):
            item = items[response_idx]
            item['model_response'] = model_response
            item.pop("messages")
            output.append(item)

    if args.save:
        with open(output_fname, 'w') as fo:
            json.dump(output, fo, ensure_ascii=False, indent=4)
    
    y_label, y_out = [], []
    error_count = 0
    for i, item in enumerate(output):
        model_response = item['model_response']
        try:
            out = extract_answer_from_response(model_response.strip().rstrip("."))
            
            # Skip if extraction failed (returned None)
            if out is None:
                error_count += 1
                print(f"{i}th error:\t extraction returned None")
                continue
                
            y_out.append(out)
            y_label.append(float(item['mos_perception']))
        except Exception as e:
            error_count += 1
            print(f"{i}th error:\t", e)
            y_out.append(0)
            y_label.append(float(item['mos_perception']))
            
    print(error_count)
    output1 = performance_fit(y_label, y_out, func_fit=True)
    output2 = performance_fit(y_label, y_out, func_fit=False)

    print(output1)
    print(output2)
    
    if args.save:
        out_score = args.output_fname.replace(".json", "_score.txt")
        with open(out_score, 'w') as fo:
            fo.write(f"PLCC: {output1[0]}\n")
            fo.write(f"SRCC: {output1[1]}\n")
            fo.write(f"MainScore: {output1[2]}\n")
            fo.write(f"PLCC: {output2[0]}\n")
            fo.write(f"SRCC: {output2[1]}\n")
            fo.write(f"MainScore: {output2[2]}\n")