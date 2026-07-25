#!/usr/bin/env python3
"""
Evaluation Harness for Document Extraction VLM (Qwen2-VL)
Runs zero-shot baseline or fine-tuned LoRA models on clean and degraded document image datasets.
Calculates field-level Exact Match Accuracy and Normalized Edit-Distance Similarity, saving detailed CSV reports.
"""

import os
import re
import glob
import json
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

def compute_edit_distance_score(pred_str: str, gt_str: str) -> float:
    """Compute normalized Levenshtein similarity score between 0.0 and 1.0."""
    pred_str = str(pred_str).strip().upper()
    gt_str = str(gt_str).strip().upper()
    
    if pred_str == gt_str:
        return 1.0
    if not pred_str and not gt_str:
        return 1.0
    
    max_len = max(len(pred_str), len(gt_str))
    if max_len == 0:
        return 1.0
        
    if Levenshtein is not None:
        dist = Levenshtein.distance(pred_str, gt_str)
    else:
        # Fallback DP Levenshtein implementation
        dp = [[0] * (len(gt_str) + 1) for _ in range(len(pred_str) + 1)]
        for i in range(len(pred_str) + 1):
            dp[i][0] = i
        for j in range(len(gt_str) + 1):
            dp[0][j] = j
        for i in range(1, len(pred_str) + 1):
            for j in range(1, len(gt_str) + 1):
                cost = 0 if pred_str[i - 1] == gt_str[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        dist = dp[len(pred_str)][len(gt_str)]
        
    score = max(0.0, 1.0 - (dist / max_len))
    return round(score, 4)

def parse_json_from_vlm_output(text: str) -> dict:
    """Robustly extract and parse JSON object from VLM output text."""
    if not text:
        return {}
        
    # Strip markdown block tags if present
    cleaned = re.sub(r"^```json\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {k.lower(): str(v) for k, v in data.items()}
    except Exception:
        pass
        
    # Regex fallback for embedded JSON object
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return {k.lower(): str(v) for k, v in data.items()}
        except Exception:
            pass
            
    return {}

def build_prompt_for_qwen2_vl():
    """Prompt template asking Qwen2-VL for structured JSON extraction."""
    return (
        "Extract all key-value fields from this ID document image into a JSON object matching this exact schema:\n"
        "{\n"
        '  "name": "<full name>",\n'
        '  "dob": "<YYYY-MM-DD>",\n'
        '  "id_number": "<id number>",\n'
        '  "address": "<full address>",\n'
        '  "issue_date": "<YYYY-MM-DD>",\n'
        '  "expiry_date": "<YYYY-MM-DD>"\n'
        "}\n"
        "Output ONLY the JSON object. Do not include markdown or explanatory text."
    )

def load_vlm_model_and_processor(model_path, adapter_path=None, use_cpu=False):
    """Load Qwen2-VL model and processor with optional PEFT adapter."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    print(f"Loading Qwen2-VL processor from '{model_path}'...")
    processor = AutoProcessor.from_pretrained(model_path)

    device_map = "cpu" if use_cpu or not torch.cuda.is_available() else "auto"
    torch_dtype = torch.float32 if use_cpu else torch.bfloat16

    print(f"Loading base model '{model_path}' (device_map={device_map}, dtype={torch_dtype})...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map
    )

    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        print(f"Loading LoRA adapter weights from '{adapter_path}'...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, processor

def run_vlm_inference(model, processor, image_path, prompt, use_cpu=False):
    """Perform single-image visual question answering inference using qwen-vl-utils if available."""
    import torch
    image = Image.open(image_path).convert("RGB")

    try:
        from qwen_vl_utils import process_vision_info
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except ImportError:
        # Standard HF Vision-Text fallback
        inputs = processor(text=prompt, images=image, return_tensors="pt")

    device = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

def mock_heuristic_inference_for_dry_run(gt_dict, degradation_type):
    """Mock inference engine for testing harness execution without GPU model load."""
    pred = dict(gt_dict)
    # Simulate zero-shot degradation error patterns
    if degradation_type == "blur":
        # Simulates OCR typo in address and id_number
        if "address" in pred:
            pred["address"] = pred["address"].replace("ST", "SI")
    elif degradation_type == "rotation":
        # Simulates swapped issue/expiry dates in rotated documents
        if "issue_date" in pred and "expiry_date" in pred:
            pred["issue_date"] = gt_dict["expiry_date"]
    elif degradation_type == "jpeg":
        if "id_number" in pred:
            pred["id_number"] = pred["id_number"].replace("B", "8")
    return json.dumps(pred)

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on Document Datasets.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Base model checkpoint ID or path.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA fine-tuned adapter directory.")
    parser.add_argument("--data_dirs", nargs="+", default=["data/clean", "data/degraded"], help="Directories containing images & JSON GTs.")
    parser.add_argument("--output_csv", type=str, default="results/baseline_results.csv", help="Path to save output evaluation CSV.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples evaluated per folder.")
    parser.add_argument("--dry_run", action="store_true", help="Run harness with mock inference to verify pipeline without GPU.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    prompt = build_prompt_for_qwen2_vl()

    if not args.dry_run:
        model, processor = load_vlm_model_and_processor(args.model_path, args.adapter_path, use_cpu=args.cpu)
    else:
        print("--- RUNNING IN HARNESS DRY-RUN MODE (MOCK VLM INFERENCE) ---")
        model, processor = None, None

    eval_fields = ["name", "dob", "id_number", "address", "issue_date", "expiry_date"]
    results_rows = []

    for ddir in args.data_dirs:
        if not os.path.exists(ddir):
            print(f"Directory '{ddir}' not found. Skipping...")
            continue

        json_paths = sorted(glob.glob(os.path.join(ddir, "*.json")))
        if args.max_samples:
            json_paths = json_paths[:args.max_samples]

        print(f"Evaluating {len(json_paths)} files in '{ddir}'...")

        for json_path in tqdm(json_paths):
            stem = os.path.splitext(os.path.basename(json_path))[0]
            img_path = os.path.join(ddir, f"{stem}.png")
            if not os.path.exists(img_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                gt_dict = json.load(f)

            deg_type = gt_dict.get("_meta_degradation_type", "clean" if "clean" in ddir else "degraded")

            if args.dry_run:
                raw_output = mock_heuristic_inference_for_dry_run(gt_dict, deg_type)
            else:
                raw_output = run_vlm_inference(model, processor, img_path, prompt, use_cpu=args.cpu)

            pred_dict = parse_json_from_vlm_output(raw_output)

            for field in eval_fields:
                gt_val = str(gt_dict.get(field, "")).strip()
                pred_val = str(pred_dict.get(field, "")).strip()

                exact_match = 1 if gt_val.upper() == pred_val.upper() and len(gt_val) > 0 else 0
                edit_dist_score = compute_edit_distance_score(pred_val, gt_val)

                results_rows.append({
                    "image_id": stem,
                    "degradation_type": deg_type,
                    "field": field,
                    "ground_truth": gt_val,
                    "prediction": pred_val,
                    "exact_match": exact_match,
                    "edit_distance_score": edit_dist_score
                })

    df = pd.DataFrame(results_rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nEvaluation completed! Summary saved to '{args.output_csv}'.")

    if not df.empty:
        summary = df.groupby(["degradation_type", "field"])[["exact_match", "edit_distance_score"]].mean()
        print("\n--- SUMMARY METRICS (EXACT MATCH & EDIT DISTANCE SIMILARITY) ---")
        print(summary.to_string())

if __name__ == "__main__":
    main()
