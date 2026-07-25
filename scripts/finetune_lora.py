#!/usr/bin/env python3
"""
PEFT LoRA Fine-Tuning Script for Qwen2-VL-2B-Instruct
Fine-tunes Qwen2-VL on clean synthetic document images for structured field extraction.
"""

import os
import glob
import json
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

class DocumentVLMDataset(Dataset):
    """PyTorch Dataset loading clean ID images and ground-truth JSON target messages."""
    
    def __init__(self, data_dir, processor, max_samples=None):
        self.data_dir = data_dir
        self.processor = processor
        
        json_paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        if max_samples:
            json_paths = json_paths[:max_samples]
            
        self.samples = []
        for jpath in json_paths:
            stem = os.path.splitext(os.path.basename(jpath))[0]
            ipath = os.path.join(data_dir, f"{stem}.png")
            if os.path.exists(ipath):
                with open(jpath, "r", encoding="utf-8") as f:
                    gt = json.load(f)
                self.samples.append((ipath, gt))
                
        print(f"Loaded {len(self.samples)} training pairs from '{data_dir}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_dict = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Format target JSON response string
        clean_gt = {k: v for k, v in gt_dict.items() if not k.startswith("_")}
        target_json_str = json.dumps(clean_gt, indent=2)
        
        user_prompt = (
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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": target_json_str
            }
        ]

        try:
            from qwen_vl_utils import process_vision_info
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        except ImportError:
            text_prompt = f"User: {user_prompt}\nAssistant: {target_json_str}"
            inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")

        # Squeeze batch dimension for DataLoader collator
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        # For causal LM training, labels are input_ids
        item["labels"] = item["input_ids"].clone()
        return item

def plot_training_loss(log_history, output_path):
    """Plot training loss progression over steps."""
    steps = []
    losses = []
    for entry in log_history:
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
            
    if steps and losses:
        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses, marker='o', color='#2b5c8f', linewidth=2, label="Train Loss")
        plt.title("Qwen2-VL LoRA Fine-Tuning Loss Curve", fontsize=14, fontweight='bold')
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Training loss curve saved to '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL with PEFT LoRA.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Base HuggingFace model ID.")
    parser.add_argument("--train_dir", type=str, default="data/clean", help="Clean dataset directory for training.")
    parser.add_argument("--output_dir", type=str, default="results/qwen2_vl_lora_checkpoint", help="Directory to save LoRA adapters.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for testing.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Initializing Qwen2-VL LoRA training from '{args.model_path}'...")

    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Configure PEFT LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if not torch.cuda.is_available():
        print("WARNING: CUDA GPU not detected! Training on CPU is very slow. Set max_samples for small demo runs.")

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=device_map
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = DocumentVLMDataset(args.train_dir, processor, max_samples=args.max_samples)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting LoRA fine-tuning...")
    train_result = trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA weights and processor to '{args.output_dir}'.")

    plot_training_loss(trainer.state.log_history, os.path.join("results", "training_loss.png"))

if __name__ == "__main__":
    main()
