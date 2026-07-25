#!/usr/bin/env python3
"""
Visual Quality Assurance Script
Samples clean and degraded ID card images and builds a composite QA grid image
showing the visual corruptions and ground truth field overlays.
"""

import os
import glob
import json
import random
import argparse
from PIL import Image, ImageDraw, ImageFont

def draw_caption(img, gt_data, title_text):
    """Create a panel image containing the card image on top and GT key-values on bottom."""
    w, h = img.size
    caption_h = 160
    panel = Image.new("RGB", (w, h + caption_h), (255, 255, 255))
    panel.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(panel)
    # Header strip for panel caption
    draw.rectangle([0, h, w, h + 30], fill=(40, 50, 60))
    draw.text((10, h + 5), title_text, fill=(255, 255, 255))
    
    # Ground truth key values
    y_offset = h + 38
    fields = ["name", "id_number", "dob", "address", "issue_date", "expiry_date"]
    for i, f in enumerate(fields):
        val = gt_data.get(f, "N/A")
        if f == "address" and len(val) > 35:
            val = val[:35] + "..."
        x_pos = 10 if i % 2 == 0 else w // 2 + 10
        if i % 2 == 1:
            draw.text((x_pos, y_offset), f"{f}: {val}", fill=(30, 30, 30))
            y_offset += 24
        else:
            draw.text((x_pos, y_offset), f"{f}: {val}", fill=(30, 30, 30))
            
    return panel

def main():
    parser = argparse.ArgumentParser(description="Generate visual QA sample grid.")
    parser.add_argument("--clean_dir", type=str, default="data/clean", help="Path to clean dataset.")
    parser.add_argument("--degraded_dir", type=str, default="data/degraded", help="Path to degraded dataset.")
    parser.add_argument("--output_path", type=str, default="results/sample_qa_grid.png", help="Path for saved QA grid image.")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples per dataset to include.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    clean_jsons = sorted(glob.glob(os.path.join(args.clean_dir, "*.json")))
    degraded_jsons = sorted(glob.glob(os.path.join(args.degraded_dir, "*.json")))

    if not clean_jsons or not degraded_jsons:
        print(f"Warning: Ensure data exists in '{args.clean_dir}' and '{args.degraded_dir}'. Generating sample grid from whatever is available...")

    selected_clean = random.sample(clean_jsons, min(args.num_samples, len(clean_jsons))) if clean_jsons else []
    selected_degraded = random.sample(degraded_jsons, min(args.num_samples, len(degraded_jsons))) if degraded_jsons else []

    panels = []
    for c_json in selected_clean:
        stem = os.path.splitext(os.path.basename(c_json))[0]
        img_path = os.path.join(args.clean_dir, f"{stem}.png")
        if os.path.exists(img_path):
            with open(c_json, "r") as f:
                gt = json.load(f)
            img = Image.open(img_path).convert("RGB")
            # Resize for grid consistency
            img = img.resize((400, 250))
            panels.append(draw_caption(img, gt, f"CLEAN: {stem}"))

    for d_json in selected_degraded:
        stem = os.path.splitext(os.path.basename(d_json))[0]
        img_path = os.path.join(args.degraded_dir, f"{stem}.png")
        if os.path.exists(img_path):
            with open(d_json, "r") as f:
                gt = json.load(f)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((400, 250))
            deg_type = gt.get("_meta_degradation_type", "degraded")
            panels.append(draw_caption(img, gt, f"DEGRADED [{deg_type.upper()}]: {stem}"))

    if not panels:
        print("No valid image panels could be loaded. Run synthetic data generation first.")
        return

    # Tile panels into grid (2 columns)
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    panel_w, panel_h = panels[0].size
    grid_w = panel_w * cols + 20 * (cols + 1)
    grid_h = panel_h * rows + 20 * (rows + 1)

    grid = Image.new("RGB", (grid_w, grid_h), (230, 235, 240))
    for idx, panel in enumerate(panels):
        r = idx // cols
        c = idx % cols
        x = 20 + c * (panel_w + 20)
        y = 20 + r * (panel_h + 20)
        grid.paste(panel, (x, y))

    grid.save(args.output_path)
    print(f"Visual QA grid saved to '{args.output_path}'.")

if __name__ == "__main__":
    main()
