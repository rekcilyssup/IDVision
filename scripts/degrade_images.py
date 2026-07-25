#!/usr/bin/env python3
"""
Synthetic Image Degradation Pipeline
Applies realistic corruptions (Gaussian Blur, JPEG compression, Rotation, Glare, Combined)
to clean synthetic ID document images and saves degraded images with labeled metadata.
"""

import os
import glob
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

def apply_gaussian_blur(img, radius=None):
    """Apply random Gaussian blur."""
    if radius is None:
        radius = random.uniform(2.5, 5.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_jpeg_compression(img, quality=None):
    """Apply low-quality JPEG compression artifacts."""
    if quality is None:
        quality = random.randint(15, 25)
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def apply_rotation(img, angle=None):
    """Apply small rotation with background fill."""
    if angle is None:
        angle = random.choice([-1, 1]) * random.uniform(8.0, 15.0)
    bg_color = (220, 222, 225)
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=bg_color)
    return rotated

def apply_camera_glare(img):
    """Simulate camera flash or light reflection with semi-transparent radial glare gradient."""
    width, height = img.size
    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Random glare center point
    cx = random.randint(width // 4, 3 * width // 4)
    cy = random.randint(height // 4, 3 * height // 4)
    max_r = int(max(width, height) * random.uniform(0.35, 0.65))
    
    # Draw concentric circles with decreasing alpha
    for r in range(max_r, 0, -5):
        alpha = int(140 * (1.0 - (r / max_r) ** 1.5))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 240, alpha))
        
    img_rgba = img.convert("RGBA")
    composite = Image.alpha_composite(img_rgba, overlay)
    return composite.convert("RGB")

def apply_degradation(img, deg_type):
    """Dispatch image through requested degradation function."""
    if deg_type == "blur":
        return apply_gaussian_blur(img)
    elif deg_type == "jpeg":
        return apply_jpeg_compression(img)
    elif deg_type == "rotation":
        return apply_rotation(img)
    elif deg_type == "glare":
        return apply_camera_glare(img)
    elif deg_type == "combined":
        # Random pick 2 degradations
        chosen = random.sample(["blur", "jpeg", "rotation", "glare"], 2)
        out = img
        for c in chosen:
            out = apply_degradation(out, c)
        return out
    else:
        raise ValueError(f"Unknown degradation type: {deg_type}")

def main():
    parser = argparse.ArgumentParser(description="Degrade synthetic ID images.")
    parser.add_argument("--input_dir", type=str, default="data/clean", help="Directory containing clean images and JSONs.")
    parser.add_argument("--output_dir", type=str, default="data/degraded", help="Directory to save degraded images and JSONs.")
    parser.add_argument("--degradations", nargs="+", default=["blur", "jpeg", "rotation", "glare", "combined"],
                        help="List of degradation types to apply.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    print(f"Found {len(json_files)} clean samples in '{args.input_dir}'.")
    print(f"Applying degradation types: {args.degradations}")

    processed_count = 0
    for json_file in json_files:
        stem = os.path.splitext(os.path.basename(json_file))[0]
        img_file = os.path.join(args.input_dir, f"{stem}.png")
        if not os.path.exists(img_file):
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        clean_img = Image.open(img_file).convert("RGB")

        for deg_type in args.degradations:
            degraded_img = apply_degradation(clean_img, deg_type)
            
            deg_stem = f"{stem}_{deg_type}"
            deg_img_path = os.path.join(args.output_dir, f"{deg_stem}.png")
            deg_json_path = os.path.join(args.output_dir, f"{deg_stem}.json")
            
            degraded_img.save(deg_img_path, format="PNG")
            
            # Save ground truth with degradation tag metadata
            deg_gt = dict(gt_data)
            deg_gt["_meta_degradation_type"] = deg_type
            
            with open(deg_json_path, "w", encoding="utf-8") as f:
                json.dump(deg_gt, f, indent=2)

            processed_count += 1

    print(f"Finished processing! Created {processed_count} degraded image/json pairs in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
