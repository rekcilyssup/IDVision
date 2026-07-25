#!/usr/bin/env python3
"""
Synthetic ID Document Generator
Generates synthetic ID card images (driver's license / state ID layout) using PIL and Faker,
and saves ground-truth JSON files for fine-tuning and evaluation.
"""

import os
import json
import random
import argparse
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
from faker import Faker

fake = Faker()

# Color schemes for card themes
CARD_THEMES = [
    {"header_bg": (24, 43, 73), "header_fg": (255, 255, 255), "card_bg": (245, 247, 250), "accent": (41, 128, 185)},
    {"header_bg": (39, 110, 144), "header_fg": (255, 255, 255), "card_bg": (250, 250, 245), "accent": (230, 126, 34)},
    {"header_bg": (44, 62, 80), "header_fg": (255, 255, 255), "card_bg": (240, 244, 248), "accent": (39, 174, 96)},
    {"header_bg": (120, 40, 60), "header_fg": (255, 255, 255), "card_bg": (252, 250, 247), "accent": (142, 68, 173)},
    {"header_bg": (20, 80, 70), "header_fg": (255, 255, 255), "card_bg": (246, 248, 246), "accent": (211, 84, 0)},
]

TITLES = [
    "DRIVER LICENSE",
    "STATE IDENTIFICATION CARD",
    "NATIONAL IDENTITY CARD",
    "OFFICIAL DRIVER PERMIT"
]

AUTHORITIES = [
    "STATE OF CALIFORNIA",
    "STATE OF NEW YORK",
    "STATE OF TEXAS",
    "COMMONWEALTH OF SYNTHESIA",
    "DEPARTMENT OF MOTOR VEHICLES"
]

def load_fonts():
    """Attempt to load system fonts, fallback to default if not available."""
    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    
    label_font = None
    value_font = None
    header_font = None
    title_font = None
    
    for path in font_candidates:
        if os.path.exists(path):
            try:
                header_font = ImageFont.truetype(path, 22)
                title_font = ImageFont.truetype(path, 16)
                label_font = ImageFont.truetype(path, 11)
                value_font = ImageFont.truetype(path, 15)
                break
            except Exception:
                continue
                
    if label_font is None:
        header_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        value_font = ImageFont.load_default()
        
    return header_font, title_font, label_font, value_font

def draw_avatar_placeholder(draw, box, theme):
    """Draw a synthetic avatar box simulating a passport/ID photo."""
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], fill=(220, 225, 230), outline=(180, 185, 190), width=2)
    
    # Head circle
    cx = (x0 + x1) // 2
    head_r = (x1 - x0) // 4
    cy = y0 + (y1 - y0) // 3
    draw.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r], fill=(160, 170, 180))
    
    # Shoulders ellipse
    shoulder_w = (x1 - x0) * 3 // 4
    shoulder_h = (y1 - y0) // 2
    draw.ellipse([cx - shoulder_w // 2, y1 - shoulder_h // 2, cx + shoulder_w // 2, y1 + shoulder_h // 2], fill=(140, 150, 160))
    
    # Border overlay
    draw.rectangle([x0, y0, x1, y1], outline=theme["accent"], width=2)

def generate_random_record():
    """Generate fake user record."""
    dob_dt = fake.date_of_birth(minimum_age=18, maximum_age=75)
    issue_dt = fake.date_between(start_date='-5y', end_date='today')
    expiry_dt = issue_dt + timedelta(days=365 * random.randint(4, 8))
    
    id_num = fake.bothify(text='??#####??').upper()
    id_number = f"DL-{id_num}"
    
    street = fake.street_address().upper()
    city = fake.city().upper()
    state = fake.state_abbr()
    zipcode = fake.zipcode()
    address = f"{street}, {city}, {state} {zipcode}"
    
    return {
        "name": fake.name().upper(),
        "dob": dob_dt.strftime("%Y-%m-%d"),
        "id_number": id_number,
        "address": address,
        "issue_date": issue_dt.strftime("%Y-%m-%d"),
        "expiry_date": expiry_dt.strftime("%Y-%m-%d")
    }

def render_id_card(record, layout_variant, theme, fonts):
    """Render the synthetic ID card on PIL Image."""
    header_font, title_font, label_font, value_font = fonts
    width, height = 856, 540
    
    card = Image.new("RGB", (width, height), theme["card_bg"])
    draw = ImageDraw.Draw(card)
    
    # Draw Outer Border
    draw.rectangle([0, 0, width - 1, height - 1], outline=(200, 205, 210), width=3)
    
    authority = random.choice(AUTHORITIES)
    title = random.choice(TITLES)
    
    if layout_variant == 0:
        # Layout A: Header bar top (80px), Photo Left, Fields Right
        header_h = 85
        draw.rectangle([0, 0, width, header_h], fill=theme["header_bg"])
        draw.text((30, 18), authority, fill=theme["header_fg"], font=header_font)
        draw.text((30, 48), title, fill=(220, 230, 240), font=title_font)
        
        # Photo box
        photo_box = (35, 115, 215, 345)
        draw_avatar_placeholder(draw, photo_box, theme)
        
        # Micro text security strip
        draw.line([(35, 375), (width - 35, 375)], fill=theme["accent"], width=2)
        
        # Field positions (Right side)
        start_x = 245
        fields = [
            ("FN / LN / NAME", record["name"], 245, 115),
            ("LIC NO / ID NUM", record["id_number"], 245, 180),
            ("DATE OF BIRTH", record["dob"], 570, 180),
            ("ADDRESS", record["address"], 245, 245),
            ("ISSUE DATE", record["issue_date"], 245, 310),
            ("EXPIRY DATE", record["expiry_date"], 570, 310),
        ]
        
    elif layout_variant == 1:
        # Layout B: Header bar top, Photo Right, Fields Left
        header_h = 85
        draw.rectangle([0, 0, width, header_h], fill=theme["header_bg"])
        draw.text((30, 18), authority, fill=theme["header_fg"], font=header_font)
        draw.text((30, 48), title, fill=(220, 230, 240), font=title_font)
        
        # Photo box on right
        photo_box = (width - 215, 115, width - 35, 345)
        draw_avatar_placeholder(draw, photo_box, theme)
        
        draw.line([(35, 375), (width - 35, 375)], fill=theme["accent"], width=2)
        
        fields = [
            ("NAME", record["name"], 35, 115),
            ("ID NUMBER", record["id_number"], 35, 180),
            ("DOB", record["dob"], 350, 180),
            ("ADDRESS", record["address"], 35, 245),
            ("ISSUED", record["issue_date"], 35, 310),
            ("EXPIRES", record["expiry_date"], 350, 310),
        ]
        
    else:
        # Layout C: Left bar vertical accent (120px wide), Photo top-right, grid fields
        bar_w = 140
        draw.rectangle([0, 0, bar_w, height], fill=theme["header_bg"])
        
        # Vertical text or emblem in left bar
        draw.text((15, 30), authority.split()[0], fill=theme["header_fg"], font=title_font)
        draw.text((15, 55), "ID CARD", fill=(200, 210, 225), font=title_font)
        
        # Header banner on top right
        draw.rectangle([bar_w, 0, width, 65], fill=(230, 235, 242))
        draw.text((bar_w + 20, 20), title, fill=theme["header_bg"], font=header_font)
        
        # Photo box top right
        photo_box = (width - 195, 85, width - 35, 285)
        draw_avatar_placeholder(draw, photo_box, theme)
        
        fields = [
            ("FULL NAME", record["name"], bar_w + 20, 85),
            ("CARD NO.", record["id_number"], bar_w + 20, 150),
            ("DOB", record["dob"], bar_w + 20, 215),
            ("ADDRESS", record["address"], bar_w + 20, 280),
            ("ISSUE DATE", record["issue_date"], bar_w + 20, 370),
            ("EXPIRY DATE", record["expiry_date"], bar_w + 240, 370),
        ]
        
    # Render all text fields
    for label, val, x, y in fields:
        draw.text((x, y), label, fill=(120, 125, 135), font=label_font)
        
        # Handle long address wrapping
        if label == "ADDRESS" and len(val) > 40:
            parts = val.split(", ")
            if len(parts) >= 2:
                line1 = parts[0] + ","
                line2 = ", ".join(parts[1:])
                draw.text((x, y + 16), line1, fill=(20, 25, 35), font=value_font)
                draw.text((x, y + 36), line2, fill=(20, 25, 35), font=value_font)
            else:
                draw.text((x, y + 16), val, fill=(20, 25, 35), font=value_font)
        else:
            draw.text((x, y + 16), val, fill=(20, 25, 35), font=value_font)
            
    # Bottom footer strip / barcode placeholder
    draw.rectangle([35, height - 60, width - 35, height - 20], fill=(235, 238, 242), outline=(210, 215, 220))
    # Fake barcode lines
    draw.text((50, height - 48), "||| | ||||| ||| |||| || |||||| |||| | ||||| ||| |||| ||||| ||", fill=(50, 50, 50), font=value_font)
    
    return card

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ID card images and JSON GT.")
    parser.add_argument("--count", type=int, default=800, help="Number of synthetic ID images to generate.")
    parser.add_argument("--output_dir", type=str, default="data/clean", help="Directory to save generated images and JSONs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    fonts = load_fonts()
    print(f"Generating {args.count} synthetic ID card records into '{args.output_dir}'...")

    for i in range(1, args.count + 1):
        record = generate_random_record()
        theme = random.choice(CARD_THEMES)
        layout_variant = random.choice([0, 1, 2])
        
        img = render_id_card(record, layout_variant, theme, fonts)
        
        file_stem = f"id_card_{i:04d}"
        img_path = os.path.join(args.output_dir, f"{file_stem}.png")
        json_path = os.path.join(args.output_dir, f"{file_stem}.json")
        
        img.save(img_path, format="PNG")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        if i % 100 == 0 or i == args.count:
            print(f"[{i}/{args.count}] Generated {file_stem}.png & {file_stem}.json")

    print(f"Finished generating {args.count} clean samples in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
