import os
import json
import base64
import io
import subprocess
import glob
from flask import Flask, render_template, jsonify, request, send_from_directory
from PIL import Image

# Import generator functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from generate_synthetic_data import CARD_THEMES, generate_random_record, render_id_card, load_fonts
from degrade_images import apply_gaussian_blur, apply_jpeg_compression, apply_rotation, apply_camera_glare

app = Flask(__name__)

# Initialize fonts
FONTS = load_fonts()

# Ensure standard directories exist
os.makedirs("data/clean", exist_ok=True)
os.makedirs("data/degraded", exist_ok=True)
os.makedirs("results", exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory("results", filename)

@app.route("/api/stats", methods=["GET"])
def get_stats():
    clean_images = len(glob.glob("data/clean/*.png"))
    degraded_images = len(glob.glob("data/degraded/*.png"))
    
    # Check if results are generated
    has_results = os.path.exists("results/baseline_results.csv")
    has_chart = os.path.exists("results/comparison_chart.png")
    
    return jsonify({
        "clean_count": clean_images,
        "degraded_count": degraded_images,
        "has_results": has_results,
        "has_chart": has_chart
    })

@app.route("/api/generate", methods=["POST"])
def generate_card():
    data = request.json or {}
    
    # Generate record
    record = {
        "name": data.get("name", "").strip().upper(),
        "dob": data.get("dob", "").strip(),
        "id_number": data.get("id_number", "").strip().upper(),
        "address": data.get("address", "").strip().upper(),
        "issue_date": data.get("issue_date", "").strip(),
        "expiry_date": data.get("expiry_date", "").strip()
    }
    
    # Fill missing values using Faker
    fake_record = generate_random_record()
    for k, v in record.items():
        if not v:
            record[k] = fake_record[k]
            
    # Selected layout & theme index
    layout_variant = int(data.get("layout", 0))
    theme_idx = int(data.get("theme", 0))
    theme = CARD_THEMES[theme_idx % len(CARD_THEMES)]
    
    # Render card
    card_img = render_id_card(record, layout_variant, theme, FONTS)
    
    # Encode as Base64 PNG
    buffered = io.BytesIO()
    card_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Optionally save to data/clean if requested
    save_to_clean = data.get("save_to_clean", False)
    if save_to_clean:
        clean_count = len(glob.glob("data/clean/*.png"))
        stem = f"id_card_{clean_count + 1:04d}"
        
        img_path = os.path.join("data/clean", f"{stem}.png")
        json_path = os.path.join("data/clean", f"{stem}.json")
        
        card_img.save(img_path, format="PNG")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            
    return jsonify({
        "image_b64": f"data:image/png;base64,{img_str}",
        "record": record
    })

@app.route("/api/degrade", methods=["POST"])
def degrade_card():
    data = request.json or {}
    image_data_url = data.get("image")
    deg_type = data.get("type", "blur")
    
    if not image_data_url:
        return jsonify({"error": "No image provided"}), 400
        
    # Parse base64 image
    header, encoded = image_data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Apply degradation based on sliders/types
    if deg_type == "blur":
        radius = float(data.get("blur_radius", 3.0))
        degraded = apply_gaussian_blur(img, radius=radius)
    elif deg_type == "jpeg":
        quality = int(data.get("jpeg_quality", 20))
        degraded = apply_jpeg_compression(img, quality=quality)
    elif deg_type == "rotation":
        angle = float(data.get("rotation_angle", 10.0))
        degraded = apply_rotation(img, angle=angle)
    elif deg_type == "glare":
        degraded = apply_camera_glare(img)
    elif deg_type == "combined":
        # Apply random combined corruptions
        degraded = apply_gaussian_blur(img, radius=2.5)
        degraded = apply_jpeg_compression(degraded, quality=35)
        degraded = apply_rotation(degraded, angle=5.0)
        degraded = apply_camera_glare(degraded)
    else:
        return jsonify({"error": "Invalid degradation type"}), 400
        
    # Encode output as base64
    buffered = io.BytesIO()
    degraded.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return jsonify({
        "image_b64": f"data:image/png;base64,{img_str}"
    })

@app.route("/api/run-eval", methods=["POST"])
def run_eval():
    try:
        # Run scripts/eval_harness.py in dry_run mode via subprocess
        eval_proc = subprocess.run(
            [sys.executable, "scripts/eval_harness.py", "--dry_run", "--output_csv", "results/baseline_results.csv"],
            capture_output=True, text=True, check=True
        )
        
        # Run scripts/analyze_results.py via subprocess to generate chart
        analyze_proc = subprocess.run(
            [sys.executable, "scripts/analyze_results.py", "--baseline_csv", "results/baseline_results.csv", "--output_chart", "results/comparison_chart.png"],
            capture_output=True, text=True, check=True
        )
        
        # Parse stdout of analyze_results.py to extract printed summary reports
        stdout = analyze_proc.stdout
        summary_marker = "======================================================="
        report_sections = stdout.split(summary_marker)
        
        report_content = ""
        if len(report_sections) >= 3:
            report_content = report_sections[1].strip()
        else:
            report_content = stdout
            
        return jsonify({
            "status": "success",
            "log": report_content,
            "has_chart": os.path.exists("results/comparison_chart.png")
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "error": f"Process failed: {e.stderr or e.stdout or str(e)}"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
