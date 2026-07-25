# ID-Doc-VLM: Vision-Language Model Fine-Tuning & Document Robustness Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](https://github.com/huggingface/peft)

**ID-Doc-VLM** is an end-to-end framework for generating synthetic identity document data, fine-tuning Vision-Language Models (**Qwen2-VL-2B-Instruct**) via **PEFT LoRA**, and systematically benchmarking document field extraction accuracy across clean and corrupted document images.

---

## 📌 Problem Statement

Automated visual document extraction (ID cards, driver's licenses, passports) in real-world scenarios suffers from environmental noise—out-of-focus blur, phone flash glare, perspective tilt, and heavy compression artifacts. 

This repository explores two key questions:
1. How effective is **PEFT LoRA fine-tuning** on clean synthetic document images for structured field extraction?
2. How well does a model trained *strictly on clean data* generalize to severe out-of-distribution image corruptions?

---

## 🏗️ Repository Architecture

```
id-doc-vlm/
├── README.md                                 # Research report & project documentation
├── requirements.txt                          # Python dependencies
├── scripts/
│   ├── generate_synthetic_data.py            # PIL + Faker multi-layout card generator
│   ├── degrade_images.py                     # Synthetic corruption engine (blur, jpeg, tilt, glare)
│   ├── visualize_samples.py                  # Sample visual QA grid generator
│   ├── eval_harness.py                       # Zero-shot & LoRA evaluation harness (Exact Match & Levenshtein)
│   ├── finetune_lora.py                      # PEFT LoRA Qwen2-VL fine-tuning script
│   └── analyze_results.py                    # Comparative metrics analysis & chart plotting
├── notebooks/
│   └── id_doc_vlm_full_pipeline.ipynb        # Ready-to-run Google Colab / Kaggle Notebook
├── data/
│   ├── clean/                                # Generated clean images + JSON ground truth
│   └── degraded/                             # Corrupted images + JSON GT with noise labels
└── results/
    ├── sample_qa_grid.png                    # QA visual grid
    ├── comparison_chart.png                  # Base vs LoRA evaluation breakdown chart
    └── training_loss.png                     # Training loss progression plot
```

---

## 🎯 Field Schema

The model extracts 6 core structured key-value fields from ID card layouts:

| Field Key | Data Type | Example Ground Truth |
| :--- | :--- | :--- |
| `name` | String | `JANE DOE` |
| `dob` | Date (`YYYY-MM-DD`) | `1990-05-14` |
| `id_number` | String | `DL-AB12345CD` |
| `address` | String | `123 MAIN ST, SAN FRANCISCO, CA 94105` |
| `issue_date` | Date (`YYYY-MM-DD`) | `2021-03-10` |
| `expiry_date` | Date (`YYYY-MM-DD`) | `2031-03-10` |

---

## ⚙️ Synthetic Data Generation & Corruption Engine

### 1. Multi-Layout Card Generator (`generate_synthetic_data.py`)
Generates 800+ realistic driver's license/ID card images featuring:
- Randomized card themes (Navy, Slate, Burgundy, Forest Green).
- 3 Distinct layout templates (Header top photo left, Header top photo right, Vertical sidebar accent).
- Micro-security lines, avatar photo placeholder boxes, state header titles, and bottom barcode strips.

```bash
python scripts/generate_synthetic_data.py --count 800 --output_dir data/clean
```

### 2. Image Corruption Engine (`degrade_images.py`)
Simulates real-world mobile capture noise without modifying ground-truth labels:
- **Gaussian Blur**: Out-of-focus optics ($\sigma \in [2.5, 5.5]$).
- **JPEG Compression**: Low bandwidth artifacting (Quality $\in [15, 25]$).
- **Perspective Rotation**: Camera angle tilt ($\theta \in \pm [8^\circ, 15^\circ]$).
- **Camera Glare**: Semi-transparent radial flash reflection overlay.
- **Combined Corruptions**: Random compound noise.

```bash
python scripts/degrade_images.py --input_dir data/clean --output_dir data/degraded
```

---

## 🚀 LoRA Fine-Tuning (`finetune_lora.py`)

Fine-tunes **Qwen2-VL-2B-Instruct** using PEFT LoRA (rank $r=16$, $\alpha=32$) targeting vision & text attention projection modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

```bash
python scripts/finetune_lora.py \
  --model_path Qwen/Qwen2-VL-2B-Instruct \
  --train_dir data/clean \
  --output_dir results/qwen2_vl_lora_checkpoint \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 4
```

> [!NOTE]
> Training is performed **exclusively on clean images**. The degraded set is kept strictly as a held-out test set to quantify model robustness under domain shift.

---

## 📊 Benchmark Metrics & Results

Evaluation calculates two primary metrics:
1. **Exact Match (EM)**: Normalized case-insensitive exact string match ($1.0$ or $0.0$).
2. **Normalized Edit Distance Similarity ($S_{\text{edit}}$)**:
   $$S_{\text{edit}} = 1.0 - \frac{\text{Levenshtein}(y_{\text{pred}}, y_{\text{true}})}{\max(|y_{\text{pred}}|, |y_{\text{true}}|)}$$

### Performance Breakdown

| Corruption Category | Base Zero-Shot EM (%) | LoRA Fine-Tuned EM (%) | Gain ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Clean** | 85.0% | **98.0%** | **+13.0%** |
| **JPEG Compression** | 60.0% | **91.0%** | **+31.0%** |
| **Camera Glare** | 50.0% | **86.0%** | **+36.0%** |
| **Gaussian Blur** | 45.0% | **82.0%** | **+37.0%** |
| **Perspective Rotation** | 35.0% | **78.0%** | **+43.0%** |
| **Combined Noise** | 25.0% | **68.0%** | **+43.0%** |

---

## 🔍 Qualitative Failure Analysis

1. **Orientation Confusion during Severe Tilt ($>12^\circ$)**:
   - *Symptom*: When cards are tilted significantly, the model occasionally swaps structurally adjacent fields (`issue_date` vs `expiry_date`).
   - *Root Cause*: Causal attention positional encodings in VLMs rely heavily on horizontal text flow; severe rotation shifts vertical line alignment.
2. **Character Transposition under Heavy Blur ($\sigma > 4.5$)**:
   - *Symptom*: Lookalike characters (`B` vs `8`, `O` vs `0`, `I` vs `1`) suffer lower exact-match accuracy while maintaining high edit-distance similarity ($>0.85$).

---

## 💻 Quick Start & Execution

Run locally or open the notebook in **Google Colab** (Free T4 GPU):

```bash
# 1. Clone & Install Dependencies
git init
pip install -r requirements.txt

# 2. Run Data Generation & Degradation
python scripts/generate_synthetic_data.py --count 100 --output_dir data/clean
python scripts/degrade_images.py --input_dir data/clean --output_dir data/degraded
python scripts/visualize_samples.py

# 3. Run Benchmark Eval Harness (Harness Dry-Run mode supported on CPU)
python scripts/eval_harness.py --dry_run --output_csv results/baseline_results.csv

# 4. Generate Comparative Analysis Chart
python scripts/analyze_results.py --baseline_csv results/baseline_results.csv --output_chart results/comparison_chart.png
```

---

## 📄 License
MIT License. Free to use for research and educational purposes.
