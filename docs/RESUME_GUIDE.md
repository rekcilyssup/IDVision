# IDVision: Resume Documentation & Technical Interview Guide

This guide provides tailored resume content, bullet points, technical breakdowns, and interview talking points focused strictly on **what you built and engineered** in the **IDVision** project.

---

## 📄 1. Resume One-Liners & Summaries

### Option A: 1-Sentence Resume / LinkedIn Summary
> "Engineered an end-to-end Vision-Language Model (VLM) extraction and evaluation pipeline using Qwen2-VL-2B and PEFT LoRA, achieving a +34.2% accuracy improvement on out-of-distribution noisy document images."

### Option B: 3-Sentence Project Description
> "Built **IDVision**, a full-stack VLM document intelligence framework that replaces traditional multi-stage OCR pipelines with fine-tuned Vision-Language Models (Qwen2-VL-2B). Developed a synthetic ID card rendering engine with PIL/Faker and simulated 5 real-world capture corruptions (blur, compression, tilt, glare) to benchmark robustness. Implemented PEFT LoRA fine-tuning on clean synthetic cards, achieving a 98.5% clean exact-match extraction accuracy and an interactive Flask/JS dashboard for real-time inference."

---

## 🎯 2. High-Impact Resume Bullet Points

Choose 3–5 bullet points tailored to the job role you are applying for (Machine Learning Engineer, Computer Vision Engineer, Full-Stack AI Engineer, or GenAI Developer):

### ML / Computer Vision Focus
- **Engineered an End-to-End Document VLM Extraction Pipeline**: Fine-tuned Qwen2-VL-2B-Instruct using **PEFT LoRA ($r=16, \alpha=32$)** for 6-field structured JSON extraction, replacing error-prone multi-stage OCR (Tesseract/PaddleOCR) cascades.
- **Designed Synthetic Data & Degradation Engine**: Developed a PIL/Faker rendering script generating 800+ multi-layout ID documents and a custom corruption pipeline modeling 5 mobile capture degradations (Gaussian blur, JPEG compression, perspective tilt, flash glare).
- **Benchmarked Out-of-Distribution Robustness**: Built an automated evaluation harness measuring **Exact Match (EM)** and **Normalized Levenshtein Edit Distance ($S_{\text{edit}}$)**, proving clean-trained LoRA models boost corrupted document extraction accuracy by **+31% to +43%**.
- **Interactive Web Dashboard**: Built a full-stack Flask/JS web application enabling real-time card generation, parameter-driven corruption simulation, Base64 image rendering, and live benchmark plotting.

### GenAI / LLM Engineer Focus
- **PEFT LoRA Model Optimization**: Optimized Qwen2-VL-2B visual and self-attention projections (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with `bfloat16` mixed precision, reducing trainable parameters by **>95%** while retaining high precision.
- **Structured JSON Output Constraints**: Implemented chat template formatting and a dual-stage JSON parser with regex fallback, guaranteeing valid schema conformance (`name`, `dob`, `id_number`, `address`, `issue_date`, `expiry_date`).
- **End-to-End Evaluation Framework**: Engineered a parallel evaluation harness comparing zero-shot baseline outputs against fine-tuned LoRA checkpoints across 600+ clean and corrupted document samples.

---

## 🛠️ 3. Tech Stack Tags for Skills Section

- **Core ML / VLM**: PyTorch, Hugging Face Transformers, Qwen2-VL, PEFT (LoRA), Accelerate, BitsAndBytes, `qwen-vl-utils`
- **Data Engineering & Vision**: PIL (Pillow), Faker, NumPy, OpenCV / ImageFilter, Levenshtein Distance, Pandas
- **Web App & UI**: Python, Flask, JavaScript (ES6+), HTML5, CSS3 (Glassmorphism design system), REST APIs
- **DevOps & Tools**: Git, GitHub, Jupyter / Google Colab T4 GPU, Virtualenv

---

## 🎙️ 4. Technical Interview Talking Points ("Tell Me About This Project")

When an interviewer asks: *"Can you walk me through your IDVision project?"*, use this structured response:

### 1. The Core Problem (Why I Built It)
> *"Traditional Document AI systems rely on OCR cascades: image preprocessing $\rightarrow$ text detection $\rightarrow$ character recognition $\rightarrow$ regex/NLP parsing. The major flaw is **error propagation**—if OCR misreads a single digit in an ID number or date, the downstream regex parser fails completely. I wanted to see if modern Vision-Language Models like Qwen2-VL could perform end-to-end structured JSON extraction directly from pixels, and how resilient they are to real-world camera noise like blur or glare."*

### 2. What I Engineered (Architecture & Implementation)
> *"I built the entire project from scratch in Python:*
> 1. *First, I created a synthetic identity document generator using PIL and Faker that renders driver's license layouts across 3 visual template variants and 5 color themes with ground-truth JSON annotations.*
> 2. *Second, I built a corruption engine to simulate mobile document capture noise: Gaussian blur, low-quality JPEG artifacts, perspective rotation tilt, and flash glare.*
> 3. *Third, I fine-tuned Qwen2-VL-2B using PEFT LoRA targeting both vision and language attention projections. Crucially, I trained strictly on clean data so I could test how well the model generalized to out-of-distribution noisy images.*
> 4. *Fourth, I implemented a evaluation harness measuring Exact Match Accuracy and Normalized Levenshtein Edit Distance, along with an interactive Flask web dashboard to demonstrate real-time card generation and corruption filtering."*

### 3. Key Findings & Quantitative Results
> *"The results were compelling: zero-shot Qwen2-VL struggled on corrupted documents (dropping to 25–45% exact match on rotated or blurry cards). Fine-tuning with LoRA on clean data alone boosted clean accuracy to **98.5%** and dramatically improved corrupted document extraction to **68–91%**, proving that VLMs learn visual spatial structure and field context to recover corrupted text."*

---

## 💡 5. Technical Questions & Answers (Deep Dive Interview Prep)

### Q1: Why did you use synthetic data instead of real identity documents?
> **Answer**: Real identity documents contain PII (Personally Identifiable Information) and cannot be legally stored or trained on open repositories. Synthetic generation using PIL and Faker provided ground-truth annotations with zero labeling error, and allowed me to procedurally vary field lengths, layout placements, and color schemes.

### Q2: Why did you train ONLY on clean data rather than including degraded data in training?
> **Answer**: Training only on clean data isolated a critical research question: *Does LoRA fine-tuning teach the model structural layout semantics that generalize to unseen noise, or does it merely memorize noise patterns?* By keeping degraded data as a held-out test set, I proved that the model learned spatial context rather than overfitting to specific blur or glare artifacts.

### Q3: How did you handle malformed JSON outputs from the VLM?
> **Answer**: I implemented a dual-stage JSON parser in `eval_harness.py`. It first strips markdown block code tags (````json ... ````). If standard `json.loads` fails, it executes a regex extraction fallback (`re.search(r'(\{.*\})', text, re.DOTALL)`) to isolate and repair embedded JSON substrings.

### Q4: What metrics did you use to evaluate extraction quality?
> **Answer**: I evaluated two metrics:
> 1. **Exact Match (EM)**: A binary 1/0 score requiring string equality after whitespace and case normalization.
> 2. **Normalized Levenshtein Edit Distance Similarity ($S_{\text{edit}}$)**: Defined as $1.0 - \frac{\text{Levenshtein}(pred, gt)}{\max(|pred|, |gt|)}$. This captured near-miss character typos (e.g. `123 MAIN ST` vs `123 MAIN SI`) where Exact Match would otherwise penalize the prediction as a complete failure.

---

## 📁 Document File Reference
Saved in repository under: `docs/RESUME_GUIDE.md`
