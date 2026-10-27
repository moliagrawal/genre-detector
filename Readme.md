# 🎵 Messy Mashup: Audio Genre Classification with AST & CNNs

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/-Pytorch%20Lightning-792ee5?logo=pytorch-lightning&logoColor=white)](https://www.pytorchlightning.ai/)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/HarshitSahu01/ast-genre-predictor)

## 🚀 Project Overview
This project targets the **Jan 2026 DLGenAI "Messy Mashup" Challenge**. The goal is to classify 30-second audio mashups into 10 musical genres. The primary difficulty lies in a significant **domain gap**: the test data consists of mixed stems (vocals, drums, bass, etc.) overlaid with heavy environmental noise from the **ESC-50 dataset**.

### 🏆 Key Results
| Metric | Value |
| :--- | :--- |
| **Top Macro F1** | **0.97327** (Audio Spectrogram Transformer) |
| Architectures Explored | AST (Transformer), EfficientNet-B0 (CNN), Custom 5-block CNN |
| Training Set Size | 14,000+ augmented audio chunks ("Megaset") |

---

## 🌐 Live Demo

> **Try it yourself — no setup required!**

**➡️ [https://huggingface.co/spaces/HarshitSahu01/ast-genre-predictor](https://huggingface.co/spaces/HarshitSahu01/ast-genre-predictor)**

Upload any `.wav`, `.mp3`, or `.m4a` file and get an instant genre prediction powered by the fine-tuned AST model running on **Hugging Face ZeroGPU** (dynamic A100 allocation).

---

## 🛠️ Tech Stack & Architecture

### 1. Data Pipeline & "Megaset"
To bridge the gap between clean training samples and noisy test data, I synthesized a custom training distribution:
* **Synthetic Mashups:** 10,000+ chunks created by mixing stems with random ESC-50 noise (SNR 5 to 15).
* **Pseudo-Labeling:** Re-integrated 449 high-confidence test predictions (>95%) into the training loop.
* **Augmentations:** Applied **SpecAugment** (Time/Frequency masking) and **Mixup** ($\alpha=0.2$) for robust regularization.

### 2. Model Zoo
| Model | Type | Logic | Macro F1 |
| :--- | :--- | :--- | :--- |
| **AST** | Transformer | Global attention on spectrogram patches; captures long-range temporal dependencies. | **0.97327** |
| **EfficientNet-B0** | CNN | Pre-trained on ImageNet; uses Depthwise Separable Convolutions to isolate textures. | 0.94974 |
| **Custom CNN** | CNN (Scratch) | 5-block architecture designed to learn local frequency patterns from scratch. | 0.90574 |

---

## 🧩 Technical Implementation

### Feature Extraction (Patchification)
Instead of standard text tokenization, this project uses **Spectrogram Patchification**.
1.  **Audio to Image:** 16kHz waveforms are converted to 128-bin Mel-spectrograms via `librosa`.
2.  **Patching:** The `ASTFeatureExtractor` splits the 2D spectrogram into 16x16 overlapping patches, allowing the Transformer to attend to localized acoustic events.

### Training Strategies
* **Soft Freeze-Thaw:** Differential learning rates for the AST backbone ($2e^{-5}$) vs. the classification head ($1e^{-4}$).
* **Skip-Intro TTA (Test-Time Augmentation):** A custom TTA strategy that skips the intro of a track and averages predictions across 3 overlapping temporal crops of the audio, correcting for genre-establishing riffs that often appear mid-song.
* **Optimization:** AdamW optimizer with a Cosine Annealing schedule.

---

## 🚢 Production Deployment

The fine-tuned AST model is served via a **Streamlit** app containerized in a **custom Docker image** on Hugging Face Spaces with **ZeroGPU**.

### Stack
| Component | Technology |
| :--- | :--- |
| Frontend/UI | Streamlit |
| Inference Runtime | PyTorch + PyTorch Lightning |
| Audio Processing | librosa |
| Containerization | Docker (Python 3.11 slim) |
| Hardware | Hugging Face ZeroGPU (A100) |

### Inference Pipeline (per request)
1. User uploads a `.wav`, `.mp3`, or `.m4a` file.
2. Audio is resampled to **16kHz** and converted to a **128-bin Mel-Spectrogram**.
3. The Space requests a **ZeroGPU** instance dynamically.
4. The **Skip-Intro TTA** strategy generates 3 overlapping crops.
5. The AST runs a forward pass on all 3 crops, and probabilities are **averaged**.
6. The GPU is released; the predicted genre and confidence bar chart are displayed.

```
production/
├── app.py          # Streamlit UI with ZeroGPU @spaces.GPU decorator
├── inference.py    # Model architecture + loadAST() + run_inference()
├── Dockerfile      # Custom Docker image (Python 3.11)
└── docker-compose.yaml
```

---

## 📈 Challenges & Solutions
* **Overfitting:** Solved by shifting from "pure" audio to a noisy "Megaset" and implementing heavy Dropout (0.5).
* **Engineering Bottlenecks:** Resolved Kaggle DataLoader deadlocks by optimizing `num_workers` and bypassed PyTorch 2.6 security errors during checkpoint loading with `weights_only=False`.
* **Genre Ambiguity at Intros:** Developed the **Skip-Intro TTA** strategy after noticing that genre-defining musical elements (e.g., guitar riffs, drum patterns) often appear only after the intro, causing misclassifications on the first crop.

---

## 🔮 Future Work: The Ensemble
The roadmap involves a **Hard-Voting Ensemble**. Since CNNs ("local listeners") and Transformers ("global listeners") make different mathematical errors, blending their prediction probabilities will likely push the F1 score beyond **0.99**.