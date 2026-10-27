---
title: AST Genre Predictor
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🎵 Messy Mashup AST Predictor

This Space uses a custom Docker container and ZeroGPU to run inference on a 10-genre audio classifier.

### Architecture
- **Framework:** Streamlit + PyTorch
- **Environment:** Custom Docker (Python 3.11)
- **Hardware:** Hugging Face ZeroGPU (Dynamic A100 Allocation)
- **Model:** Audio Spectrogram Transformer (AST) Fine-Tuned

### How it Works
1. Upload a 30-second audio track (`.wav`, `.mp3`, `.m4a`).
2. The track is dynamically converted into a 128-bin Mel-Spectrogram.
3. The Space requests a ZeroGPU instance (or runs locally on CPU/GPU).
4. The AST processes 3 overlapping Test-Time Augmentation (TTA) crops.
5. The predicted genre and top probabilities are displayed!

### Running Locally with Docker

The model checkpoint is decoupled from the Docker image build and attached via an external volume mount.

#### Option 1: Docker Compose (Recommended)
Make sure `best-ast-model.ckpt` is present in the `production/` directory:
```bash
cd production
docker compose up --build
```
Access the application at `http://localhost:7860`.

#### Option 2: Docker Run with Volume Mount
```bash
cd production
docker build -t messy-mashup-app .
docker run -d -p 7860:7860 -v $(pwd)/best-ast-model.ckpt:/app/best-ast-model.ckpt:ro --name messy-mashup-classifier messy-mashup-app
```