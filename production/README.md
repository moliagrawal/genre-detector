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
3. The Space requests a ZeroGPU instance.
4. The AST processes 3 overlapping Test-Time Augmentation (TTA) crops.
5. The GPU is released, and the predicted genre is displayed!