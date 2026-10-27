---
title: Ast Genre Predictor
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Audio Genre Classification with AST and Skip-Intro TTA
---

# 🎵 Messy Mashup: Audio Genre Predictor

Classify audio tracks and mashups into 10 musical genres using a fine-tuned **Audio Spectrogram Transformer (AST)** with **Skip-Intro Test-Time Augmentation (TTA)** running on **Hugging Face ZeroGPU**.

### Features
- **Transformer Backbone**: AST pre-trained on AudioSet with Spectrogram Patchification.
- **Skip-Intro TTA**: Overcomes intro ambiguity by taking 3 temporal crops after the musical intro and averaging their softmax distributions.
- **ZeroGPU Accelerated**: Dynamically claims A100 GPU compute during inference.
- **Supported Genres**: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.
