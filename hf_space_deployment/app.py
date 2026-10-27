import sys
print("=== Starting Audio Genre Predictor ===", flush=True)

import os
import threading
import torch
import gradio as gr
print("✅ Core libraries loaded", flush=True)

# Safe ZeroGPU integration
try:
    import spaces
    has_spaces = True
    print("✅ ZeroGPU spaces extension loaded", flush=True)
except Exception as e:
    has_spaces = False
    print(f"ℹ️ Running without spaces extension: {e}", flush=True)

from inference import loadAST, run_inference, GENRES
print("✅ Inference module loaded", flush=True)

MODEL_PATH = "best-ast-model.ckpt"

# Global model holder with thread-safe lazy initialization
_model = None
_model_lock = threading.Lock()

def get_model():
    """Lazily loads and caches the fine-tuned AST model."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("🚀 Loading AST fine-tuned model checkpoint...", flush=True)
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(f"Model file '{MODEL_PATH}' was not found.")
                _model = loadAST(MODEL_PATH)
                print("✅ AST model successfully initialized!", flush=True)
    return _model

def preload_model():
    """Background warm-up thread so app binds to port 7860 immediately."""
    try:
        get_model()
    except Exception as e:
        print(f"⚠️ Preload warning: {e}", flush=True)

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def create_bar_chart(genres, probabilities):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    x = range(len(genres))
    ax.bar(x, probabilities, color='#60b4ff', width=0.7)
    
    ax.grid(axis='y', linestyle='-', alpha=0.25, color='#8b949e')
    ax.set_axisbelow(True)
    
    ax.set_xticks(x)
    ax.set_xticklabels(genres, rotation=90, color='white', fontsize=10)
    ax.tick_params(colors='white', which='both', labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    max_val = max(probabilities)
    ax.set_ylim(0, max(0.35, max_val * 1.15))
    plt.tight_layout()
    return fig

def predict_genre_fn(audio_path):
    if not audio_path:
        raise gr.Error("Please upload an audio file first.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()
    try:
        model = get_model()
        predicted_genre, probabilities = run_inference(audio_path, model, device=device)
        elapsed = time.time() - start_time
        
        banner_html = f"""
<div style="background-color: #163622; border: 1px solid #238636; border-radius: 8px; padding: 14px 18px; margin-bottom: 8px;">
    <h2 style="color: #3fb950; margin: 0; font-size: 24px; font-weight: 700;">Predicted Genre: {predicted_genre.upper()}</h2>
</div>
<p style="color: #8b949e; font-size: 14px; margin: 0 0 12px 0;">⏱️ Total prediction time: {elapsed:.2f} seconds.</p>
"""
        chart_fig = create_bar_chart(GENRES, probabilities)
        results = {genre: float(prob) for genre, prob in zip(GENRES, probabilities)}
        return banner_html, chart_fig, results
    except Exception as e:
        raise gr.Error(f"Inference error: {str(e)}")

# Apply ZeroGPU decorator if available
if has_spaces:
    predict_genre = spaces.GPU(duration=60)(predict_genre_fn)
else:
    predict_genre = predict_genre_fn

# ==========================================
# GRADIO UI SETUP
# ==========================================
custom_css = """
#header { text-align: center; margin-bottom: 1.5rem; }
#main-card { max-width: 800px; margin: 0 auto; }
"""

with gr.Blocks(title="🎵 AST Genre Predictor", css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="main-card"):
        gr.Markdown(
            """
            # 🎵 Audio Genre Predictor
            ### Audio Spectrogram Transformer (AST) with Skip-Intro TTA
            Upload any audio track (`.wav`, `.mp3`, `.m4a`, `.flac`) to predict its musical genre among 10 categories.
            Powered by **Hugging Face ZeroGPU**.
            """,
            elem_id="header"
        )

        with gr.Row():
            audio_input = gr.Audio(
                label="Upload or Record Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )

        predict_btn = gr.Button("🚀 Predict Genre", variant="primary", size="lg")

        banner_output = gr.HTML(visible=False)
        chart_output = gr.Plot(label="Genre Probabilities Distribution", visible=False)
        label_output = gr.Label(
            label="Genre Prediction & Confidence",
            num_top_classes=5,
            visible=False
        )

        def on_predict_click(audio_path):
            banner, chart, results = predict_genre(audio_path)
            return (
                gr.update(value=banner, visible=True),
                gr.update(value=chart, visible=True),
                gr.update(value=results, visible=True)
            )

        predict_btn.click(
            fn=on_predict_click,
            inputs=audio_input,
            outputs=[banner_output, chart_output, label_output]
        )

        with gr.Accordion("ℹ️ Architecture & Pipeline Details", open=False):
            gr.Markdown(
                """
                - **Backbone:** Fine-tuned `MIT/ast-finetuned-audioset-10-10-0.4593` (Vision Transformer on Mel-spectrograms).
                - **Skip-Intro TTA (Test-Time Augmentation):** Musical intros frequently lack genre-defining motifs. 
                  This model skips the intro and evaluates 3 overlapping temporal crops, averaging predictions.
                - **Genres Supported:** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.
                """
            )

# Warm up model in background thread
threading.Thread(target=preload_model, daemon=True).start()

print("🚀 Launching Gradio web server on 0.0.0.0:7860...", flush=True)
demo.launch(server_name="0.0.0.0", server_port=7860)
