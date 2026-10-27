import sys
print("=== Starting Audio Genre Predictor (AST) ===", flush=True)

import os
import time
import base64
import threading
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ==========================================
# ASSET ENCODING & CACHING
# ==========================================
def get_background_base64():
    """Encodes the vintage music studio background image to base64."""
    candidate_paths = [
        "assets/bg.jpg",
        "production/assets/bg.jpg",
        "/app/assets/bg.jpg"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    return ""

BG_BASE64 = get_background_base64()

# ==========================================
# SVG COMPONENT LIBRARY
# ==========================================
SVG_VINYL_HEADER = """<svg width="40" height="40" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 12px; filter: drop-shadow(0 2px 5px rgba(0,0,0,0.6));"><circle cx="50" cy="50" r="48" fill="#141414" stroke="#2a2a2a" stroke-width="2"/><circle cx="50" cy="50" r="40" stroke="#222222" stroke-width="1.2" stroke-dasharray="4 2"/><circle cx="50" cy="50" r="32" stroke="#262626" stroke-width="1.2"/><circle cx="50" cy="50" r="24" stroke="#1f1f1f" stroke-width="1.2" stroke-dasharray="3 2"/><circle cx="50" cy="50" r="18" fill="#c4614a" stroke="#8c3e2d" stroke-width="1.5"/><circle cx="50" cy="50" r="14" stroke="#e28c78" stroke-width="0.8" fill="none"/><circle cx="50" cy="50" r="4.5" fill="#141414"/></svg>"""

SVG_VINYL_RESULT = """<svg width="68" height="68" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink: 0; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));"><circle cx="50" cy="50" r="48" fill="#181818" stroke="#333333" stroke-width="2"/><circle cx="50" cy="50" r="41" stroke="#262626" stroke-width="1"/><circle cx="50" cy="50" r="34" stroke="#2c2c2c" stroke-width="1"/><circle cx="50" cy="50" r="27" stroke="#222222" stroke-width="1"/><circle cx="50" cy="50" r="18" fill="#c4614a" stroke="#8e3d2b" stroke-width="1.5"/><circle cx="50" cy="50" r="14" stroke="#df8571" stroke-width="0.8" fill="none"/><circle cx="50" cy="50" r="5" fill="#eddcc4"/></svg>"""

SVG_SINE_WAVE = """<svg width="190" height="60" viewBox="0 0 200 60" fill="none" xmlns="http://www.w3.org/2000/svg" style="opacity: 0.75; flex-shrink: 0;"><path d="M 0 30 C 25 10, 50 50, 75 22 C 100 -6, 125 58, 150 30 C 165 14, 185 36, 200 30" stroke="#bfa382" stroke-width="1.6" fill="none"/><path d="M 0 30 C 30 20, 60 40, 90 18 C 120 -2, 150 52, 175 30 C 185 22, 195 34, 200 30" stroke="#d5be9f" stroke-width="1.2" fill="none" opacity="0.75"/><path d="M 0 30 C 20 38, 45 12, 70 42 C 95 62, 120 18, 145 38 C 165 50, 185 24, 200 30" stroke="#a88e6e" stroke-width="1.0" fill="none" opacity="0.6"/></svg>"""

SVG_SOUNDWAVE = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#c4614a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 10v4M6 6v12M10 3v18M14 8v8M18 5v14M22 10v4"/></svg>"""

SVG_CHART_ICON = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#c4614a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>"""

SVG_TAG_ICON = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#c4614a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>"""

# ==========================================
# CUSTOM GENRE PROBABILITY CHART GENERATOR
# ==========================================
def render_styled_genre_chart(genres, probabilities, predicted_genre):
    """Generates an aesthetic dark-themed genre distribution chart matching the reference."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor='#16191b')
    ax.set_facecolor('#16191b')
    
    # Sage green for others, warm terracotta for predicted
    colors = ['#c2654f' if g.lower() == predicted_genre.lower() else '#6e8974' for g in genres]
    
    x = np.arange(len(genres))
    bars = ax.bar(x, probabilities, color=colors, width=0.62, zorder=3)
    
    # Grid lines & limits
    max_val = max(probabilities) if len(probabilities) > 0 else 0.5
    y_max = max(0.5, np.ceil(max_val * 10) / 10 + 0.05)
    ax.set_ylim(0, y_max)
    
    ax.grid(axis='y', color='#262c30', linestyle='-', linewidth=0.75, zorder=1)
    
    # Spines styling
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#333a3f')
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Ticks & labels
    ax.set_xticks(x)
    ax.set_xticklabels(genres, color='#9e988f', fontsize=10, fontfamily='sans-serif')
    ax.tick_params(axis='x', length=0, pad=8)
    
    y_ticks = np.arange(0, y_max + 0.01, 0.1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{val:.1f}" for val in y_ticks], color='#78726b', fontsize=9)
    ax.tick_params(axis='y', length=0, pad=8)
    
    plt.tight_layout()
    return fig

# ==========================================
# PREDICTION LOGIC
# ==========================================
def predict_genre_fn(audio_path):
    if not audio_path:
        raise gr.Error("Please upload or record an audio file first.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()
    try:
        model = get_model()
        predicted_genre, probabilities = run_inference(audio_path, model, device=device)
        elapsed = time.time() - start_time
        
        result_html = f"""
        <div class="result-card">
            <div class="result-left">
                {SVG_VINYL_RESULT}
                <div>
                    <div class="result-label">Predicted Genre</div>
                    <div class="result-genre">{predicted_genre.upper()}</div>
                </div>
            </div>
            {SVG_SINE_WAVE}
        </div>
        <div style="font-size: 0.82rem; color: #9c9285; margin-bottom: 14px; display: flex; align-items: center; gap: 6px;">
            <span>⏱️</span>
            <span>Total prediction time: <b>{elapsed:.2f} seconds</b>.</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 14px; margin-bottom: 6px;">
            <span style="font-size: 1.1rem;">📊</span>
            <h4 style="margin: 0; font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 600; color: #e8e2d8;">Genre Probabilities</h4>
        </div>
        """
        
        chart_fig = render_styled_genre_chart(GENRES, probabilities, predicted_genre)
        return result_html, chart_fig
    except Exception as e:
        raise gr.Error(f"Inference error: {str(e)}")

# Apply ZeroGPU decorator if available
if has_spaces:
    predict_genre = spaces.GPU(duration=60)(predict_genre_fn)
else:
    predict_genre = predict_genre_fn

# ==========================================
# GRADIO THEME & CUSTOM CSS
# ==========================================
bg_css = f"""
    background-image: linear-gradient(rgba(10, 8, 7, 0.68), rgba(10, 8, 7, 0.82)), url('data:image/jpeg;base64,{BG_BASE64}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
""" if BG_BASE64 else "background-color: #0d0b0a;"

custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

.gradio-container {{
    {bg_css}
    max-width: 1350px !important;
    margin: 0 auto !important;
    padding: 30px 20px !important;
    font-family: 'Inter', sans-serif !important;
    color: #f3ede2 !important;
}}

/* Left Hero Typography */
.hero-title {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f5f0ea;
    margin: 0;
    line-height: 1.25;
}}

.hero-title .accent-text {{
    color: #c4614a;
}}

.hero-subtitle {{
    color: #a89880;
    font-size: 0.92rem;
    line-height: 1.5;
    margin-top: 8px;
    margin-bottom: 22px;
}}

/* Feature Pills */
.feature-pills-row {{
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
}}

.feature-pill {{
    flex: 1;
    background: rgba(22, 18, 16, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 14px;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease;
}}

.feature-pill:hover {{
    transform: translateY(-2px);
    border-color: rgba(196, 97, 74, 0.45);
}}

.pill-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #f5f0ea;
    margin-bottom: 4px;
}}

.pill-desc {{
    font-size: 0.76rem;
    color: #a89880;
}}

/* Right Dashboard Card */
.dashboard-card-gradio {{
    background: rgba(20, 22, 24, 0.94) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 24px 26px !important;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.55) !important;
    backdrop-filter: blur(16px) !important;
}}

/* Status Badge */
.status-badge {{
    display: flex;
    align-items: center;
    gap: 10px;
    background: #0f2619;
    border: 1px solid #1a5433;
    color: #4ade80;
    border-radius: 30px;
    padding: 9px 18px;
    font-size: 0.88rem;
    font-weight: 500;
    margin-bottom: 16px;
}}

/* Action Button */
button.primary-btn {{
    background: #c4614a !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(196, 97, 74, 0.35) !important;
    transition: all 0.2s ease !important;
    margin-top: 10px !important;
    margin-bottom: 12px !important;
}}

button.primary-btn:hover {{
    background: #d4725b !important;
    box-shadow: 0 8px 24px rgba(196, 97, 74, 0.5) !important;
    transform: translateY(-1px) !important;
}}

/* Result Card */
.result-card {{
    background: #eeddc6;
    border-radius: 14px;
    padding: 18px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    margin-top: 10px;
    margin-bottom: 8px;
}}

.result-left {{
    display: flex;
    align-items: center;
    gap: 18px;
}}

.result-label {{
    font-size: 0.82rem;
    font-weight: 600;
    color: #635649;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 2px;
}}

.result-genre {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #191614;
    line-height: 1;
}}
"""

# ==========================================
# GRADIO APPLICATION
# ==========================================
with gr.Blocks(title="Audio Genre Predictor (AST)", css=custom_css, theme=gr.themes.Base()) as demo:
    with gr.Row(equal_height=False):
        # ----------------------------------
        # LEFT COLUMN: HERO & AUDIO INPUT
        # ----------------------------------
        with gr.Column(scale=11):
            gr.HTML(f"""
            <div style="padding: 0 0 12px 0;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    {SVG_VINYL_HEADER}
                    <h1 class="hero-title">Audio Genre ♫<br>Predictor <span class="accent-text">(AST)</span></h1>
                </div>
                <p class="hero-subtitle">
                    Upload any popular audio format. The AST model will bypass intros, 
                    run Test-Time Augmentation, and predict the genre.
                </p>
            </div>
            <div class="feature-pills-row">
                <div class="feature-pill">
                    <div class="pill-header">
                        {SVG_SOUNDWAVE}
                        <span>AI Powered</span>
                    </div>
                    <div class="pill-desc">Uses AST model</div>
                </div>
                <div class="feature-pill">
                    <div class="pill-header">
                        {SVG_CHART_ICON}
                        <span>Skips Intros</span>
                    </div>
                    <div class="pill-desc">Focuses on music</div>
                </div>
                <div class="feature-pill">
                    <div class="pill-header">
                        {SVG_TAG_ICON}
                        <span>Predicts Genre</span>
                    </div>
                    <div class="pill-desc">Across 10+ genres</div>
                </div>
            </div>
            """)

            audio_input = gr.Audio(
                label="📁 Audio Source (Upload or Record)",
                type="filepath",
                sources=["upload", "microphone"]
            )

        # ----------------------------------
        # RIGHT COLUMN: DASHBOARD & PREDICTION
        # ----------------------------------
        with gr.Column(scale=10, elem_classes=["dashboard-card-gradio"]):
            gr.HTML(f"""
            <div style="display: flex; align-items: center; margin-bottom: 14px;">
                {SVG_VINYL_HEADER}
                <div>
                    <h3 style="margin: 0; font-size: 1.35rem; color: #f5f0ea; font-family: 'Playfair Display', Georgia, serif;">Audio Genre Predictor <span style="color: #c4614a;">(AST)</span> ♫</h3>
                    <div style="font-size: 0.78rem; color: #8e8478; margin-top: 2px;">Audio Spectrogram Transformer Inference Dashboard</div>
                </div>
            </div>
            <div class="status-badge">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
                <span>Model loaded and active (ZeroGPU / Standby).</span>
            </div>
            """)

            predict_btn = gr.Button("Predict Genre 🚀", elem_classes=["primary-btn"], size="lg")

            result_output = gr.HTML(visible=False)
            chart_output = gr.Plot(visible=False)

            def on_predict_click(audio_path):
                res_html, chart_fig = predict_genre(audio_path)
                return (
                    gr.update(value=res_html, visible=True),
                    gr.update(value=chart_fig, visible=True)
                )

            predict_btn.click(
                fn=on_predict_click,
                inputs=audio_input,
                outputs=[result_output, chart_output]
            )

            with gr.Accordion("ℹ️ Architecture & Pipeline Details", open=False):
                gr.Markdown(
                    """
                    - **Backbone:** Fine-tuned `MIT/ast-finetuned-audioset-10-10-0.4593` (Vision Transformer on Mel-spectrograms).
                    - **Skip-Intro TTA:** Takes 3 overlapping temporal crops after intro, averaging predictions.
                    - **Genres Supported:** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.
                    """
                )

# Warm up model in background thread
threading.Thread(target=preload_model, daemon=True).start()

print("🚀 Launching Gradio web server on 0.0.0.0:7860...", flush=True)
demo.launch(server_name="0.0.0.0", server_port=7860)
