import streamlit as st
import os
import time
import base64
import textwrap
import torch
import spaces
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from inference import loadAST, run_inference, GENRES

# ==========================================
# PAGE CONFIGURATION (WIDE LAYOUT FOR TWO COLUMNS)
# ==========================================
st.set_page_config(
    page_title="Audio Genre Predictor (AST)", 
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Path must match what Dockerfile extracts and renames
MODEL_PATH = "best-ast-model.ckpt"

# ==========================================
# CACHED MODEL LOADING (UNTOUCHED BACKEND)
# ==========================================
@st.cache_resource
def init_model():
    """Loads the model into CPU memory once at app startup."""
    if not os.path.exists(MODEL_PATH):
        return None
    return loadAST(MODEL_PATH)

# ==========================================
# ZEROGPU INFERENCE WRAPPER (UNTOUCHED BACKEND)
# ==========================================
@spaces.GPU(duration=60)
def gpu_predict(audio_path, loaded_model):
    """
    Dynamically routes to A100 GPU on Hugging Face, 
    or falls back to CPU when testing locally.
    """
    if torch.cuda.is_available():
        target_device = "cuda"
    else:
        target_device = "cpu"
        
    return run_inference(audio_path, loaded_model, device=target_device)

# ==========================================
# ASSET ENCODING & CACHING
# ==========================================
@st.cache_data
def get_background_base64():
    """Encodes the vintage music studio background image to base64."""
    candidate_paths = [
        "assets/bg.jpg",
        "production/assets/bg.jpg",
        "/app/assets/bg.jpg",
        "/home/moli/projects/dlgenai-project-messy-mashup-main/dlgenai-project-messy-mashup-main/production/assets/bg.jpg"
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

SVG_CLOUD_UPLOAD = """<svg width="58" height="46" viewBox="0 0 24 24" fill="none" stroke="#c4614a" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/></svg>"""

SVG_SOUNDWAVE = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#c4614a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 10v4M6 6v12M10 3v18M14 8v8M18 5v14M22 10v4"/></svg>"""

SVG_CHART_ICON = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#c4614a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>"""

SVG_TAG_ICON = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#c4614a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>"""

# ==========================================
# CUSTOM GENRE PROBABILITY CHART GENERATOR
# ==========================================
def render_styled_genre_chart(genres, probabilities, predicted_genre):
    """Generates an aesthetic dark-themed genre distribution chart matching the reference."""
    if plt is not None:
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
    return None

def render_genre_chart_svg(genres, probabilities, predicted_genre):
    """Fallback pure SVG chart requiring zero external chart dependencies."""
    max_val = max(probabilities) if len(probabilities) > 0 else 0.5
    y_max = max(0.5, float(np.ceil(max_val * 10) / 10 + 0.05))
    
    width, height = 640, 260
    pad_left, pad_right, pad_top, pad_bottom = 45, 20, 20, 35
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom
    baseline_y = pad_top + chart_h
    
    svg = [f'<div style="width:100%; overflow-x:auto;"><svg viewBox="0 0 {width} {height}" width="100%" height="{height}" style="background:#16191b; border-radius:10px; font-family:Inter,sans-serif;">']
    
    # Grid lines
    y_ticks = np.arange(0, y_max + 0.01, 0.1)
    for val in y_ticks:
        y_pos = pad_top + chart_h - (val / y_max) * chart_h
        svg.append(f'<line x1="{pad_left}" y1="{y_pos:.1f}" x2="{width - pad_right}" y2="{y_pos:.1f}" stroke="#262c30" stroke-width="0.8"/>')
        svg.append(f'<text x="{pad_left - 8}" y="{y_pos + 3.5:.1f}" fill="#78726b" font-size="10" text-anchor="end">{val:.1f}</text>')
        
    svg.append(f'<line x1="{pad_left}" y1="{baseline_y}" x2="{width - pad_right}" y2="{baseline_y}" stroke="#333a3f" stroke-width="1.2"/>')
    
    # Bars
    n = len(genres)
    slot_w = chart_w / n
    bar_w = slot_w * 0.62
    
    for i, (g, p) in enumerate(zip(genres, probabilities)):
        bar_h = (p / y_max) * chart_h
        bar_x = pad_left + i * slot_w + (slot_w - bar_w) / 2
        bar_y = baseline_y - bar_h
        color = "#c2654f" if g.lower() == predicted_genre.lower() else "#6e8974"
        svg.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2" fill="{color}"/>')
        label_x = pad_left + i * slot_w + slot_w / 2
        svg.append(f'<text x="{label_x:.1f}" y="{baseline_y + 18}" fill="#9e988f" font-size="10.5" text-anchor="middle">{g}</text>')
        
    svg.append('</svg></div>')
    return "".join(svg)

# ==========================================
# GLOBAL THEME & STYLESHEET (CSS INJECTION)
# ==========================================
bg_css = f"""
    background-image: linear-gradient(rgba(10, 8, 7, 0.68), rgba(10, 8, 7, 0.82)), url('data:image/jpeg;base64,{BG_BASE64}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
""" if BG_BASE64 else "background-color: #0d0b0a;"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

/* Reset and Streamlit chrome removal */
header, footer, #MainMenu {{
    visibility: hidden !important;
    height: 0 !important;
}}

.stApp {{
    {bg_css}
    color: #f3ede2;
    font-family: 'Inter', -apple-system, sans-serif;
}}

.block-container {{
    max-width: 1400px !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

/* Typography */
h1, h2, h3, .serif-font {{
    font-family: 'Playfair Display', Georgia, serif;
}}

/* Left Panel Hero Container */
.hero-container {{
    padding: 0 0 16px 0;
}}

.hero-title {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: #f5f0ea;
    margin: 0;
    line-height: 1.25;
    letter-spacing: -0.5px;
}}

.hero-title .accent-text {{
    color: #c4614a;
}}

.hero-subtitle {{
    color: #a89880;
    font-size: 0.92rem;
    line-height: 1.5;
    margin-top: 8px;
    margin-bottom: 24px;
}}

/* Feature Pills Row */
.feature-pills-row {{
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
}}

.feature-pill {{
    flex: 1;
    background: rgba(22, 18, 16, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 14px;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
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

/* Custom Upload Card / Dropzone */
.dropzone-container {{
    background: rgba(20, 16, 14, 0.85);
    border: 2px dashed rgba(196, 97, 74, 0.6);
    border-radius: 18px;
    padding: 28px 24px 22px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    margin-bottom: 16px;
}}

.dropzone-title {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f5f0ea;
    margin: 6px 0 3px 0;
}}

.dropzone-sub {{
    color: #c4614a;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 10px;
}}

.dropzone-formats {{
    font-size: 0.75rem;
    color: #928475;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 2px;
}}

.dropzone-limit {{
    font-size: 0.72rem;
    color: #6e6459;
    margin-bottom: 4px;
}}

/* Right Panel Styling (applied directly to right column vertical block) */
div[data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {{
    background: rgba(20, 22, 24, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 26px 28px 26px 28px;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(16px);
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

/* File Info Card */
.file-card {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #1c2226;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 14px;
}}

.file-card-left {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.file-icon-box {{
    width: 38px;
    height: 38px;
    border-radius: 8px;
    background: rgba(96, 180, 255, 0.12);
    display: flex;
    align-items: center;
    justify-content: center;
}}

.file-name {{
    font-weight: 600;
    font-size: 0.92rem;
    color: #f5f0ea;
}}

.file-meta {{
    font-size: 0.76rem;
    color: #8c8276;
}}

/* Styled Action Button */
.stButton > button {{
    width: 100% !important;
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
    letter-spacing: 0.3px !important;
}}

.stButton > button:hover {{
    background: #d4725b !important;
    box-shadow: 0 8px 24px rgba(196, 97, 74, 0.5) !important;
    transform: translateY(-1px) !important;
}}

.stButton > button:active {{
    transform: translateY(1px) !important;
}}

/* Result Card (Parchment Vintage) */
.result-card {{
    background: #eeddc6;
    border-radius: 14px;
    padding: 18px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    margin-top: 18px;
    margin-bottom: 10px;
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

/* Loading Banner */
.loading-badge {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: #142738;
    border: 1px solid #1f4260;
    color: #8ac0e8;
    border-radius: 30px;
    padding: 10px 20px;
    font-size: 0.9rem;
    margin-top: 14px;
}}

.spinner-icon {{
    animation: spin 1s linear infinite;
    display: inline-block;
}}

@keyframes spin {{
    0% {{ transform: rotate(0deg); }}
    100% {{ transform: rotate(360deg); }}
}}

/* Custom Tabs Styling */
.stTabs [data-baseweb="tab-list"] {{
    background-color: rgba(22, 18, 16, 0.7);
    border-radius: 12px;
    padding: 4px;
    gap: 6px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important;
    color: #a89880 !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}}

.stTabs [aria-selected="true"] {{
    background-color: #c4614a !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}}

/* Streamlit Audio Player styling */
audio {{
    width: 100% !important;
    border-radius: 10px !important;
    margin-bottom: 16px !important;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# BOOT UP THE MODEL
# ==========================================
model = init_model()
if model is None:
    st.error(f"❌ Model weights not found at `{MODEL_PATH}`. Please upload the .ckpt file.")
    st.stop()

# ==========================================
# APPLICATION LAYOUT (TWO COLUMNS)
# ==========================================
col_hero, col_dash = st.columns([1.1, 1.0], gap="large")

# ------------------------------------------
# LEFT COLUMN: HERO & INPUT PANEL
# ------------------------------------------
with col_hero:
    # 1. Hero Header
    st.markdown(textwrap.dedent(f"""
    <div class="hero-container">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            {SVG_VINYL_HEADER}
            <h1 class="hero-title">Audio Genre ♫<br>Predictor <span class="accent-text">(AST)</span></h1>
        </div>
        <p class="hero-subtitle">
            Upload any popular audio format. The AST model will bypass intros, 
            run Test-Time Augmentation, and predict the genre.
        </p>
    </div>
    """).strip(), unsafe_allow_html=True)

    # 2. Feature Pills Row
    st.markdown(textwrap.dedent(f"""
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
    """).strip(), unsafe_allow_html=True)

    # 3. Audio Input Section (Upload or Record Tabs)
    tab_upload, tab_record = st.tabs(["📁 Upload Audio File", "🎙️ Record Audio"])

    with tab_upload:
        st.markdown(textwrap.dedent(f"""
        <div class="dropzone-container">
            {SVG_CLOUD_UPLOAD}
            <div class="dropzone-title">Drag & drop your audio file here</div>
            <div class="dropzone-sub">or click browse below</div>
            <div class="dropzone-formats">WAV, MP3, M4A, AAC, OGG, FLAC</div>
            <div class="dropzone-limit">Max 200MB per file</div>
        </div>
        """).strip(), unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose Audio File", 
            type=['wav', 'mp3', 'm4a', 'aac', 'ogg', 'flac'],
            label_visibility="collapsed"
        )

    with tab_record:
        st.markdown(textwrap.dedent("""
        <div class="dropzone-container" style="padding: 24px;">
            <div class="dropzone-title" style="margin-bottom: 10px;">Microphone Recording</div>
            <div class="dropzone-formats" style="margin-bottom: 12px;">Speak or play music directly into your mic</div>
        </div>
        """).strip(), unsafe_allow_html=True)
        recorded_audio = st.audio_input("Record from microphone", label_visibility="collapsed")

# ------------------------------------------
# RIGHT COLUMN: DASHBOARD & PREDICTION
# ------------------------------------------
with col_dash:
    # Header & Status Badge inside right column
    st.markdown(textwrap.dedent(f"""
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
        <span>Model loaded and active (CPU Standby).</span>
    </div>
    """).strip(), unsafe_allow_html=True)

    # Determine Active Audio Source
    active_audio = recorded_audio if recorded_audio is not None else uploaded_file

    if active_audio is not None:
        is_recorded = (active_audio == recorded_audio)
        file_name = "recorded_audio.wav" if is_recorded else active_audio.name
        
        # Calculate file size
        file_size_mb = 0.0
        if hasattr(active_audio, "size") and active_audio.size is not None:
            file_size_mb = active_audio.size / (1024 * 1024)
        elif hasattr(active_audio, "getbuffer"):
            file_size_mb = len(active_audio.getbuffer()) / (1024 * 1024)
            
        file_ext = file_name.rsplit(".", 1)[-1].upper() if "." in file_name else "WAV"
        
        # File Info Card
        st.markdown(textwrap.dedent(f"""
        <div class="file-card">
            <div class="file-card-left">
                <div class="file-icon-box">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#60b4ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M9 18V5l12-2v13"/>
                        <circle cx="6" cy="18" r="3" fill="#60b4ff"/>
                        <circle cx="18" cy="16" r="3" fill="#60b4ff"/>
                    </svg>
                </div>
                <div>
                    <div class="file-name">{file_name}</div>
                    <div class="file-meta">{file_size_mb:.1f} MB • {file_ext}</div>
                </div>
            </div>
            <div style="color: #6e6459; font-size: 1.1rem;">✕</div>
        </div>
        """).strip(), unsafe_allow_html=True)

        # Audio Player
        st.audio(active_audio)

        # Predict Button
        predict_clicked = st.button("Predict Genre 🚀", use_container_width=True)

        if predict_clicked:
            # File handling
            temp_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "wav"
            temp_path = f"/tmp/temp_audio_{int(time.time())}.{temp_ext}"
            
            with open(temp_path, "wb") as f:
                if hasattr(active_audio, "getbuffer"):
                    f.write(active_audio.getbuffer())
                else:
                    f.write(active_audio.read())

            # Loading Banner
            loading_placeholder = st.empty()
            loading_placeholder.markdown(textwrap.dedent("""
            <div class="loading-badge">
                <span class="spinner-icon">⭕</span>
                <span>Requesting compute and analyzing audio...</span>
            </div>
            """).strip(), unsafe_allow_html=True)
            
            start_time = time.time()

            try:
                # Trigger GPU/CPU inference
                genre, probabilities = gpu_predict(temp_path, model)
                elapsed = time.time() - start_time
                
                # Clear loading banner
                loading_placeholder.empty()

                # Display Result Card matching reference design
                st.markdown(textwrap.dedent(f"""
                <div class="result-card">
                    <div class="result-left">
                        {SVG_VINYL_RESULT}
                        <div>
                            <div class="result-label">Predicted Genre</div>
                            <div class="result-genre">{genre.upper()}</div>
                        </div>
                    </div>
                    {SVG_SINE_WAVE}
                </div>
                <div style="font-size: 0.82rem; color: #9c9285; margin-bottom: 18px; display: flex; align-items: center; gap: 6px;">
                    <span>⏱️</span>
                    <span>Total prediction time: <b>{elapsed:.2f} seconds</b>.</span>
                </div>
                """).strip(), unsafe_allow_html=True)

                # Genre Probabilities Header
                st.markdown(textwrap.dedent("""
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 14px; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem;">📊</span>
                    <h4 style="margin: 0; font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 600; color: #e8e2d8;">Genre Probabilities</h4>
                </div>
                """).strip(), unsafe_allow_html=True)

                # Render Styled Genre Bar Chart (Matplotlib or Vector SVG)
                chart_fig = render_styled_genre_chart(GENRES, probabilities, genre)
                if chart_fig is not None:
                    st.pyplot(chart_fig, use_container_width=True)
                else:
                    st.markdown(render_genre_chart_svg(GENRES, probabilities, genre), unsafe_allow_html=True)

            except Exception as e:
                loading_placeholder.empty()
                st.error(f"Prediction failed: {e}")
                st.markdown("*Tip: Ensure the file is a valid audio format and is not corrupted.*")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    else:
        st.markdown(textwrap.dedent("""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 36px 20px; text-align: center; color: #8e8478; margin-top: 10px;">
            <div style="font-size: 1.6rem; margin-bottom: 8px;">🎵</div>
            <div style="font-weight: 500; font-size: 0.95rem; color: #c4baa9; margin-bottom: 4px;">No audio loaded yet</div>
            <div style="font-size: 0.82rem;">Upload a song or record via microphone on the left to begin analysis.</div>
        </div>
        """).strip(), unsafe_allow_html=True)