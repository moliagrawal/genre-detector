import streamlit as st
import os
import time
import torch
import spaces
from inference import loadAST, run_inference, GENRES

# ==========================================
# SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="AST Genre Predictor", page_icon="🎵")

# Path must match what the Dockerfile extracts and renames
MODEL_PATH = "best-ast-model.ckpt"

# ==========================================
# CACHED MODEL LOADING
# ==========================================
@st.cache_resource
def init_model():
    """Loads the model into CPU memory once at app startup."""
    if not os.path.exists(MODEL_PATH):
        return None
    return loadAST(MODEL_PATH)

# ==========================================
# ZEROGPU INFERENCE WRAPPER
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
# UI DASHBOARD
# ==========================================
st.title("🎵 Audio Genre Predictor (AST)")
st.markdown("Upload any popular audio format. The AST will bypass intros, run Test-Time Augmentation, and predict the genre.")

# Boot up the model
model = init_model()
if model is None:
    st.error(f"❌ Model weights not found at `{MODEL_PATH}`. Please upload the .ckpt file.")
    st.stop()
else:
    st.success("✅ Model loaded and active (CPU Standby).")

# File Upload
uploaded_file = st.file_uploader(
    "Upload Audio (wav, mp3, m4a, aac, flac)", 
    type=['wav', 'mp3', 'm4a', 'aac', 'ogg', 'flac']
)

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Predict Genre 🚀", use_container_width=True):
        
        # Security/Permission fix: Route temp files directly to Linux /tmp/
        temp_path = f"/tmp/temp_audio_{uploaded_file.name}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("🕒 Requesting compute and analyzing audio...")
        start_time = time.time()

        try:
            # Trigger the GPU/CPU inference engine
            genre, probabilities = gpu_predict(temp_path, model)
            elapsed = time.time() - start_time
            
            # Display Results
            st.success(f"### Predicted Genre: **{genre.upper()}**")
            st.caption(f"⏱️ Total prediction time: {elapsed:.2f} seconds.")
            
            # Interactive Bar Chart
            st.bar_chart(dict(zip(GENRES, probabilities)))
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.markdown("*Tip: Ensure the file is a valid audio format and is not corrupted.*")
            
        finally:
            # Clean up the temp file to prevent container bloat
            if os.path.exists(temp_path):
                os.remove(temp_path)