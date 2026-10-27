# Use official Python slim image for a lightweight base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# 1. Install System Dependencies (ffmpeg for librosa, curl for downloads)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    tar \
    && rm -rf /var/lib/apt/lists/*

# 2. Install UV (Astral's ultra-fast package manager)
RUN pip install uv

# 3. Copy requirements and install dependencies instantly using UV
COPY requirements.txt .
# --system flag installs directly to the container's python environment
RUN uv pip install --system --no-cache -r requirements.txt

# 4. Download and Extract the Model
# We use --fail to stop the build if the link is dead, and --retry for stability
# 4. Download, Extract, Relocate, and Clean Up
RUN echo "Downloading AST Model from Kaggle..." \
    && curl -L --fail --retry 3 -o model.tar.gz "https://www.kaggle.com/api/v1/models/harshitsahu01/ast-psuedo-tuned-v3/pyTorch/v3/4/download" \
    && echo "Download complete. Extracting archive..." \
    && tar -xzvf model.tar.gz \
    && rm model.tar.gz \
    && echo "Moving weights to root directory..." \
    && mv checkpoints_v3/best-ast-v3.ckpt ./best-ast-model.ckpt \
    && echo "Cleaning up unnecessary Kaggle output files..." \
    && rm -rf checkpoints_v3 wandb __huggingface_repos__* submission.csv \
    && echo "✅ Model setup successful!"

# 5. Copy your application code (app.py, inference.py, etc.)
COPY . .

# 6. Expose the standard Hugging Face Space port
EXPOSE 7860

# 7. Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]