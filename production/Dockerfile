# Use official Python slim image for a lightweight base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# 1. Install System Dependencies (ffmpeg for librosa)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Install UV (Astral's ultra-fast package manager)
RUN pip install uv

# 3. Copy requirements and install dependencies instantly using UV
COPY requirements.txt .
# --system flag installs directly to the container's python environment
RUN uv pip install --system --no-cache -r requirements.txt

# 4. Model checkpoint is mounted at runtime as an external volume
# Example: docker run -v ./best-ast-model.ckpt:/app/best-ast-model.ckpt:ro ...
VOLUME ["/app/best-ast-model.ckpt"]

# 5. Copy your application code (app.py, inference.py, etc.)
COPY . .

# 6. Expose the standard Hugging Face Space port
EXPOSE 7860

# 7. Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]