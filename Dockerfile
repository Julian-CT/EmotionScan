# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Git and Git LFS (needed to download LFS files)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app/ .

# Copy .pkl files and src directories to temp location
# We'll copy them to the volume at runtime since volumes overlay /app/models
RUN mkdir -p /tmp/models/mnb && \
    mkdir -p /tmp/models/bertimbau-mlp-ai && \
    mkdir -p /tmp/models/bertimbau-mlp-sentiment && \
    if [ -d "models/mnb" ]; then \
        cp -r models/mnb/* /tmp/models/mnb/ 2>/dev/null || true; \
    fi && \
    if [ -d "models/bertimbau-mlp-ai/src" ]; then \
        cp -r models/bertimbau-mlp-ai/src /tmp/models/bertimbau-mlp-ai/ 2>/dev/null || true; \
    fi && \
    if [ -d "models/bertimbau-mlp-sentiment/src" ]; then \
        cp -r models/bertimbau-mlp-sentiment/src /tmp/models/bertimbau-mlp-sentiment/ 2>/dev/null || true; \
    fi


# Expose port (Railway will set PORT env var)
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]

