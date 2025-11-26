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

# Copy .pkl files to a temp location (they're not in Git LFS, so they're in regular Git)
# We'll copy them to the volume at runtime since volumes overlay /app/models
RUN mkdir -p /tmp/models/mnb && \
    if [ -d "models/mnb" ]; then \
        cp -r models/mnb/* /tmp/models/mnb/ 2>/dev/null || true; \
    fi

# Copy .git directory and .gitattributes (needed for Git LFS)
# Note: Railway volumes are mounted at runtime, not during build
# So we'll download models to the volume at runtime instead
COPY .git/ .git/
COPY .gitattributes .gitattributes

# Note: We don't run git lfs pull here because:
# 1. Volumes are only mounted at runtime, not during build
# 2. Models will be downloaded to the volume when the app starts
# 3. This avoids authentication issues during build

# Expose port (Railway will set PORT env var)
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]

