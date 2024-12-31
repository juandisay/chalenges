FROM python:3.8-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy serve code
COPY model_serve /app/model_serve

# Copy models directory
COPY models/ /app/models/

# Install dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e model_serve && \
    pip install --no-cache-dir -e models

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["python", "-m", "neural_api"]