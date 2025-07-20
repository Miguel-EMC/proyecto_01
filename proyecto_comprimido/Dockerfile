FROM python:3.9-slim

# Install system dependencies for espeak-ng and audio processing
RUN apt-get update && apt-get install -y \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng-dev \
    sox \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Default command
CMD ["python", "transcription_alignment.py"]
