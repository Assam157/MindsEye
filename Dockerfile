# Use a slim Python image
FROM python:3.10-slim

# Install system libraries needed by MediaPipe and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgles2-mesa \          # <-- this provides libGLESv2.so.2
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Render expects (default 10000, but we'll use 5000 and let Render map)
EXPOSE 5000

# Start Gunicorn with the same command you used before
CMD ["gunicorn", "-w", "1", "--threads", "100", "app:app"]
