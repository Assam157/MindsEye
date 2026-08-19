 FROM python:3.10

# Install all OpenGL, EGL, GLES, and other system libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libegl1 \
    libgles2 \
    libopengl0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Disable GPU acceleration and OpenCL
ENV OPENCV_OPENCL_RUNTIME=
ENV MEDIAPIPE_DISABLE_GPU=1

CMD ["gunicorn", "-w", "1", "--threads", "100", "--bind", "0.0.0.0:5000", "app:app"]
