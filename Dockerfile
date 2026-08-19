FROM python:3.10

# Install the missing OpenGL library (and clean up)
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV OPENCV_OPENCL_RUNTIME=

CMD ["gunicorn", "-w", "1", "--threads", "100", "--bind", "0.0.0.0:5000", "app:app"]
