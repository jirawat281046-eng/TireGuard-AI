# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TF_ENABLE_ONEDNN_OPTS 0
ENV CUDA_VISIBLE_DEVICES -1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Flask and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the uploads directory exists and is writable
RUN mkdir -p uploads && chmod 777 uploads

# Create a non-root user and switch to it for security (Optional but good for HF)
RUN useradd -m myuser
USER myuser

# Expose the port the app runs on (Hugging Face standard)
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
