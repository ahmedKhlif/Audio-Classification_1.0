# Use a base image with Python 3.8
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for librosa and soundfile
RUN apt-get update && \
    apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/clean /app/data /app/models /app/evaluation /app/pickles

# Copy the essential files first
COPY instruments.csv /app/
COPY cfg.py /app/
COPY convert_wav.py /app/
COPY data_split.py /app/
COPY model.py /app/
COPY evaluate.py /app/
COPY eda.py /app/
COPY predict.py /app/
COPY train.py /app/

# Copy the wavfiles directory
COPY wavfiles /app/wavfiles

# Set environment variable to disable interactive matplotlib
ENV MPLBACKEND=Agg

# Command to run the integrated workflow
CMD ["python", "train.py"]