# Use TensorFlow's official base image (CPU version)
FROM tensorflow/tensorflow:2.3.0

# Set the working directory
WORKDIR /app

# Install system dependencies for audio processing and visualization
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-liberation \
    python3-dev \
    build-essential \
    pkg-config \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies with optimizations
# We're using a specific order to handle dependencies correctly
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.18.5 && \
    pip install --no-cache-dir h5py==2.10.0 && \
    pip install --no-cache-dir scipy==1.4.1 && \
    pip install --no-cache-dir pandas==1.1.5 && \
    pip install --no-cache-dir scikit-learn==0.24.2 && \
    pip install --no-cache-dir matplotlib==3.3.4 seaborn==0.11.2 && \
    pip install --no-cache-dir Pillow>=8.0.0 && \
    pip install --no-cache-dir tqdm>=4.0 && \
    pip install --no-cache-dir librosa==0.8.1 && \
    pip install --no-cache-dir python-speech-features==0.6 && \
    pip install --no-cache-dir keras==2.4.3

# Configure matplotlib for high-quality non-interactive plots
ENV MPLBACKEND=Agg
RUN mkdir -p /root/.config/matplotlib && \
    echo "backend: Agg" > /root/.config/matplotlib/matplotlibrc && \
    echo "figure.figsize: 12, 8" >> /root/.config/matplotlib/matplotlibrc && \
    echo "figure.dpi: 300" >> /root/.config/matplotlib/matplotlibrc && \
    echo "savefig.dpi: 300" >> /root/.config/matplotlib/matplotlibrc && \
    echo "font.size: 12" >> /root/.config/matplotlib/matplotlibrc && \
    echo "axes.grid: True" >> /root/.config/matplotlib/matplotlibrc && \
    echo "grid.alpha: 0.3" >> /root/.config/matplotlib/matplotlibrc

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Create necessary directories with proper permissions
RUN mkdir -p /app/clean /app/data /app/models /app/evaluation /app/plots && \
    chmod -R 777 /app

# Copy Python scripts (using specific order for better layer caching)
COPY cfg.py convert_wav.py data_split.py eda.py /app/
COPY model.py evaluate.py predict.py train.py /app/
COPY display_plots.py run_and_display.py /app/

# Copy data files
COPY instruments.csv /app/
COPY wavfiles /app/wavfiles

# Create volumes for persistent data
VOLUME ["/app/evaluation", "/app/models", "/app/clean", "/app/plots"]

# Set the entrypoint to allow for different commands
ENTRYPOINT ["python"]

# Default command
CMD ["run_and_display.py"]

# Add labels for better documentation
LABEL maintainer="Audio Classification Project" \
      description="Docker image for audio classification with TensorFlow (CPU version)" \
      version="1.0"
