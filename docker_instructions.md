# Running Audio Classification in Docker

This document provides comprehensive instructions for running the audio classification project in Docker containers, with support for both GPU and CPU environments.

## Prerequisites

- Docker installed on your system ([Download Docker](https://www.docker.com/products/docker-desktop/))
- Docker Compose installed (included with Docker Desktop for Windows/Mac)
- For GPU support:
  - NVIDIA GPU with CUDA support
  - [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed
  - NVIDIA drivers installed

## Quick Start (Recommended)

The easiest way to run the project is using the provided helper scripts that automatically detect if you have GPU support:

### On Linux/macOS:

```bash
# Make the script executable
chmod +x run_docker.sh

# Run the project
./run_docker.sh
```

### On Windows:

```cmd
run_docker.bat
```

These scripts will:

1. Check if you have GPU support
2. Create necessary directories
3. Build the appropriate Docker image (GPU or CPU)
4. Run the container with proper volume mappings
5. Process the audio files and generate visualizations

## Manual Setup

### Option 1: Using Docker Compose with GPU Support

If you have an NVIDIA GPU with proper drivers installed:

```bash
# Build the image
docker-compose build

# Run the container
docker-compose up
```

### Option 2: Using Docker Compose with CPU Only

If you don't have GPU support:

```bash
# Build the image
docker-compose -f docker-compose.cpu.yml build

# Run the container
docker-compose -f docker-compose.cpu.yml up
```

### Option 3: Using Docker Directly

#### With GPU:

```bash
# Build the image
docker build -t audio-classification .

# Run the container
docker run -it --gpus all --name audio-classifier \
  -v ./evaluation:/app/evaluation \
  -v ./models:/app/models \
  -v ./clean:/app/clean \
  -v ./plots:/app/plots \
  -v ./output:/app/output \
  audio-classification
```

#### CPU Only:

```bash
# Build the image
docker build -t audio-classification-cpu -f Dockerfile.cpu .

# Run the container
docker run -it --name audio-classifier \
  -v ./evaluation:/app/evaluation \
  -v ./models:/app/models \
  -v ./clean:/app/clean \
  -v ./plots:/app/plots \
  -v ./output:/app/output \
  audio-classification-cpu
```

## Running Individual Scripts

You can run individual scripts using the helper scripts:

### On Linux/macOS:

```bash
./run_docker.sh script display_plots.py
./run_docker.sh script model.py
./run_docker.sh script evaluate.py
```

### On Windows:

```cmd
run_docker.bat script display_plots.py
run_docker.bat script model.py
run_docker.bat script evaluate.py
```

Or manually with Docker Compose:

```bash
# With GPU
docker-compose run --rm audio-tools display_plots.py

# CPU only
docker-compose -f docker-compose.cpu.yml run --rm audio-tools display_plots.py
```

## Viewing Results

The results will be automatically saved to the following directories on your host machine:

- `./evaluation/`: Contains evaluation metrics and confusion matrix
- `./plots/`: Contains visualizations (spectrograms, class distribution, etc.)
- `./models/`: Contains the trained model
- `./clean/`: Contains processed audio files
- `./output/`: Contains additional output files

To view all plots in a web browser, open:

```
./plots/index.html
```

## Available Scripts

The Docker container includes several scripts:

- `run_and_display.py`: Runs the complete workflow and generates visualizations
- `display_plots.py`: Generates and saves plots with detailed explanations
- `show_plots.py`: Displays existing plot images
- `model.py`: Trains the audio classification model
- `evaluate.py`: Evaluates the model and generates metrics
- `data_split.py`: Splits the dataset into train/validation/test sets
- `eda.py`: Performs exploratory data analysis
- `convert_wav.py`: Converts audio files to a standard format
- `predict.py`: Makes predictions on new audio files

## Performance Considerations

### GPU vs CPU

- The GPU version is significantly faster for training the model (typically 5-10x speedup)
- The CPU version works on any machine but will be slower for training
- For just generating plots or evaluating a pre-trained model, the CPU version is sufficient

### Memory Requirements

- The project requires at least 4GB of RAM
- For GPU training, you'll need a GPU with at least 4GB of VRAM
- Docker container disk usage is approximately 5-6GB

## Cleaning Up

To remove all Docker containers and images created by this project:

### On Linux/macOS:

```bash
./run_docker.sh clean
```

### On Windows:

```cmd
run_docker.bat clean
```

Or manually:

```bash
docker-compose down
docker-compose -f docker-compose.cpu.yml down
docker rmi audio-classification:latest audio-classification-cpu:latest
```

## Troubleshooting

### Common Issues

1. **GPU not detected**:

   - Ensure NVIDIA drivers are installed: `nvidia-smi`
   - Ensure NVIDIA Container Toolkit is installed: `docker info | grep -i nvidia`
   - Try running with CPU version if GPU support isn't available

2. **Permission errors with mounted volumes**:

   - Ensure the directories exist on your host machine
   - Check Docker Desktop settings for file sharing permissions
   - Try running Docker with elevated privileges

3. **Out of memory errors**:

   - Increase Docker's memory allocation in Docker Desktop settings
   - For GPU version, ensure your GPU has sufficient memory
   - Try using the CPU version with smaller batch sizes

4. **Container exits immediately**:
   - Check logs: `docker-compose logs`
   - Ensure all required files are present
   - Try running an individual script to isolate the issue

### Getting Help

If you encounter persistent issues:

1. Check the Docker logs: `docker-compose logs`
2. Run a specific script to isolate the problem
3. Check if the required directories exist and have proper permissions
4. Verify your Docker and NVIDIA driver installations
