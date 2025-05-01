# Running Audio Classification in Docker

This document provides simple instructions for running the audio classification project in Docker.

## Prerequisites

- Docker installed on your system ([Download Docker](https://www.docker.com/products/docker-desktop/))
- Docker Compose installed (included with Docker Desktop for Windows/Mac)

## Quick Start

### On Windows:

```cmd
run_docker.bat
```

### On Linux/macOS:

```bash
chmod +x run_docker.sh
./run_docker.sh
```

These scripts will:
1. Create necessary directories
2. Build the Docker image
3. Run the container with proper volume mappings
4. Process the audio files and generate visualizations

## Manual Setup

If you prefer to run the commands manually:

```bash
# Build the image
docker-compose build

# Run the container
docker-compose up
```

## Running Individual Scripts

You can run individual scripts using the helper scripts:

### On Windows:

```cmd
run_docker.bat script display_plots.py
run_docker.bat script model.py
run_docker.bat script evaluate.py
```

### On Linux/macOS:

```bash
./run_docker.sh script display_plots.py
./run_docker.sh script model.py
./run_docker.sh script evaluate.py
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

## Cleaning Up

To remove all Docker containers and images created by this project:

### On Windows:

```cmd
run_docker.bat clean
```

### On Linux/macOS:

```bash
./run_docker.sh clean
```

## Troubleshooting

If you encounter issues:

1. Make sure Docker Desktop is running
2. Check that you have sufficient disk space
3. Ensure all required files are present in your project directory
4. Check Docker logs for detailed error messages:
   ```bash
   docker-compose logs
   ```
5. If you get permission errors with the mounted volumes, check your Docker Desktop settings for file sharing permissions
