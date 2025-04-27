# Running Audio Classification in Docker

This document provides instructions for running the audio classification project in a Docker container.

## Prerequisites

- Docker installed on your system ([Download Docker](https://www.docker.com/products/docker-desktop/))
- Docker Desktop running (for Windows/Mac users)

## Building the Docker Image

1. Navigate to the project directory:
   ```bash
   cd Audio-Classification
   ```

2. Build the Docker image:
   ```bash
   docker build -t audio-classification .
   ```

   This will create a Docker image named 'audio-classification'.

## Running the Container

Run the container with the following command:

```bash
docker run -it --name audio-classifier audio-classification
```

This will:
1. Start a container from the audio-classification image
2. Run the complete workflow (data preparation, training, evaluation)
3. Display the output in the terminal

## Accessing Results

To copy the results from the container to your local machine:

```bash
docker cp audio-classifier:/app/evaluation ./evaluation_results
docker cp audio-classifier:/app/models ./model_results
docker cp audio-classifier:/app/training_history.png ./
```

## Troubleshooting

If you encounter issues with the Docker build or run processes:

1. Make sure Docker Desktop is running
2. Check that you have sufficient disk space
3. Ensure all required files are present in your project directory:
   - instruments.csv
   - All Python scripts (train.py, model.py, etc.)
   - wavfiles directory with audio samples

## Docker Container Structure

The Docker container contains the following structure:

```
/app/
  ├── instruments.csv
  ├── cfg.py
  ├── convert_wav.py
  ├── data_split.py
  ├── model.py
  ├── evaluate.py
  ├── eda.py
  ├── predict.py
  ├── train.py
  ├── wavfiles/
  ├── clean/
  ├── data/
  ├── models/
  ├── evaluation/
  └── pickles/
```

## Running Individual Scripts

If you need to run individual scripts instead of the full workflow:

```bash
# Run only the data preparation
docker run -it audio-classification python data_split.py

# Run only the model training
docker run -it audio-classification python model.py

# Run only the evaluation
docker run -it audio-classification python evaluate.py
``` 