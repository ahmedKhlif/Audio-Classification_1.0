# Audio Classification Project

A deep learning project for classifying audio samples into multiple categories (Swords, WildAnimals, Alarms) using Convolutional Neural Networks and MFCC features.

## Project Overview

This project implements a complete machine learning pipeline that:
1. Processes audio files (.wav format)
2. Splits the dataset into train/validation/test sets
3. Extracts MFCC (Mel-frequency cepstral coefficients) features
4. Trains a CNN model for classification
5. Evaluates the model with comprehensive metrics
6. Provides prediction capabilities for new audio samples

## Requirements

- Python 3.6+
- TensorFlow 2.3.0
- librosa
- scikit-learn
- pandas
- numpy
- matplotlib
- python_speech_features
- seaborn (for evaluation visualizations)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Audio-Classification
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv myvenv
myvenv\Scripts\activate

# Linux/Mac
python -m venv myvenv
source myvenv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Quick Start

To run the complete workflow (data preparation, training, and evaluation):

```bash
python train.py
```

This will:
1. Split your data into train/validation/test sets
2. Train the model on the training set
3. Validate the model on the validation set
4. Evaluate the model on the test set
5. Generate performance metrics and visualizations

## Dataset

The project uses a custom dataset of audio files divided into three classes:
- Swords: Sword sound effects
- WildAnimals: Animal sounds
- Alarms: Alarm sound effects

The dataset information is stored in `instruments.csv`.

## Detailed Usage

### 1. Data Preparation

Before training, you need to convert and split your data:

a. Convert WAV files to the correct format:
```bash
python convert_wav.py
```

b. Split the dataset into train/validation/test sets:
```bash
python data_split.py
```

This will:
- Create train/validation/test splits with balanced class distributions
- Save splits to CSV files in the `data` directory
- Copy audio files to train/val/test directories

### 2. Exploratory Data Analysis

To analyze the dataset and generate visualization plots:
```bash
python eda.py
```

This generates visualizations of:
- Class distribution
- Audio signals
- Fourier transforms
- Filter banks
- MFCC features

### 3. Model Training

To train the model:
```bash
python model.py
```

This will:
- Check if data has been split (and run data_split.py if needed)
- Extract features from the training and validation sets
- Train the CNN model
- Save the trained model to the `models` directory
- Generate and save training history plots
- Automatically run evaluation

### 4. Model Evaluation

The evaluation is automatically run after training, but you can also run it separately:
```bash
python evaluate.py
```

This generates:
- Accuracy, precision, recall, and F1-score metrics
- Confusion matrix visualization
- ROC curves with AUC values
- Precision-recall curves
- A comprehensive classification report

All evaluation results are saved to the `evaluation` directory.

### 5. Making Predictions

To make predictions on new audio files:
```bash
python predict.py
```

The predictions will be saved to `predictions.csv`.

## Project Structure

- `cfg.py`: Configuration file with model parameters
- `convert_wav.py`: Script to convert audio files to the correct format
- `data_split.py`: Script to split data into train/validation/test sets
- `eda.py`: Exploratory data analysis script
- `model.py`: Model training script with integrated data preparation and evaluation
- `evaluate.py`: Comprehensive model evaluation script
- `predict.py`: Model prediction script
- `train.py`: Master script that runs the entire workflow
- `clean/`: Directory containing processed audio files
- `wavfiles/`: Directory containing raw audio files
- `data/`: Directory containing train/validation/test splits
- `models/`: Directory containing saved models
- `pickles/`: Directory containing serialized data
- `evaluation/`: Directory containing evaluation results and visualizations

## Output Files

The project generates several output files:
- `training_class_distribution.png`: Pie chart of class distribution
- `training_history.png`: Training and validation accuracy/loss curves
- `training_history.csv`: CSV file with training metrics
- `evaluation/confusion_matrix.png`: Confusion matrix visualization
- `evaluation/roc_curves.png`: ROC curves for each class
- `evaluation/precision_recall_curves.png`: Precision-recall curves
- `evaluation/metrics.csv`: Comprehensive evaluation metrics

## Docker

A Dockerfile is provided for containerization:
```bash
docker build -t audio-classification .
docker run -it audio-classification
```

## License

[Your License Here] 