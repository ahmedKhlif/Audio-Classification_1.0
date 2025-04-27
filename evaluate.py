import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from python_speech_features import mfcc
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm
import itertools

def load_model_and_config():
    """
    Load the trained model and its configuration.
    
    Returns:
        model: The trained tensorflow model
        config: Configuration object from pickle
    """
    # Load configuration
    pickle_path = os.path.join('pickles', 'conv.p')
    with open(pickle_path, 'rb') as handle:
        config = pickle.load(handle)
    
    # Load model
    model = load_model(config.model_path)
    
    return model, config

def extract_features(audio_file, config):
    """
    Extract MFCC features from an audio file.
    
    Args:
        audio_file: Path to the audio file
        config: Configuration object
    
    Returns:
        features: Normalized MFCC features
    """
    sample_rate, audio_waveform = wavfile.read(audio_file)
    features_list = []
    
    # Process audio in chunks
    for start_idx in range(0, audio_waveform.shape[0] - config.step, config.step):
        audio_sample = audio_waveform[start_idx:start_idx + config.step]
        mfcc_features = mfcc(
            audio_sample, 
            sample_rate, 
            numcep=config.nfeat, 
            nfilt=config.nfilt, 
            nfft=config.nfft
        )
        
        # Normalize features
        normalized_features = (mfcc_features - config.min) / (config.max - config.min)
        
        if config.mode == 'conv':
            normalized_features = normalized_features.reshape(
                1, normalized_features.shape[0], normalized_features.shape[1], 1
            )
        
        features_list.append(normalized_features)
    
    return features_list

def predict_file(model, features_list):
    """
    Make predictions for an audio file.
    
    Args:
        model: The trained model
        features_list: List of feature arrays
    
    Returns:
        predictions: Array of class probabilities
    """
    predictions = [model.predict(features) for features in features_list]
    mean_prediction = np.mean(predictions, axis=0)
    return mean_prediction

def evaluate_model(test_csv='data/test_data.csv', audio_dir='clean'):
    """
    Evaluate the model using various metrics.
    
    Args:
        test_csv: Path to the test data CSV
        audio_dir: Directory containing audio files
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("Loading model and data...")
    model, config = load_model_and_config()
    test_data = pd.read_csv(test_csv)
    class_labels = list(np.unique(test_data.label))
    
    y_true = []  # True labels
    y_pred = []  # Predicted labels
    y_proba = []  # Predicted probabilities
    
    print("Extracting features and making predictions...")
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        file_name = row['fname']
        file_path = os.path.join(audio_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue
        
        # Extract features and predict
        features_list = extract_features(file_path, config)
        if not features_list:
            print(f"Warning: Could not extract features from {file_path}. Skipping.")
            continue
        
        # Get predictions
        file_proba = predict_file(model, features_list)
        file_pred = np.argmax(file_proba)
        
        # Get true label
        true_label = class_labels.index(row['label'])
        
        y_true.append(true_label)
        y_pred.append(file_pred)
        y_proba.append(file_proba.flatten())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    # Calculate metrics
    print("Calculating metrics...")
    acc = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Compile metrics
    metrics = {
        'accuracy': acc,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
    }
    
    # Print metrics
    print("\n===== Model Evaluation =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (micro): {precision_micro:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (micro): {recall_micro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print("\n===== Classification Report =====")
    print(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_labels)
    
    # Plot ROC curves if we have more than 2 classes
    if len(class_labels) > 2:
        plot_roc_curves(y_true, y_proba, class_labels)
        plot_precision_recall_curves(y_true, y_proba, class_labels)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_labels):
    """
    Plot and save the confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('evaluation/confusion_matrix.png')
    print("Confusion matrix saved to 'evaluation/confusion_matrix.png'")
    plt.close()

def plot_roc_curves(y_true, y_proba, class_labels):
    """
    Plot and save ROC curves for each class.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_labels: List of class names
    """
    # Convert true labels to one-hot encoding
    y_true_onehot = np.zeros((len(y_true), len(class_labels)))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1
    
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve and AUC for each class
    for i, class_name in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('evaluation/roc_curves.png')
    print("ROC curves saved to 'evaluation/roc_curves.png'")
    plt.close()

def plot_precision_recall_curves(y_true, y_proba, class_labels):
    """
    Plot and save precision-recall curves for each class.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_labels: List of class names
    """
    # Convert true labels to one-hot encoding
    y_true_onehot = np.zeros((len(y_true), len(class_labels)))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1
    
    plt.figure(figsize=(10, 8))
    
    # Calculate PR curve and AP for each class
    for i, class_name in enumerate(class_labels):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_proba[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {ap:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('evaluation/precision_recall_curves.png')
    print("Precision-Recall curves saved to 'evaluation/precision_recall_curves.png'")
    plt.close()

if __name__ == "__main__":
    # Create evaluation directory if it doesn't exist
    os.makedirs('evaluation', exist_ok=True)
    
    # Evaluate the model
    metrics = evaluate_model()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('evaluation/metrics.csv', index=False)
    print("Metrics saved to 'evaluation/metrics.csv'") 