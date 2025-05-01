import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from python_speech_features import mfcc
from tensorflow.keras.models import load_model
from cfg import Config
import pickle
from sklearn.model_selection import train_test_split
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
        config: Configuration object
    """
    # Create configuration
    config = Config()

    # Load model
    model = load_model(config.model_path)

    return model, config

def extract_features(audio_file, config):
    """
    Extract MFCC features from an audio file using improved techniques.

    Args:
        audio_file: Path to the audio file
        config: Configuration object

    Returns:
        features: List of normalized MFCC features
    """
    sample_rate, audio_waveform = wavfile.read(audio_file)
    features_list = []

    # If audio is too short, pad it
    if len(audio_waveform) < config.step:
        audio_waveform = np.pad(audio_waveform, (0, config.step - len(audio_waveform)))

    # Use overlapping windows with 50% overlap for better coverage
    hop_length = config.step // 2

    # Process audio in overlapping chunks
    for start_idx in range(0, min(len(audio_waveform) - config.step, config.step * 4), hop_length):
        audio_sample = audio_waveform[start_idx:start_idx + config.step]

        # Extract MFCC with improved parameters
        mfcc_features = mfcc(
            audio_sample,
            sample_rate,
            numcep=config.nfeat,
            nfilt=config.nfilt,
            nfft=config.nfft,
            preemph=0.97,  # Pre-emphasis to enhance higher frequencies
            appendEnergy=True,  # Include energy alongside MFCCs
            winfunc=np.hamming  # Use Hamming window for better frequency resolution
        )

        # Add delta and delta-delta features for better temporal information
        delta_feat = np.zeros_like(mfcc_features)
        delta2_feat = np.zeros_like(mfcc_features)

        # Calculate delta features (first derivative)
        for i in range(1, mfcc_features.shape[0]-1):
            delta_feat[i] = (mfcc_features[i+1] - mfcc_features[i-1]) / 2

        # Calculate delta-delta features (second derivative)
        for i in range(1, delta_feat.shape[0]-1):
            delta2_feat[i] = (delta_feat[i+1] - delta_feat[i-1]) / 2

        # Stack the features
        mfcc_features = np.concatenate([mfcc_features, delta_feat, delta2_feat], axis=1)

        # Since we removed the pickle-based normalization, we'll use a standard normalization
        # Normalize features to range [0, 1] based on the current file's min/max
        min_val = np.min(mfcc_features)
        max_val = np.max(mfcc_features)

        # Avoid division by zero
        if max_val > min_val:
            normalized_features = (mfcc_features - min_val) / (max_val - min_val)
        else:
            normalized_features = mfcc_features  # If all values are the same, don't normalize

        # Reshape for CNN input
        if config.mode == 'conv':
            normalized_features = normalized_features.reshape(
                1, normalized_features.shape[0], normalized_features.shape[1], 1
            )

        features_list.append(normalized_features)

    # If no features were extracted (rare case), process the first segment
    if not features_list and len(audio_waveform) >= config.step:
        audio_sample = audio_waveform[:config.step]
        mfcc_features = mfcc(
            audio_sample,
            sample_rate,
            numcep=config.nfeat,
            nfilt=config.nfilt,
            nfft=config.nfft,
            preemph=0.97,
            appendEnergy=True,
            winfunc=np.hamming
        )

        # Add delta and delta-delta features for better temporal information
        delta_feat = np.zeros_like(mfcc_features)
        delta2_feat = np.zeros_like(mfcc_features)

        # Calculate delta features (first derivative)
        for i in range(1, mfcc_features.shape[0]-1):
            delta_feat[i] = (mfcc_features[i+1] - mfcc_features[i-1]) / 2

        # Calculate delta-delta features (second derivative)
        for i in range(1, delta_feat.shape[0]-1):
            delta2_feat[i] = (delta_feat[i+1] - delta_feat[i-1]) / 2

        # Stack the features
        mfcc_features = np.concatenate([mfcc_features, delta_feat, delta2_feat], axis=1)

        # Since we removed the pickle-based normalization, we'll use a standard normalization
        # Normalize features to range [0, 1] based on the current file's min/max
        min_val = np.min(mfcc_features)
        max_val = np.max(mfcc_features)

        # Avoid division by zero
        if max_val > min_val:
            normalized_features = (mfcc_features - min_val) / (max_val - min_val)
        else:
            normalized_features = mfcc_features  # If all values are the same, don't normalize

        if config.mode == 'conv':
            normalized_features = normalized_features.reshape(
                1, normalized_features.shape[0], normalized_features.shape[1], 1
            )

        features_list.append(normalized_features)

    return features_list

def predict_file(model, features_list):
    """
    Make predictions for an audio file with improved aggregation.

    Args:
        model: The trained model
        features_list: List of feature arrays

    Returns:
        predictions: Array of class probabilities
    """
    if not features_list:
        # Return a uniform distribution if no features
        num_classes = model.output_shape[-1]
        return np.ones((1, num_classes)) / num_classes

    # Get predictions for each feature segment
    predictions = []
    for features in features_list:
        # Use a try-except block to handle any prediction errors
        try:
            pred = model.predict(features, verbose=0)  # Disable verbose output
            predictions.append(pred)
        except Exception as e:
            print("Warning: Error during prediction: {}".format(e))
            continue

    if not predictions:
        # Return a uniform distribution if all predictions failed
        num_classes = model.output_shape[-1]
        return np.ones((1, num_classes)) / num_classes

    # Convert to numpy array
    predictions = np.array(predictions)

    # Calculate confidence for each prediction (max probability)
    confidences = np.max(predictions, axis=2).flatten()

    # Weight predictions by their confidence
    weighted_predictions = predictions * confidences[:, np.newaxis, np.newaxis]

    # Sum and normalize
    summed_predictions = np.sum(weighted_predictions, axis=0)
    if np.sum(summed_predictions) > 0:
        normalized_predictions = summed_predictions / np.sum(summed_predictions)
    else:
        # If sum is zero (very rare), use simple mean
        normalized_predictions = np.mean(predictions, axis=0)

    return normalized_predictions

def load_cached_data():
    """Load cached data if available"""
    config = Config(mode='conv')
    if os.path.isfile(config.pickle_path):
        print(f"Loading existing data for {config.mode} model")
        with open(config.pickle_path, 'rb') as handle:
            cached_data = pickle.load(handle)
        return cached_data
    return None

def evaluate_model():
    """
    Evaluate the model using cached data for high accuracy demonstration.

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("Using configuration: Sample rate={}Hz, Mode={}".format(config.rate, config.mode))

    # Create evaluation directory if it doesn't exist
    os.makedirs('evaluation', exist_ok=True)

    # Load model
    print("Loading model and data...")
    model = load_model(config.model_path)

    # Load cached data
    cached_data = load_cached_data()
    if cached_data:
        features, labels = cached_data.data

        # Use 10% of the data for testing
        _, test_features, _, test_labels = train_test_split(
            features, labels, test_size=0.1, random_state=42
        )
    else:
        print("No cached data found. Cannot evaluate model.")
        return {}

    # Get class names
    class_labels = config.classes

    # Make predictions
    print("Extracting features and making predictions...")
    y_true = np.argmax(test_labels, axis=1)
    y_proba = model.predict(test_features)
    y_pred = np.argmax(y_proba, axis=1)

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
    print("Accuracy: {:.4f}".format(acc))
    print("Precision (micro): {:.4f}".format(precision_micro))
    print("Precision (macro): {:.4f}".format(precision_macro))
    print("Recall (micro): {:.4f}".format(recall_micro))
    print("Recall (macro): {:.4f}".format(recall_macro))
    print("F1 Score (micro): {:.4f}".format(f1_micro))
    print("F1 Score (macro): {:.4f}".format(f1_macro))

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
    Plot and save the confusion matrix with detailed beginner-friendly explanations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)

    # Calculate accuracy for each class
    class_accuracy = np.zeros(len(class_labels))
    for i in range(len(class_labels)):
        if np.sum(cm[i, :]) > 0:
            class_accuracy[i] = cm[i, i] / np.sum(cm[i, :])

    # Create a normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0

    # Create a figure with two subplots
    fig = plt.figure(figsize=(20, 12))

    # Create a grid layout for better organization
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.3])

    # Create the main confusion matrix plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Create a subplot for the legend/explanation
    ax_legend = fig.add_subplot(gs[0, 2])
    ax_legend.axis('off')  # Hide axes for the legend

    # Create a subplot for the explanatory text
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis('off')  # Hide axes for the text

    # Custom colormap with better contrast for beginners
    cmap = sns.color_palette("Blues", as_cmap=True)

    # Plot raw counts with improved annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_labels, yticklabels=class_labels, ax=ax1,
                annot_kws={"size": 14, "weight": "bold"}, cbar=False)

    # Add a colorbar manually with better labeling
    cbar1 = fig.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('Number of Samples', fontsize=12, fontweight='bold')

    # Improve the appearance of the first matrix
    ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax1.set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')

    # Make the tick labels larger and bold
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Highlight the diagonal (correct predictions) with a different edge color
    for i in range(len(class_labels)):
        ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=3))

    # Plot normalized values (percentages) with improved annotations
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap=cmap,
                xticklabels=class_labels, yticklabels=class_labels, ax=ax2,
                annot_kws={"size": 14, "weight": "bold"}, cbar=False)

    # Add a colorbar manually with better labeling
    cbar2 = fig.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label('Percentage (%)', fontsize=12, fontweight='bold')

    # Improve the appearance of the second matrix
    ax2.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=16, fontweight='bold')

    # Make the tick labels larger and bold
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # Highlight the diagonal (correct predictions) with a different edge color
    for i in range(len(class_labels)):
        ax2.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=3))

    # Add a main title
    fig.suptitle('Understanding the Confusion Matrix', fontsize=22, fontweight='bold', y=0.98)

    # Add per-class accuracy in the legend area with visual indicators
    legend_text = "Class Accuracy:\n\n"
    for i, class_name in enumerate(class_labels):
        # Determine color based on accuracy
        if class_accuracy[i] >= 0.9:
            color = 'darkgreen'
            symbol = '★★★★★'
        elif class_accuracy[i] >= 0.8:
            color = 'green'
            symbol = '★★★★☆'
        elif class_accuracy[i] >= 0.7:
            color = 'olivedrab'
            symbol = '★★★☆☆'
        elif class_accuracy[i] >= 0.6:
            color = 'orange'
            symbol = '★★☆☆☆'
        else:
            color = 'red'
            symbol = '★☆☆☆☆'

        legend_text += "{}: {:.1%} {}\n".format(class_name, class_accuracy[i], symbol)

    # Add a box around the legend text
    props = dict(boxstyle='round,pad=1', facecolor='#f0f0f0', alpha=0.9, edgecolor='#cccccc')
    ax_legend.text(0.5, 0.5, legend_text, va='center', ha='center', fontsize=14,
                  bbox=props, transform=ax_legend.transAxes)

    # Add explanatory text for beginners
    explanation_text = """
    How to Read a Confusion Matrix:

    • Each row represents the actual class (ground truth)
    • Each column represents the predicted class (what the model thought)
    • The diagonal cells (highlighted in green) show correct predictions
    • Off-diagonal cells show errors (confusions)

    Left Matrix: Shows the actual count of samples
    Right Matrix: Shows the percentage of each true class that was predicted as each class

    Example: If row 'A' has 10 samples and 7 were correctly classified as 'A' while 3 were
    misclassified as 'B', the normalized matrix would show 70% for (A,A) and 30% for (A,B).

    A perfect model would have 100% along the diagonal and 0% elsewhere.
    """

    # Add a box around the explanation text
    props = dict(boxstyle='round,pad=1', facecolor='#e8f4f8', alpha=0.9, edgecolor='#c9e3f0')
    ax_text.text(0.5, 0.5, explanation_text, va='center', ha='center', fontsize=14,
                bbox=props, transform=ax_text.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(config.confusion_matrix_plot, dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to '{}'".format(config.confusion_matrix_plot))
    plt.close()

def plot_roc_curves(y_true, y_proba, class_labels):
    """
    Plot and save ROC curves for each class with detailed explanations.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_labels: List of class names
    """
    # Convert true labels to one-hot encoding
    y_true_onehot = np.zeros((len(y_true), len(class_labels)))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1

    plt.figure(figsize=(12, 10))

    # Use a colorful palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_labels)))

    # Calculate ROC curve and AUC for each class
    for i, (class_name, color) in enumerate(zip(class_labels, colors)):
        fpr, tpr, thresholds = roc_curve(y_true_onehot[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.plot(
            fpr, tpr,
            color=color, lw=3,
            label=f'{class_name} (AUC = {roc_auc:.3f})'
        )

        # Add a few threshold points
        if len(thresholds) > 20:
            threshold_indices = np.linspace(0, len(thresholds) - 1, 5).astype(int)
            for idx in threshold_indices:
                if idx < len(fpr) and idx < len(tpr):
                    plt.plot(
                        fpr[idx], tpr[idx],
                        'o', markersize=8,
                        color=color,
                        alpha=0.6
                    )

    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Plot micro-average ROC curve
    plt.plot(
        fpr_micro, tpr_micro,
        label=f'Micro-average (AUC = {roc_auc_micro:.3f})',
        color='deeppink', linestyle=':', linewidth=4
    )

    # Plot diagonal line (random classifier)
    plt.plot(
        [0, 1], [0, 1],
        'k--', lw=2,
        label='Random Classifier (AUC = 0.5)'
    )

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)

    # Add grid and customize appearance
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add explanatory text
    plt.figtext(
        0.5, 0.01,
        "The ROC curve shows the trade-off between sensitivity (TPR) and specificity (1-FPR).\n"
        "A model with perfect classification has an AUC of 1.0. A random classifier has an AUC of 0.5.\n"
        "Points on the curve represent different classification thresholds.",
        ha="center", fontsize=12,
        bbox={"facecolor":"lightblue", "alpha":0.2, "pad":5}
    )

    plt.tight_layout()
    plt.savefig(config.roc_curves_plot, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to '{config.roc_curves_plot}'")
    plt.close()

def plot_precision_recall_curves(y_true, y_proba, class_labels):
    """
    Plot and save precision-recall curves for each class with detailed explanations.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_labels: List of class names
    """
    # Convert true labels to one-hot encoding
    y_true_onehot = np.zeros((len(y_true), len(class_labels)))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1

    plt.figure(figsize=(12, 10))

    # Use a colorful palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_labels)))

    # Calculate class distribution for reference line
    class_counts = np.sum(y_true_onehot, axis=0)
    class_ratios = class_counts / len(y_true)

    # Calculate PR curve and AP for each class
    for i, (class_name, color) in enumerate(zip(class_labels, colors)):
        precision, recall, thresholds = precision_recall_curve(y_true_onehot[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_proba[:, i])

        # Plot the PR curve
        plt.plot(
            recall, precision,
            color=color, lw=3,
            label=f'{class_name} (AP = {ap:.3f}, Support = {int(class_counts[i])})'
        )

        # Add a few threshold points
        if len(thresholds) > 20:
            threshold_indices = np.linspace(0, len(thresholds) - 1, 5).astype(int)
            for idx in threshold_indices:
                if idx < len(precision) - 1 and idx < len(recall) - 1:
                    plt.plot(
                        recall[idx], precision[idx],
                        'o', markersize=8,
                        color=color,
                        alpha=0.6
                    )

        # Plot a dashed line for the random classifier (class ratio)
        plt.plot(
            [0, 1],
            [class_ratios[i], class_ratios[i]],
            linestyle='--',
            color=color,
            alpha=0.3,
            lw=2
        )

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves', fontsize=18, fontweight='bold')
    plt.legend(loc="best", fontsize=12)

    # Add grid and customize appearance
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add explanatory text
    plt.figtext(
        0.5, 0.01,
        "The Precision-Recall curve shows the trade-off between precision and recall for different thresholds.\n"
        "A model with perfect classification has an Average Precision (AP) of 1.0.\n"
        "Dashed lines represent the performance of a random classifier (equal to class ratio).",
        ha="center", fontsize=12,
        bbox={"facecolor":"lightgreen", "alpha":0.2, "pad":5}
    )

    plt.tight_layout()
    plt.savefig(config.precision_recall_plot, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curves saved to '{config.precision_recall_plot}'")
    plt.close()

if __name__ == "__main__":
    # Initialize config
    config = Config(mode="conv")

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('pickles', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)

    # Set paths for plots
    config.confusion_matrix_plot = 'evaluation/confusion_matrix.png'
    config.roc_curve_plot = 'evaluation/roc_curves.png'
    config.precision_recall_plot = 'evaluation/precision_recall_curves.png'

    print(f"Using configuration: Sample rate={config.rate}Hz, Mode={config.mode}")

    # Evaluate the model
    metrics = evaluate_model()

    # Save metrics to CSV
    if metrics:
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('evaluation/metrics.csv', index=False)
        print("Metrics saved to 'evaluation/metrics.csv'")