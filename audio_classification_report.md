# Audio Classification System: Technical Report

## Executive Summary

This report presents a comprehensive analysis of an audio classification system designed to categorize audio samples into three distinct classes: Alarms, Swords, and Wild Animals. The system achieves high accuracy (97.26%) through the implementation of convolutional neural networks and advanced audio processing techniques. This document explains the technical components in accessible language for readers with basic understanding of machine learning concepts.

## Table of Contents

1. Introduction
2. Model Architecture
3. Audio Processing Pipeline
4. Preprocessing Techniques
5. Training Methodology
6. Evaluation Metrics
7. Results and Performance
8. Conclusion
9. References

## 1. Introduction

Audio classification is the process of automatically categorizing sound recordings into predefined classes. This technology has numerous applications, including:

- Security systems (detecting alarms or unusual sounds)
- Environmental monitoring (identifying animal sounds)
- Content organization (categorizing audio files)
- Accessibility features (sound recognition for hearing-impaired users)

Our system focuses on distinguishing between three specific sound categories: Alarms, Swords (weapon sounds), and Wild Animals. The classification is performed using deep learning techniques, specifically convolutional neural networks (CNNs), which have proven highly effective for pattern recognition tasks.

## 2. Model Architecture

### 2.1 Convolutional Neural Network Overview

The core of our audio classification system is a Convolutional Neural Network (CNN). CNNs are particularly well-suited for audio classification because they can identify patterns across both time and frequency dimensions in audio spectrograms.

### 2.2 Layer-by-Layer Breakdown

Our model consists of the following layers:

1. **Input Layer**

   - Purpose: Receives the processed audio features (spectrograms or MFCCs)
   - Dimensions: Varies based on the audio length and feature extraction parameters
   - Reason: Serves as the entry point for data into the neural network

2. **Convolutional Layers**

   - Purpose: Extract patterns and features from the input data
   - Configuration: Multiple layers with increasing filter counts (32, 64, 128)
   - Kernel Size: 3x3 (scans the input in small windows)
   - Activation: ReLU (Rectified Linear Unit)
   - Reason: Detects increasingly complex patterns in the audio data, from simple edges to complex sound signatures

3. **Max Pooling Layers**

   - Purpose: Reduce the spatial dimensions of the data
   - Configuration: 2x2 pooling windows
   - Reason: Decreases computational load while preserving important features and providing some translation invariance

4. **Dropout Layers**

   - Purpose: Prevent overfitting
   - Rate: 0.25-0.5 (randomly deactivates 25-50% of neurons during training)
   - Reason: Forces the network to learn redundant representations, improving generalization to new data

5. **Flatten Layer**

   - Purpose: Convert multi-dimensional data to a one-dimensional vector
   - Reason: Prepares the data for processing by fully connected layers

6. **Dense (Fully Connected) Layers**

   - Purpose: Combine features for final classification
   - Configuration: 128 neurons with ReLU activation, followed by a final layer with 3 neurons (one per class)
   - Reason: Performs the actual classification based on the features extracted by earlier layers

7. **Output Layer**
   - Purpose: Produce final classification probabilities
   - Activation: Softmax (converts raw scores to probabilities that sum to 1)
   - Neurons: 3 (one for each class: Alarms, Swords, Wild Animals)
   - Reason: Provides interpretable probabilities for each class

### 2.3 Model Visualization

```
Input → Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling → Dropout → Flatten → Dense → Dropout → Dense (Output)
```

## 3. Audio Processing Pipeline

### 3.1 How the Model Processes Audio

Audio files cannot be directly fed into a neural network. Instead, we transform them into visual representations that capture the relevant characteristics of the sound. The process works as follows:

1. **Audio Loading**

   - Raw audio files (.wav format) are loaded into memory
   - All files are standardized to a consistent sample rate (16,000 Hz)
   - Reason: Ensures all audio inputs have the same temporal resolution

2. **Feature Extraction**

   - The audio is converted into either:
     - **Spectrograms**: Visual representations showing how frequencies change over time
     - **MFCCs (Mel-Frequency Cepstral Coefficients)**: Features that represent the short-term power spectrum of sound in a way that approximates human auditory perception
   - Reason: Transforms audio into a format that highlights patterns relevant for classification

3. **Data Representation**

   - The extracted features are represented as 2D arrays (similar to images)
   - These arrays capture both time (x-axis) and frequency (y-axis) information
   - Reason: Allows convolutional layers to detect patterns in both dimensions

4. **Batch Processing**
   - Audio samples are processed in batches during training
   - Reason: Improves training efficiency and stability

### 3.2 Feature Extraction Details

#### Spectrograms

- Created using Short-Time Fourier Transform (STFT)
- Window Size: 25ms with 10ms overlap
- Frequency Range: 0-8000 Hz (covers most relevant sounds)
- Advantage: Preserves detailed frequency information

#### MFCCs

- Number of coefficients: 13-40 (we use 20)
- Based on human auditory perception (Mel scale)
- Advantage: Compact representation that focuses on perceptually relevant information

## 4. Preprocessing Techniques

### 4.1 Audio Standardization

1. **Resampling**

   - All audio files are converted to 16,000 Hz sample rate
   - Reason: Ensures consistency across all inputs and reduces computational requirements

2. **Duration Normalization**

   - Audio clips are either trimmed or padded to a standard length
   - Reason: Neural networks require fixed-size inputs

3. **Amplitude Normalization**
   - Audio amplitudes are scaled to the range [-1, 1]
   - Reason: Prevents loud sounds from dominating the learning process

### 4.2 Noise Handling

1. **Silence Removal**

   - Silent portions at the beginning and end of recordings are trimmed
   - Reason: Focuses the model on the relevant sound content

2. **Background Noise Reduction**
   - Optional preprocessing step to reduce ambient noise
   - Reason: Helps the model focus on the primary sound source

### 4.3 Data Augmentation

To improve model robustness and prevent overfitting, we apply several augmentation techniques:

1. **Time Shifting**

   - Randomly shifting the audio forward or backward in time
   - Shift Range: ±0.1 seconds
   - Reason: Makes the model robust to variations in when a sound occurs

2. **Pitch Shifting**

   - Slightly altering the pitch of the audio
   - Shift Range: ±2 semitones
   - Reason: Helps the model generalize across different pitches of the same sound

3. **Speed Modification**

   - Speeding up or slowing down the audio slightly
   - Speed Range: 0.9x to 1.1x
   - Reason: Makes the model robust to variations in sound speed

4. **Adding Background Noise**
   - Mixing in low-level random noise
   - Noise Level: 0.005 to 0.02 of the signal amplitude
   - Reason: Improves performance in real-world noisy environments

## 5. Training Methodology

### 5.1 Data Splitting

The dataset is divided into three parts:

- **Training Set (70%)**: Used to update the model weights
- **Validation Set (15%)**: Used to tune hyperparameters and monitor for overfitting
- **Test Set (15%)**: Used only for final evaluation

### 5.2 Training Techniques

1. **Batch Training**

   - Batch Size: 32 samples
   - Reason: Provides stable gradient updates while maintaining computational efficiency

2. **Optimization Algorithm**

   - Algorithm: Adam (Adaptive Moment Estimation)
   - Initial Learning Rate: 0.001
   - Reason: Adapts the learning rate for each parameter, leading to faster convergence

3. **Loss Function**
   - Function: Categorical Cross-Entropy
   - Reason: Appropriate for multi-class classification problems

### 5.3 Training Enhancements

1. **Checkpointing**

   - The model's weights are saved whenever performance on the validation set improves
   - Reason: Preserves the best model version and allows recovery from interruptions

2. **Early Stopping**

   - Training stops when validation performance stops improving for a set number of epochs
   - Patience: 10 epochs
   - Reason: Prevents overfitting by stopping training when the model starts to memorize the training data

3. **Learning Rate Reduction**

   - The learning rate is reduced when validation performance plateaus
   - Reduction Factor: 0.5
   - Patience: 5 epochs
   - Reason: Allows for finer weight adjustments as training progresses

4. **Batch Normalization**
   - Normalizes the outputs of each layer before passing to the next layer
   - Reason: Accelerates training and improves stability

## 6. Evaluation Metrics

### 6.1 Primary Metrics

1. **Accuracy**

   - Definition: Percentage of correctly classified samples
   - Formula: (True Positives + True Negatives) / Total Samples
   - Interpretation: Overall correctness of the model
   - Our Result: 97.26%

2. **Precision**

   - Definition: Proportion of positive identifications that were actually correct
   - Formula: True Positives / (True Positives + False Positives)
   - Interpretation: Measures how reliable positive predictions are
   - Our Result (Macro): 98.39%

3. **Recall (Sensitivity)**

   - Definition: Proportion of actual positives that were correctly identified
   - Formula: True Positives / (True Positives + False Negatives)
   - Interpretation: Measures the model's ability to find all positive samples
   - Our Result (Macro): 97.47%

4. **F1 Score**
   - Definition: Harmonic mean of precision and recall
   - Formula: 2 _ (Precision _ Recall) / (Precision + Recall)
   - Interpretation: Balance between precision and recall
   - Our Result (Macro): 97.89%

### 6.2 Additional Evaluation Tools

1. **Confusion Matrix**

   - Description: Table showing predicted vs. actual class assignments
   - Interpretation: Reveals which classes are being confused with each other
   - Our Result: Minimal confusion between classes, with most samples correctly classified

2. **ROC Curves (Receiver Operating Characteristic)**

   - Description: Plot of true positive rate vs. false positive rate at various thresholds
   - Interpretation: Area Under Curve (AUC) indicates classification performance
   - Our Result: High AUC values for all classes

3. **Precision-Recall Curves**
   - Description: Plot of precision vs. recall at various thresholds
   - Interpretation: Shows trade-off between precision and recall
   - Our Result: High area under the curve, indicating good performance

## 7. Results and Performance

### 7.1 Overall Performance

Our audio classification system achieved excellent results:

- **Accuracy**: 97.26%
- **Macro Precision**: 98.39%
- **Macro Recall**: 97.47%
- **Macro F1 Score**: 97.89%

These metrics indicate that the model can reliably classify audio samples into the correct categories with very few errors.

### 7.2 Per-Class Performance

| Class        | Precision | Recall | F1 Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Swords       | 100%      | 100%   | 100%     | 87      |
| Wild Animals | 99%       | 93%    | 96%      | 578     |
| Alarms       | 96%       | 100%   | 98%      | 1014    |

This breakdown shows that:

- The model perfectly classifies sword sounds
- Wild animal sounds have excellent precision but slightly lower recall
- Alarm sounds have perfect recall with very high precision

### 7.3 Error Analysis

The few misclassifications that occur are primarily:

- Wild animal sounds classified as alarms (likely due to similar frequency patterns)
- No significant confusion between swords and other categories

## 8. Conclusion

The audio classification system developed in this project demonstrates the effectiveness of convolutional neural networks for audio classification tasks. By transforming audio data into visual representations and applying appropriate preprocessing techniques, we achieved high classification accuracy across all target classes.

Key strengths of the system include:

- High overall accuracy (97.26%)
- Excellent per-class performance
- Robust preprocessing pipeline
- Effective training methodology with overfitting prevention

Potential areas for future improvement:

- Expanding to more audio classes
- Testing with more diverse and challenging audio samples
- Exploring alternative model architectures (e.g., recurrent neural networks)
- Implementing real-time classification capabilities

## 9. References

1. Hershey, S., Chaudhuri, S., Ellis, D. P., et al. (2017). CNN architectures for large-scale audio classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

2. McFee, B., Raffel, C., Liang, D., et al. (2015). librosa: Audio and music signal analysis in python. In Proceedings of the 14th python in science conference.

3. Piczak, K. J. (2015). Environmental sound classification with convolutional neural networks. In 2015 IEEE 25th International Workshop on Machine Learning for Signal Processing (MLSP).

4. Salamon, J., Bello, J. P. (2017). Deep convolutional neural networks and data augmentation for environmental sound classification. IEEE Signal Processing Letters, 24(3), 279-283.

5. Tensorflow Team. (2021). TensorFlow: Large-scale machine learning on heterogeneous systems. Retrieved from https://www.tensorflow.org/
