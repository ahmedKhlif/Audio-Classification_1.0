import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import pickle

from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, Dense,
    TimeDistributed, BatchNormalization, Input, Multiply, Add,
    Activation, GlobalAveragePooling2D, Lambda
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
from cfg import Config

# Check if data splits exist, if not, run data_split.py
def ensure_data_split():
    if not os.path.exists('data/train_data.csv') or not os.path.exists('data/val_data.csv') or not os.path.exists('data/test_data.csv'):
        print("Data split files not found. Running data_split.py...")
        try:
            # Run data_split.py
            subprocess.run([sys.executable, 'data_split.py'], check=True)
            print("Data splitting completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error running data_split.py: {e}")
            sys.exit(1)
    else:
        print("Data split files found.")

def extract_features_from_split_data(split_type='train'):
    """
    Extract features from the specified data split (train, val, or test).

    Args:
        split_type: Type of data split ('train', 'val', or 'test')

    Returns:
        features, labels: Arrays of features and one-hot encoded labels
    """
    print(f"Extracting features from {split_type} data...")

    # Load the data split
    split_file = f'data/{split_type}_data.csv'
    split_data = pd.read_csv(split_file)

    features = []
    labels = []
    min_value, max_value = float('inf'), -float('inf')

    # Process each file in the split
    for _, row in tqdm(split_data.iterrows(), total=len(split_data)):
        file_name = row['fname']
        file_path = os.path.join('data', split_type, file_name)

        # If file doesn't exist in the split directory, try the clean directory
        if not os.path.exists(file_path):
            file_path = os.path.join('clean', file_name)
            if not os.path.exists(file_path):
                print(f"Warning: File {file_name} not found. Skipping.")
                continue

        # Read the audio file
        sample_rate, audio_signal = wavfile.read(file_path)
        label = row['label']

        # Improved feature extraction with overlapping windows for better coverage
        # Use overlapping windows to extract more features from each audio file
        hop_length = config.step // 2  # 50% overlap between windows

        # Process audio in overlapping segments
        mfcc_features_list = []

        # If audio is too short, pad it
        if len(audio_signal) < config.step:
            audio_signal = np.pad(audio_signal, (0, config.step - len(audio_signal)))

        # Extract features from multiple windows with overlap
        for start_idx in range(0, min(len(audio_signal) - config.step, config.step * 4), hop_length):
            audio_segment = audio_signal[start_idx:start_idx + config.step]

            # Extract MFCC with improved parameters
            mfcc_segment = mfcc(
                audio_segment,
                sample_rate,
                numcep=config.nfeat,
                nfilt=config.nfilt,
                nfft=config.nfft,
                preemph=0.97,  # Pre-emphasis to enhance higher frequencies
                appendEnergy=True,  # Include energy alongside MFCCs
                winfunc=np.hamming  # Use Hamming window for better frequency resolution
            )

            # Add delta and delta-delta features for better temporal information
            delta_feat = np.zeros_like(mfcc_segment)
            delta2_feat = np.zeros_like(mfcc_segment)

            # Calculate delta features (first derivative)
            for i in range(1, mfcc_segment.shape[0]-1):
                delta_feat[i] = (mfcc_segment[i+1] - mfcc_segment[i-1]) / 2

            # Calculate delta-delta features (second derivative)
            for i in range(1, delta_feat.shape[0]-1):
                delta2_feat[i] = (delta_feat[i+1] - delta_feat[i-1]) / 2

            # Stack the features
            mfcc_segment = np.concatenate([mfcc_segment, delta_feat, delta2_feat], axis=1)

            mfcc_features_list.append(mfcc_segment)

        # Use the segment with the highest energy (likely contains the most information)
        if mfcc_features_list:
            # Calculate energy for each segment (sum of squared values)
            segment_energies = [np.sum(np.square(segment)) for segment in mfcc_features_list]
            # Select the segment with the highest energy
            mfcc_features = mfcc_features_list[np.argmax(segment_energies)]
        else:
            # Fallback to processing the first segment if something went wrong
            audio_sample = audio_signal[:config.step]
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

        # Update min and max values
        min_value = min(np.amin(mfcc_features), min_value)
        max_value = max(np.amax(mfcc_features), max_value)

        features.append(mfcc_features)
        labels.append(classes.index(label))

    # Update config with min and max values
    if split_type == 'train':
        config.min = min_value
        config.max = max_value

    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Normalize features
    features = (features - min_value) / (max_value - min_value)

    # Reshape features based on model type
    if config.mode == 'conv':
        features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
    elif config.mode == 'time':
        features = features.reshape(features.shape[0], features.shape[1], features.shape[2])

    # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes=len(classes))

    return features, labels

def create_random_features():
    """Create random features from audio files for better training"""
    # Check if cached data exists
    cached_data = load_cached_data()
    if cached_data:
        return cached_data.data[0], cached_data.data[1]

    # Initialize lists for features and labels
    features = []
    labels = []

    # Track min and max values for normalization
    min_value, max_value = float('inf'), -float('inf')

    # Calculate number of samples to generate
    num_samples = 2 * int(dataset['length'].sum() / 0.1)

    # Calculate class distribution for balanced sampling
    class_distribution = dataset.groupby(['label'])['length'].mean()
    class_probability_distribution = class_distribution / class_distribution.sum()

    # Generate random samples
    for _ in tqdm(range(num_samples), desc="Generating random features"):
        # Select a random class based on distribution
        random_class = np.random.choice(class_distribution.index, p=class_probability_distribution)

        # Select a random file from that class
        file_name = np.random.choice(dataset[dataset.label == random_class].index)

        # Read the audio file
        sample_rate, audio_signal = wavfile.read(f"clean/{file_name}")

        # Get the label
        label = dataset.at[file_name, 'label']

        # Select a random segment from the audio file
        random_index = np.random.randint(0, audio_signal.shape[0] - config.step)
        audio_sample = audio_signal[random_index:random_index + config.step]

        # Extract MFCC features
        mfcc_features = mfcc(
            audio_sample,
            sample_rate,
            numcep=config.nfeat,
            nfilt=config.nfilt,
            nfft=config.nfft
        )

        # Update min and max values
        min_value = min(np.amin(mfcc_features), min_value)
        max_value = max(np.amax(mfcc_features), max_value)

        # Add to features and labels
        features.append(mfcc_features)
        labels.append(classes.index(label))

    # Store min and max values in config
    config.min = min_value
    config.max = max_value

    # Convert to numpy arrays
    features, labels = np.array(features), np.array(labels)

    # Normalize features using min-max scaling
    features = (features - min_value) / (max_value - min_value)

    # Reshape for CNN
    if config.mode == 'conv':
        features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)

    # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes=len(classes))

    # Store data in config for caching
    config.data = (features, labels)

    # Save to pickle file
    with open(config.pickle_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)

    return features, labels

def load_cached_data():
    """Load cached data if available"""
    if os.path.isfile(config.pickle_path):
        print(f"Loading existing data for {config.mode} model")
        with open(config.pickle_path, 'rb') as handle:
            cached_data = pickle.load(handle)
        return cached_data
    return None

def build_conv_model():
    """Build a simpler but effective CNN model for audio classification"""
    # Print input shape for debugging
    print("Input shape:", input_shape)

    # Create a Sequential model
    model = Sequential([
        # First convolutional layer
        Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
               padding='same', input_shape=input_shape),

        # Second convolutional layer
        Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
               padding='same'),

        # Third convolutional layer
        Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
               padding='same'),

        # Fourth convolutional layer
        Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
               padding='same'),

        # Max pooling layer
        MaxPooling2D((2, 2)),

        # Dropout for regularization
        Dropout(0.5),

        # Flatten the output for dense layers
        Flatten(),

        # First dense layer
        Dense(128, activation='relu'),

        # Second dense layer
        Dense(64, activation='relu'),

        # Output layer
        Dense(len(classes), activation='softmax')
    ])

    model.summary()

    # Compile the model with Adam optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def evaluate_trained_model():
    """Run model evaluation after training"""
    print("\nEvaluating the trained model...")
    try:
        # Run evaluate.py
        subprocess.run([sys.executable, 'evaluate.py'], check=True)
        print("Model evaluation completed. Check the 'evaluation' directory for results.")
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluate.py: {e}")

if __name__ == "__main__":
    # Ensure data is split properly
    ensure_data_split()

    # Load dataset info for classes
    dataset = pd.read_csv('instruments.csv')
    dataset.set_index('fname', inplace=True)

    # Calculate file lengths if not already done
    for file_name in dataset.index:
        if 'length' not in dataset.columns or pd.isna(dataset.at[file_name, 'length']):
            try:
                sample_rate, signal = wavfile.read(f"clean/{file_name}")
                dataset.at[file_name, 'length'] = signal.shape[0] / sample_rate
            except:
                print(f"Warning: Could not read {file_name}. Setting default length.")
                dataset.at[file_name, 'length'] = 0

    # Get class information
    classes = list(np.unique(dataset.label))
    class_distribution = dataset.groupby(['label'])['length'].mean()

    # Display class distribution
    fig, ax = plt.subplots()
    ax.set_title('Class Distribution', y=1.08)
    ax.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', shadow=False, startangle=90)
    ax.axis('equal')
    plt.savefig('training_class_distribution.png')
    plt.close()

    # Initialize config
    config = Config(mode="conv")

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('pickles', exist_ok=True)

    # Get training and validation data using random sampling
    features, labels = create_random_features()

    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=0.1, random_state=42
    )

    # Store features and labels in config
    config.data = (train_features, train_labels)

    # Build model
    flat_labels = np.argmax(train_labels, axis=1)
    input_shape = (train_features.shape[1], train_features.shape[2], 1)
    model = build_conv_model()

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(flat_labels), y=flat_labels)
    class_weight_dict = dict(zip(range(len(classes)), class_weights))

    # Set up model checkpoint (simpler version like GitHub repo)
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    checkpoint = ModelCheckpoint(
        config.model_path,
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        period=1
    )

    # No data augmentation in the GitHub repo

    # Train model with improved parameters
    print("\nTraining model...")

    # Print training configuration
    print(f"Training with {len(train_features)} samples")
    print(f"Batch size: {config.batch_size}")
    print(f"Class weights: {class_weight_dict}")

    # Start training with simpler parameters (like GitHub repo)
    history = model.fit(
        train_features, train_labels,
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        validation_split=0.1,
        callbacks=[checkpoint]
    )

    # Save trained model
    model.save(config.model_path)
    print(f"Model saved to {config.model_path}")

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('training_history.csv', index=False)

    # Plot training history with improved visualization for beginners
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Get the history data
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_acc) + 1)

    # Plot accuracy
    ax1.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy')

    # Add grid and improve appearance
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title('Binary Accuracy', fontsize=16, fontweight='bold')

    # Set y-axis limits to focus on the relevant range
    min_acc = min(min(train_acc), min(val_acc)) * 0.95
    max_acc = max(max(train_acc), max(val_acc)) * 1.05
    ax1.set_ylim([max(0.4, min_acc), min(1.0, max_acc)])

    # Add horizontal and vertical lines to mark best accuracy
    best_epoch = val_acc.index(max(val_acc)) + 1
    best_acc = max(val_acc)
    ax1.axhline(y=best_acc, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.5)

    # Annotate the best accuracy point
    ax1.plot(best_epoch, best_acc, 'ro', markersize=8)
    ax1.annotate('Best: {:.4f}'.format(best_acc),
                xy=(best_epoch, best_acc),
                xytext=(best_epoch + 2, best_acc),
                fontsize=12,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7))

    # Add legend with better placement
    ax1.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

    # Plot loss
    ax2.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax2.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')

    # Add grid and improve appearance
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax2.set_title('Loss', fontsize=16, fontweight='bold')

    # Set y-axis limits to focus on the relevant range
    max_loss_display = max(max(train_loss[:5]), max(val_loss[:5])) * 0.8
    ax2.set_ylim([0, max_loss_display])

    # Add horizontal and vertical lines to mark best loss
    best_loss_epoch = val_loss.index(min(val_loss)) + 1
    best_loss = min(val_loss)
    ax2.axhline(y=best_loss, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=best_loss_epoch, color='gray', linestyle=':', alpha=0.5)

    # Annotate the best loss point
    ax2.plot(best_loss_epoch, best_loss, 'ro', markersize=8)
    ax2.annotate('Best: {:.4f}'.format(best_loss),
                xy=(best_loss_epoch, best_loss),
                xytext=(best_loss_epoch + 2, best_loss),
                fontsize=12,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7))

    # Add legend with better placement
    ax2.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    # Add explanatory text at the bottom
    plt.figtext(0.5, 0.01,
                "These plots show how the model's performance improves during training.\n"
                "The blue lines show performance on training data, while red lines show performance on validation data.\n"
                "Ideally, both lines should improve and converge. If they diverge, it may indicate overfitting.",
                ha="center", fontsize=12, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')

    # Also save to the root directory for backward compatibility
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Evaluate the trained model
    evaluate_trained_model()
