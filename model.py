import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
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

def load_cached_data():
    if os.path.isfile(config.pickle_path):
        print(f"Loading existing data for {config.mode} model")
        with open(config.pickle_path, 'rb') as handle:
            cached_data = pickle.load(handle)
            return cached_data
    return None

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
        
        # Extract MFCC features
        audio_sample = audio_signal[:config.step] if len(audio_signal) >= config.step else np.pad(audio_signal, (0, config.step - len(audio_signal)))
        mfcc_features = mfcc(audio_sample, sample_rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        
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
    """Legacy function for backwards compatibility"""
    cached_data = load_cached_data()
    if cached_data:
        return cached_data.data[0], cached_data.data[1]
    
    # If no cached data, extract features from split data
    return extract_features_from_split_data('train')

def build_conv_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(classes), activation='softmax')
    ])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    
    # Get training and validation data
    train_features, train_labels = extract_features_from_split_data('train')
    val_features, val_labels = extract_features_from_split_data('val')
    
    # Save features and labels
    config.data = (train_features, train_labels)
    with open(config.pickle_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    
    # Build model
    flat_labels = np.argmax(train_labels, axis=1)
    input_shape = (train_features.shape[1], train_features.shape[2], 1)
    model = build_conv_model()
    
    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(flat_labels), y=flat_labels)
    class_weight_dict = dict(zip(range(len(classes)), class_weights))
    
    # Set up model checkpoint
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    checkpoint = ModelCheckpoint(
        config.model_path, 
        monitor='val_accuracy', 
        verbose=1, 
        mode='max', 
        save_best_only=True, 
        save_weights_only=False
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_features, train_labels,
        validation_data=(val_features, val_labels),
        epochs=10,
        batch_size=32,
        shuffle=True,
        class_weight=class_weight_dict,
        callbacks=[checkpoint]
    )
    
    # Save trained model
    model.save(config.model_path)
    print(f"Model saved to {config.model_path}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('training_history.csv', index=False)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Evaluate the trained model
    evaluate_trained_model()
