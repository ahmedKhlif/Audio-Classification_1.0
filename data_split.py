import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from cfg import Config
import shutil

def split_dataset(test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the dataset into train, validation and test sets.

    Args:
        test_size: Proportion of the dataset to include in the test split
        val_size: Proportion of the training dataset to include in the validation split
        random_state: Random seed for reproducibility

    Returns:
        train_data, val_data, test_data: DataFrames containing split data
    """
    print("Splitting dataset into train, validation, and test sets...")

    # Load dataset
    df = pd.read_csv('instruments.csv')

    # First split into training and test sets
    train_data, test_data = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']  # Ensure class balance is maintained
    )

    # Then split training data into training and validation sets
    train_data, val_data = train_test_split(
        train_data,
        test_size=val_size,
        random_state=random_state,
        stratify=train_data['label']  # Ensure class balance is maintained
    )

    # Reset indices
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Save the split datasets
    train_data.to_csv('data/train_data.csv', index=False)
    val_data.to_csv('data/val_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)

    # Print distribution information
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val_data)} ({len(val_data)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")

    # Print class distribution
    print("\nClass distribution:")
    print("Training set:")
    print(train_data['label'].value_counts(normalize=True).sort_index())
    print("\nValidation set:")
    print(val_data['label'].value_counts(normalize=True).sort_index())
    print("\nTest set:")
    print(test_data['label'].value_counts(normalize=True).sort_index())

    return train_data, val_data, test_data

def organize_files():
    """
    Organize the clean audio files into train, validation, and test directories
    based on the splits.
    """
    # Create directories if they don't exist
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/val', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)

    # Load split data
    train_data = pd.read_csv('data/train_data.csv')
    val_data = pd.read_csv('data/val_data.csv')
    test_data = pd.read_csv('data/test_data.csv')

    # Copy files to their respective directories
    print("Organizing files into train, validation, and test directories...")

    for file_name in train_data['fname']:
        src_path = os.path.join('clean', file_name)
        dst_path = os.path.join('data/train', file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    for file_name in val_data['fname']:
        src_path = os.path.join('clean', file_name)
        dst_path = os.path.join('data/val', file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    for file_name in test_data['fname']:
        src_path = os.path.join('clean', file_name)
        dst_path = os.path.join('data/test', file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    print(f"Files organized: {len(train_data)} in train, {len(val_data)} in val, {len(test_data)} in test")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Initialize config
    config = Config()
    print(f"Using configuration: Sample rate={config.rate}Hz, Envelope threshold={config.envelope_threshold}")

    # Split the dataset
    train_data, val_data, test_data = split_dataset()

    # Organize files into directories
    organize_files()

    print("Data splitting and organization complete.")