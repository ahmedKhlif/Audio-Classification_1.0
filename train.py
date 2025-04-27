#!/usr/bin/env python3
"""
Audio Classification Training Script
-----------------------------------
This script runs the complete audio classification workflow:
1. Data preparation (splitting into train/val/test)
2. Model training
3. Model evaluation

Example usage:
    python train.py
"""

import subprocess
import sys
import os
import time

def print_section(title):
    """Print a formatted section title"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, '='))
    print("="*80 + "\n")

def main():
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)
    
    # Step 1: Data Preparation
    print_section("STEP 1: DATA PREPARATION")
    try:
        subprocess.run([sys.executable, 'data_split.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during data splitting: {e}")
        return 1
    
    # Step 2: Model Training
    print_section("STEP 2: MODEL TRAINING")
    try:
        subprocess.run([sys.executable, 'model.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during model training: {e}")
        return 1
    
    # Step 3: Model Evaluation (This is now integrated into model.py)
    # But we can still explicitly call it if needed
    
    # Print completion message with elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print_section("TRAINING COMPLETE")
    print(f"Total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print("\nCheck the 'evaluation' directory for performance metrics and visualizations.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 