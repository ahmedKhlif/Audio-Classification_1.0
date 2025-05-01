import os
import subprocess
import sys
import time

def run_command(command, description):
    """Run a command and print its output"""
    print("\n" + "="*80)
    print("Running: {}".format(description))
    print("="*80)

    try:
        # Use universal_newlines instead of text for Python 3.6 compatibility
        process = subprocess.run(command, check=True, universal_newlines=True)
        print("\n{} completed successfully.".format(description))
        return True
    except subprocess.CalledProcessError as e:
        print("\nError running {}: {}".format(description, e))
        return False

def main():
    """Run the entire pipeline and display plots"""
    # Step 1: Convert WAV files if needed
    success = run_command([sys.executable, 'convert_wav.py'], "Audio conversion")
    if not success:
        print("Audio conversion failed. Continuing with existing files.")

    # Step 2: Run data splitting
    success = run_command([sys.executable, 'data_split.py'], "Data splitting")
    if not success:
        print("Data splitting failed. Exiting.")
        return

    # Step 3: Run exploratory data analysis
    success = run_command([sys.executable, 'eda.py'], "Exploratory data analysis")
    if not success:
        print("EDA failed. Continuing with model training.")

    # Step 4: Train the model
    success = run_command([sys.executable, 'model.py'], "Model training")
    if not success:
        print("Model training failed. Exiting.")
        return

    # Step 5: Run evaluation
    success = run_command([sys.executable, 'evaluate.py'], "Model evaluation")
    if not success:
        print("Model evaluation failed. Exiting.")
        return

    # Step 6: Display plots
    print("\nTraining and evaluation completed. Displaying plots...")
    time.sleep(2)  # Give a moment for the user to read the message

    success = run_command([sys.executable, 'display_plots.py'], "Displaying plots")
    if not success:
        print("Failed to display plots.")
        return

    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()
