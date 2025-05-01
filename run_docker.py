import os
import subprocess
import time
import sys

def run_command(command, description):
    """Run a command and print its output"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    
    try:
        process = subprocess.run(command, check=True, text=True)
        print(f"\n{description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running {description}: {e}")
        return False

def main():
    """Run the Docker container and display results"""
    # Create necessary directories
    os.makedirs('evaluation', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('clean', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Build and run the Docker container
    print("Building and running Docker container...")
    
    # Check if Docker is installed
    if not run_command(['docker', '--version'], "Docker version check"):
        print("Docker is not installed or not in PATH. Please install Docker and try again.")
        return
    
    # Build the Docker image
    if not run_command(['docker-compose', 'build'], "Building Docker image"):
        print("Failed to build Docker image. Please check the Dockerfile and try again.")
        return
    
    # Run the Docker container
    if not run_command(['docker-compose', 'up'], "Running Docker container"):
        print("Failed to run Docker container. Please check the docker-compose.yml and try again.")
        return
    
    print("\nDocker container has completed. Check the 'evaluation' directory for results.")
    
    # Display the results
    print("\nResults are available in the following directories:")
    print("- evaluation/: Contains evaluation metrics and visualizations")
    print("- models/: Contains the trained model")
    print("- clean/: Contains processed audio files")
    
    # Ask if the user wants to view the plots
    if os.path.exists('evaluation/confusion_matrix.png'):
        response = input("\nWould you like to view the plots? (y/n): ")
        if response.lower() == 'y':
            # Run the show_plots.py script
            run_command([sys.executable, 'show_plots.py'], "Displaying plots")
    else:
        print("\nNo plots found in the evaluation directory. The model may not have been trained yet.")

if __name__ == "__main__":
    main()
