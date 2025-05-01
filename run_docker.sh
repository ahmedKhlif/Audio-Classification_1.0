#!/bin/bash

# Simple script to run the audio classification project in Docker

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p evaluation models clean plots output

# Function to run the project
run_project() {
    echo "Building and running the Docker container..."
    docker-compose build
    docker-compose up
}

# Function to build the Docker image
build_project() {
    echo "Building the Docker image..."
    docker-compose build
}

# Function to run a specific script
run_script() {
    SCRIPT=$1
    if [ -z "$SCRIPT" ]; then
        echo "Please specify a script to run."
        exit 1
    fi

    echo "Running $SCRIPT..."
    docker-compose run --rm audio-tools $SCRIPT
}

# Function to clean up Docker resources
clean_docker() {
    echo "Removing Docker containers and images..."
    docker-compose down
    docker rmi audio-classification:latest
    echo "Cleanup complete."
}

# Function to display help
show_help() {
    echo "Audio Classification Docker Helper"
    echo ""
    echo "Usage:"
    echo "  ./run_docker.sh [command]"
    echo ""
    echo "Commands:"
    echo "  run             Run the complete workflow (default)"
    echo "  build           Build the Docker image"
    echo "  script <name>   Run a specific script (e.g., display_plots.py)"
    echo "  clean           Remove all Docker containers and images"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_docker.sh run"
    echo "  ./run_docker.sh script display_plots.py"
    echo "  ./run_docker.sh script model.py"
}

# Parse command line arguments
COMMAND=${1:-run}

case $COMMAND in
    run)
        run_project
        ;;
    build)
        build_project
        ;;
    script)
        run_script $2
        ;;
    clean)
        clean_docker
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

exit 0
