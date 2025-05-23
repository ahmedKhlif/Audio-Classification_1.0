import os
import subprocess
from cfg import Config

# Load configuration to get sample rate
config = Config()

# Path to your WAV files directory
input_dir = 'wavfiles'
output_dir = 'clean'


# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Convert each .wav file to 16-bit using ffmpeg
for f in os.listdir(input_dir):
    if f.endswith('.wav'):
        input_path = os.path.join(input_dir, f)
        output_path = os.path.join(output_dir, f)

        # Prevent overwriting by checking if the output file already exists
        if os.path.exists(output_path):
            print(f"File {f} already exists in the output directory. Skipping.")
            continue

        # Convert using ffmpeg
        command = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',  # Convert to 16-bit PCM
            '-ar', str(config.rate),  # Use the sample rate from config
            output_path, '-y'        # Overwrite existing files without asking
        ]

        # Run the conversion command
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        # Check for errors
        if result.returncode == 0:
            print(f'Converted {f} to 16-bit.')
        else:
            print(f'Error converting {f}: {result.stderr.decode()}')
