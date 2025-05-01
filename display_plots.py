import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from cfg import Config

# Initialize configuration
config = Config()

def save_class_distribution():
    """Generate and save the class distribution pie chart"""
    print("Generating class distribution plot...")

    # Load dataset
    df = pd.read_csv('instruments.csv')
    df.set_index('fname', inplace=True)

    # Calculate file lengths if not already done
    for f in df.index:
        if 'length' not in df.columns or pd.isna(df.at[f, 'length']):
            try:
                sample_rate, signal = wavfile.read("wavfiles/{}".format(f))
                df.at[f, 'length'] = signal.shape[0] / sample_rate
            except:
                print("Warning: Could not read {}. Setting default length.".format(f))
                df.at[f, 'length'] = 0

    # Get class distribution
    classes = list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()

    # Plot class distribution
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(
        class_dist,
        labels=class_dist.index,
        autopct='%1.1f%%',
        shadow=False,
        startangle=90,
        explode=[0.05] * len(class_dist),  # Slightly explode each slice
        colors=sns.color_palette("viridis", len(class_dist))
    )

    # Enhance text properties
    plt.setp(autotexts, size=12, weight="bold")
    plt.setp(texts, size=14)

    plt.axis('equal')
    plt.title('Audio Class Distribution', fontsize=18, fontweight='bold')

    # Add a legend with class counts
    class_counts = df.groupby('label').size()
    legend_labels = ["{} ({} files)".format(label, count) for label, count in class_counts.items()]
    plt.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.savefig(config.class_distribution_plot, dpi=300, bbox_inches='tight')
    print("Class distribution plot saved to {}".format(config.class_distribution_plot))

    return classes

def save_spectrograms(classes):
    """Generate and save spectrograms for each class"""
    print("Generating spectrograms...")

    # Load dataset
    df = pd.read_csv('instruments.csv')

    # Create a figure with subplots for each class
    fig, axes = plt.subplots(len(classes), 1, figsize=(12, 4*len(classes)))

    # If there's only one class, axes won't be an array
    if len(classes) == 1:
        axes = [axes]

    # For each class, display a spectrogram of the first file
    for i, class_name in enumerate(classes):
        # Get the first file of this class
        file_name = df[df.label == class_name].iloc[0]['fname']
        file_path = os.path.join('wavfiles', file_name)

        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Create spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        # Display spectrogram
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', ax=axes[i])
        axes[i].set_title('Spectrogram: {}'.format(class_name), fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Time (s)', fontsize=12)
        axes[i].set_ylabel('Frequency (Hz)', fontsize=12)

    # Add a colorbar to the last subplot
    cbar = fig.colorbar(img, ax=axes, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)', fontsize=12)

    # Add a main title
    fig.suptitle('Audio Spectrograms by Class', fontsize=18, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(config.feature_plots['spectrograms'], dpi=300, bbox_inches='tight')
    print("Spectrograms saved to {}".format(config.feature_plots['spectrograms']))

def save_mfcc_features(classes):
    """Generate and save MFCC features for each class"""
    print("Generating MFCC features plot...")

    # Load dataset
    df = pd.read_csv('instruments.csv')

    # Create a figure with subplots for each class
    fig, axes = plt.subplots(len(classes), 1, figsize=(12, 4*len(classes)))

    # If there's only one class, axes won't be an array
    if len(classes) == 1:
        axes = [axes]

    # For each class, display MFCC features of the first file
    for i, class_name in enumerate(classes):
        # Get the first file of this class
        file_name = df[df.label == class_name].iloc[0]['fname']
        file_path = os.path.join('wavfiles', file_name)

        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Display MFCC features
        img = librosa.display.specshow(mfccs, x_axis='time', ax=axes[i])
        axes[i].set_title('MFCC Features: {}'.format(class_name), fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Time (frames)', fontsize=12)
        axes[i].set_ylabel('MFCC Coefficients', fontsize=12)

    # Add a colorbar to the last subplot
    cbar = fig.colorbar(img, ax=axes)
    cbar.set_label('MFCC Amplitude', fontsize=12)

    # Add a main title
    fig.suptitle('Mel-Frequency Cepstral Coefficients by Class', fontsize=18, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(config.feature_plots['mfccs'], dpi=300, bbox_inches='tight')
    print("MFCC features plot saved to {}".format(config.feature_plots['mfccs']))

def save_evaluation_metrics():
    """Generate and save evaluation metrics: precision, recall, F1-score with beginner-friendly explanations"""
    print("Generating evaluation metrics plot...")

    # Check if evaluation metrics file exists
    if not os.path.exists('evaluation/metrics.csv'):
        print("Evaluation metrics not found. Please run evaluation first.")
        return

    # Load metrics
    metrics_df = pd.read_csv('evaluation/metrics.csv')

    # Create a figure for metrics with a more appealing design
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set background color for better visual appeal
    ax.set_facecolor('#f8f9fa')

    # Create a bar chart for precision, recall, and F1-score
    metrics = ['precision_macro', 'recall_macro', 'f1_macro']
    display_names = ['Precision', 'Recall', 'F1 Score']
    values = [metrics_df[metric].values[0] for metric in metrics]

    # Use a nice color palette with higher saturation
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

    # Create bars with a slight shadow effect for depth
    bars = ax.bar(
        display_names,
        values,
        color=colors,
        width=0.6,
        edgecolor='white',
        linewidth=1.5,
        alpha=0.8
    )

    # Add a horizontal line at 0.5 and 0.75 for reference
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Add text labels for the reference lines
    ax.text(2.5, 0.51, 'Baseline (0.5)', fontsize=10, va='bottom', ha='center', color='gray')
    ax.text(2.5, 0.76, 'Good Performance (0.75)', fontsize=10, va='bottom', ha='center', color='gray')

    # Set y-axis limits and grid
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add titles and labels with improved styling
    ax.set_title('Model Performance Metrics', fontsize=20, fontweight='bold', pad=20)
    ax.set_ylabel('Score (higher is better)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')

    # Add value labels on top of bars
    for bar, value, color in zip(bars, values, colors):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            '{:.4f}'.format(value),
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold',
            color=color
        )

        # Add a qualitative assessment under each bar
        assessment = "Excellent" if value > 0.9 else "Good" if value > 0.75 else "Fair" if value > 0.6 else "Poor"
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            0.05,
            assessment,
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            color=color
        )

    # Add accuracy as a separate element with more prominence
    if 'accuracy' in metrics_df.columns:
        accuracy = metrics_df['accuracy'].values[0]

        # Create a text box for accuracy
        props = dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8, edgecolor='#cccccc')
        ax.text(
            0.5, -0.15,
            "Overall Accuracy: {:.2%}".format(accuracy),
            transform=ax.transAxes,
            ha="center",
            fontsize=16,
            fontweight='bold',
            bbox=props
        )

    # Add explanatory text for beginners
    explanation_text = (
        "Precision: How many of the predicted positives are actually correct?\n"
        "Recall: How many of the actual positives did the model correctly identify?\n"
        "F1 Score: The harmonic mean of precision and recall (balance between the two)."
    )

    fig.text(
        0.5, 0.01,
        explanation_text,
        ha="center",
        fontsize=12,
        bbox={"facecolor":"#e8f4f8", "alpha":0.9, "pad":5, "edgecolor":"#c9e3f0"}
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.savefig(config.feature_plots['metrics'], dpi=300, bbox_inches='tight')
    print("Performance metrics plot saved to {}".format(config.feature_plots['metrics']))

def save_all_plots():
    """Generate and save all plots with detailed explanations"""
    # Create directories if they don't exist
    os.makedirs(config.plots_dir, exist_ok=True)
    os.makedirs(config.evaluation_dir, exist_ok=True)

    print(f"Saving all plots to {config.plots_dir} and {config.evaluation_dir} directories...")

    # Generate and save class distribution
    classes = save_class_distribution()

    # Generate and save spectrograms
    save_spectrograms(classes)

    # Generate and save MFCC features
    save_mfcc_features(classes)

    # Generate and save evaluation metrics
    save_evaluation_metrics()

    # Create an index.html file to view all plots
    create_plot_index_html()

    print("\nAll plots have been saved with detailed explanations and legends.")
    print("You can find them in the '{}' and '{}' directories.".format(config.plots_dir, config.evaluation_dir))

def create_plot_index_html():
    """Create an HTML file to view all plots with beginner-friendly explanations"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Classification Plots</title>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 20px;
                background-color: #f8f9fa;
                color: #333;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 15px;
                border-bottom: 2px solid #eee;
            }
            h2 {
                color: #3498db;
                margin-top: 30px;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
            }
            .plot-container {
                background-color: white;
                padding: 25px;
                margin: 30px 0;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .plot-container:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.15);
            }
            .plot {
                text-align: center;
                margin: 20px 0;
            }
            .plot img {
                max-width: 100%;
                height: auto;
                border: 1px solid #eee;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .description {
                margin-top: 20px;
                line-height: 1.7;
                color: #444;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }
            .key-points {
                background-color: #e8f4f8;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
            }
            .key-points h3 {
                margin-top: 0;
                color: #2980b9;
            }
            .key-points ul {
                padding-left: 20px;
            }
            .key-points li {
                margin-bottom: 8px;
            }
            .highlight {
                font-weight: bold;
                color: #2980b9;
            }
            .note {
                font-style: italic;
                color: #7f8c8d;
                margin-top: 10px;
            }
            .header-with-icon {
                display: flex;
                align-items: center;
            }
            .header-with-icon i {
                margin-right: 10px;
                color: #3498db;
            }
            .navigation {
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                z-index: 100;
            }
            .navigation ul {
                list-style-type: none;
                padding: 0;
                margin: 0;
            }
            .navigation li {
                margin-bottom: 5px;
            }
            .navigation a {
                text-decoration: none;
                color: #3498db;
            }
            .navigation a:hover {
                text-decoration: underline;
            }
            .section-divider {
                height: 2px;
                background: linear-gradient(to right, #3498db, transparent);
                margin: 40px 0;
            }
        </style>
    </head>
    <body>
        <div class="navigation">
            <ul>
                <li><a href="#training">Training History</a></li>
                <li><a href="#distribution">Class Distribution</a></li>
                <li><a href="#spectrograms">Spectrograms</a></li>
                <li><a href="#mfcc">MFCC Features</a></li>
                <li><a href="#metrics">Performance Metrics</a></li>
                <li><a href="#confusion">Confusion Matrix</a></li>
            </ul>
        </div>

        <h1>Audio Classification Visualization Results</h1>
        <p style="text-align: center; margin-bottom: 30px;">
            This page shows the results of training and evaluating an audio classification model.
            Each visualization helps understand different aspects of the data and model performance.
        </p>

        <div class="section-divider"></div>

        <div id="training" class="plot-container">
            <h2 class="header-with-icon">Training History</h2>
            <div class="plot">
                <img src="training_history.png" alt="Training History">
            </div>
            <div class="description">
                <p>These plots show how the model's performance improved during training:</p>
                <ul>
                    <li><strong>Accuracy</strong> (left): How often the model's predictions were correct</li>
                    <li><strong>Loss</strong> (right): The error measurement that the model tries to minimize</li>
                </ul>
                <p>The blue lines show performance on training data, while red lines show performance on validation data (data the model hasn't seen during training).</p>

                <div class="key-points">
                    <h3>What to Look For:</h3>
                    <ul>
                        <li>Both accuracy and loss should improve over time (accuracy increases, loss decreases)</li>
                        <li>If the blue and red lines diverge significantly, it may indicate <span class="highlight">overfitting</span> (the model is memorizing training data rather than learning general patterns)</li>
                        <li>The best model is usually the one with the highest validation accuracy or lowest validation loss</li>
                    </ul>
                </div>

                <p class="note">Note: The training process automatically stops when the model stops improving, which is why the plots may not show all epochs.</p>
            </div>
        </div>

        <div id="distribution" class="plot-container">
            <h2 class="header-with-icon">Class Distribution</h2>
            <div class="plot">
                <img src="class_distribution.png" alt="Class Distribution">
            </div>
            <div class="description">
                <p>This pie chart shows the distribution of audio classes in the dataset:</p>
                <ul>
                    <li>Each slice represents a different audio class</li>
                    <li>The size of each slice shows the proportion of that class in the dataset</li>
                    <li>The percentages indicate what portion of the dataset belongs to each class</li>
                </ul>

                <div class="key-points">
                    <h3>Why This Matters:</h3>
                    <ul>
                        <li>A balanced dataset (similar sized slices) usually leads to better model performance</li>
                        <li>If classes are imbalanced, the model might be biased toward the majority classes</li>
                        <li>Understanding class distribution helps interpret the model's performance metrics</li>
                    </ul>
                </div>
            </div>
        </div>

        <div id="spectrograms" class="plot-container">
            <h2 class="header-with-icon">Spectrograms</h2>
            <div class="plot">
                <img src="spectrograms.png" alt="Spectrograms">
            </div>
            <div class="description">
                <p>Spectrograms visualize the frequency content of audio signals over time:</p>
                <ul>
                    <li>The <strong>x-axis</strong> represents time</li>
                    <li>The <strong>y-axis</strong> represents frequency (higher = higher pitch)</li>
                    <li>The <strong>color intensity</strong> represents amplitude/loudness (brighter = louder)</li>
                </ul>

                <div class="key-points">
                    <h3>What to Look For:</h3>
                    <ul>
                        <li>Each audio class typically has a distinct spectral pattern</li>
                        <li>Horizontal lines indicate sustained tones at specific frequencies</li>
                        <li>Vertical lines indicate sudden sounds or transients</li>
                        <li>The model learns to recognize these patterns to classify audio</li>
                    </ul>
                </div>

                <p class="note">These visual patterns are what the model uses to distinguish between different audio classes.</p>
            </div>
        </div>

        <div id="mfcc" class="plot-container">
            <h2 class="header-with-icon">MFCC Features</h2>
            <div class="plot">
                <img src="mfccs.png" alt="MFCC Features">
            </div>
            <div class="description">
                <p>Mel-Frequency Cepstral Coefficients (MFCCs) are features extracted from audio that represent how humans perceive sound:</p>
                <ul>
                    <li>The <strong>x-axis</strong> represents time frames</li>
                    <li>The <strong>y-axis</strong> represents different MFCC coefficients</li>
                    <li>The <strong>color</strong> represents the value of each coefficient</li>
                </ul>

                <div class="key-points">
                    <h3>Why We Use MFCCs:</h3>
                    <ul>
                        <li>They capture the most important aspects of sound as perceived by humans</li>
                        <li>They compress the information in spectrograms into a more compact form</li>
                        <li>They're widely used in speech and audio recognition because they work well</li>
                        <li>Different audio classes create different MFCC patterns</li>
                    </ul>
                </div>

                <p class="note">MFCCs are the actual features fed into the machine learning model, not the raw audio.</p>
            </div>
        </div>

        <div id="metrics" class="plot-container">
            <h2 class="header-with-icon">Performance Metrics</h2>
            <div class="plot">
                <img src="performance_metrics.png" alt="Performance Metrics">
            </div>
            <div class="description">
                <p>This bar chart shows the key performance metrics of the classification model:</p>

                <div class="key-points">
                    <h3>Understanding the Metrics:</h3>
                    <ul>
                        <li><strong>Precision</strong>: When the model predicts a class, how often is it correct? (Measures false positives)</li>
                        <li><strong>Recall</strong>: Out of all actual instances of a class, how many did the model correctly identify? (Measures false negatives)</li>
                        <li><strong>F1 Score</strong>: The harmonic mean of precision and recall, providing a balance between the two</li>
                    </ul>
                </div>

                <p>All metrics range from 0 to 1, where higher is better. A score above 0.7 is generally considered good, above 0.8 is very good, and above 0.9 is excellent.</p>

                <p class="note">The overall accuracy is shown at the bottom of the chart and represents the percentage of all predictions that were correct.</p>
            </div>
        </div>

        <div id="confusion" class="plot-container">
            <h2 class="header-with-icon">Confusion Matrix</h2>
            <div class="plot">
                <img src="../evaluation/confusion_matrix.png" alt="Confusion Matrix">
            </div>
            <div class="description">
                <p>The confusion matrix provides a detailed breakdown of the model's predictions:</p>

                <div class="key-points">
                    <h3>How to Read It:</h3>
                    <ul>
                        <li>Each <strong>row</strong> represents the actual class (ground truth)</li>
                        <li>Each <strong>column</strong> represents the predicted class</li>
                        <li>The <strong>diagonal cells</strong> (highlighted in green) show correct predictions</li>
                        <li>The <strong>off-diagonal cells</strong> show errors or "confusions"</li>
                    </ul>
                </div>

                <p>The left matrix shows the actual count of samples, while the right matrix shows the percentage of each true class that was predicted as each class.</p>

                <p>For example, if row 'A' has 10 samples and 7 were correctly classified as 'A' while 3 were misclassified as 'B', the normalized matrix would show 70% for (A,A) and 30% for (A,B).</p>

                <p class="note">A perfect model would have 100% along the diagonal and 0% elsewhere. The class accuracy ratings on the right show how well the model performs for each individual class.</p>
            </div>
        </div>

        <div style="text-align: center; margin: 40px 0; color: #7f8c8d;">
            <p>Audio Classification Project - Created with TensorFlow and Python</p>
        </div>
    </body>
    </html>
    """

    # Save the HTML file
    html_path = os.path.join(config.plots_dir, 'index.html')
    with open(html_path, 'w') as f:
        f.write(html_content)

    print("Created HTML index for plots at {}".format(html_path))

def display_saved_plots():
    """Display all saved plots"""
    # Check if plots exist
    plot_files = [
        config.class_distribution_plot,
        config.feature_plots['spectrograms'],
        config.feature_plots['mfccs'],
        config.feature_plots['metrics'],
        config.confusion_matrix_plot
    ]

    for plot_file in plot_files:
        if os.path.exists(plot_file):
            from PIL import Image
            img = Image.open(plot_file)
            plt.figure(figsize=(12, 8))
            plt.imshow(np.array(img))
            plt.axis('off')
            plt.title(os.path.basename(plot_file).replace('.png', '').replace('_', ' ').title(), fontsize=16)
            plt.tight_layout()
            plt.show()
        else:
            print("Plot file not found: {}".format(plot_file))

def main():
    """Main function to generate and save all plots"""
    # Generate and save all plots
    save_all_plots()

    # In Docker environment, don't try to display plots or ask for input
    # Just inform where the plots are saved
    print("\nAll plots have been saved with detailed explanations and legends.")
    print("You can find them in the '{}' and '{}' directories.".format(config.plots_dir, config.evaluation_dir))
    print("You can view them by opening '{}' in a web browser.".format(os.path.join(config.plots_dir, 'index.html')))

if __name__ == "__main__":
    main()
