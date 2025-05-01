import os

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000,
                 epochs=10, batch_size=32, learning_rate=0.001, envelope_threshold=0.0005):
        # Audio processing parameters
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)

        # Default normalization values (used when no pickle file is available)
        self.min = -100
        self.max = 100

        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.envelope_threshold = envelope_threshold

        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('evaluation', exist_ok=True)

        # Paths
        self.model_path = os.path.join('models', f'{mode}.model')
        self.pickle_path = os.path.join('pickles', f'{mode}.p')

        # Create pickles directory if it doesn't exist
        os.makedirs('pickles', exist_ok=True)

        # Plot directories
        self.plots_dir = 'plots'
        self.evaluation_dir = 'evaluation'

        # Plot paths
        self.class_distribution_plot = os.path.join(self.plots_dir, 'class_distribution.png')
        self.training_history_plot = os.path.join(self.plots_dir, 'training_history.png')
        self.confusion_matrix_plot = os.path.join(self.evaluation_dir, 'confusion_matrix.png')
        self.roc_curves_plot = os.path.join(self.evaluation_dir, 'roc_curves.png')
        self.precision_recall_plot = os.path.join(self.evaluation_dir, 'precision_recall_curves.png')
        self.feature_plots = {
            'signals': os.path.join(self.plots_dir, 'signals.png'),
            'fft': os.path.join(self.plots_dir, 'fft.png'),
            'fbank': os.path.join(self.plots_dir, 'fbank.png'),
            'mfccs': os.path.join(self.plots_dir, 'mfccs.png'),
            'spectrograms': os.path.join(self.plots_dir, 'spectrograms.png'),
            'metrics': os.path.join(self.plots_dir, 'performance_metrics.png')
        }

        # Classes
        self.classes = ['Swords', 'WildAnimals', 'Alarms']