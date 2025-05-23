from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
from cfg import Config

def plot_signals(signals):
    num_signals = len(signals)
    num_rows = num_signals // 5 + (num_signals % 5 > 0)
    fig, axes = plt.subplots(nrows=num_rows, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5 * num_rows))
    fig.suptitle('Time Series', size=16)
    axes = axes.flatten()  # Flatten the axes array
    for i, ax in enumerate(axes):
        if i < num_signals:
            ax.set_title(list(signals.keys())[i])
            ax.plot(list(signals.values())[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.axis('off')

def plot_fft(fft):
    num_fft = len(fft)
    num_rows = num_fft // 5 + (num_fft % 5 > 0)
    fig, axes = plt.subplots(nrows=num_rows, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5 * num_rows))
    fig.suptitle('Fourier Transforms', size=16)
    axes = axes.flatten()  # Flatten the axes array
    for i, ax in enumerate(axes):
        if i < num_fft:
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            ax.set_title(list(fft.keys())[i])
            ax.plot(freq, Y)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.axis('off')

def plot_fbank(fbank):
    num_fbank = len(fbank)
    num_rows = num_fbank // 5 + (num_fbank % 5 > 0)
    fig, axes = plt.subplots(nrows=num_rows, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5 * num_rows))
    fig.suptitle('Filter Bank Coefficients', size=16)
    axes = axes.flatten()  # Flatten the axes array
    for i, ax in enumerate(axes):
        if i < num_fbank:
            ax.set_title(list(fbank.keys())[i])
            ax.imshow(list(fbank.values())[i],
                      cmap='hot', interpolation='nearest')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.axis('off')

def plot_mfccs(mfccs):
    num_mfccs = len(mfccs)
    num_rows = num_mfccs // 5 + (num_mfccs % 5 > 0)
    fig, axes = plt.subplots(nrows=num_rows, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5 * num_rows))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    axes = axes.flatten()  # Flatten the axes array
    for i, ax in enumerate(axes):
        if i < num_mfccs:
            ax.set_title(list(mfccs.keys())[i])
            ax.imshow(list(mfccs.values())[i],
                      cmap='hot', interpolation='nearest')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.axis('off')

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('wavfiles/' + f)
    df.at[f, 'length'] = signal.shape[0] / rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.savefig("class_distribution.png")  # Added this line
plt.show()

df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    wav_file = df[df.label == c].iloc[0, 0]
    signal, rate = librosa.load('wavfiles/' + wav_file, sr=48000)
    mask = envelope(signal, rate,0.0005)
    signals[c] = signal[mask]
    fft[c] = calc_fft(signal, rate)

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1200).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1200).T
    mfccs[c] = mel

plot_signals(signals)
plt.savefig("signals.png")  # Added this line
plt.show()

plot_fft(fft)
plt.savefig("fft.png")  # Added this line
plt.show()

plot_fbank(fbank)
plt.savefig("fbank.png")  # Added this line
plt.show()

plot_mfccs(mfccs)
plt.savefig("mfccs.png")  # Added this line
plt.show()

# Initialize config to get envelope threshold
config = Config()

# Create clean directory if it doesn't exist
os.makedirs('clean', exist_ok=True)

if len(os.listdir('clean')) == 0:
    print("Processing audio files with envelope threshold:", config.envelope_threshold)
    for f in tqdm(df.fname):
        signal, rate = librosa.load('wavfiles/' + f, sr=config.rate)
        mask = envelope(signal, rate, config.envelope_threshold)
        wavfile.write(filename='clean/' + f, rate=rate, data=signal[mask])