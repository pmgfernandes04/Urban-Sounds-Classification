import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import json

# Constants
SAMPLE_RATE = 22050  # Standard sampling rate for audio processing
DURATION = 4         # Maximum duration of audio clips in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_MFCC = 40        # Number of MFCC coefficients to extract
N_FFT = 2048         # Length of the FFT window
HOP_LENGTH = 512     # Number of samples between successive frames

# Get the default soundata data home directory
data_home = "/home/eduardo/Downloads/UrbanSound8K"
# This is typically '~/.sound_datasets/urbansound8k'

# Paths to audio files and metadata
metadata_path = os.path.join(data_home, 'metadata', 'UrbanSound8K.csv')
audio_dataset_path = os.path.join(data_home, 'audio')

# Load the metadata
metadata = pd.read_csv(metadata_path)

# Function to extract MFCC features from an audio file
def extract_features(file_name, sample_rate=SAMPLE_RATE, duration=DURATION, num_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    try:
        # Load audio file
        audio, sr = librosa.load(file_name, sr=sample_rate, duration=duration)

        # Pad or truncate audio to ensure consistent length
        if len(audio) < SAMPLES_PER_TRACK:
            padding = SAMPLES_PER_TRACK - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            audio = audio[:SAMPLES_PER_TRACK]

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=num_mfcc,
            n_fft=n_fft, hop_length=hop_length
        )

        # Transpose to get time steps on the first dimension
        mfccs = mfccs.T  # Shape: (time_steps, num_mfcc)

        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}. Error: {e}")
        return None

# Lists to hold features and labels
X = []
y = []

# Iterate through each row in the metadata and extract features
print("Extracting features from audio files...")
for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
    fold = row['fold']
    file_name = row['slice_file_name']
    class_name = row['class']
    class_id = row['classID']

    # Get the full file path
    file_path = os.path.join(audio_dataset_path, f'fold{fold}', file_name)

    # Extract features
    features = extract_features(file_path)

    if features is not None:
        X.append(features)
        y.append(class_id)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Save the preprocessed data
np.save('X.npy', X)
np.save('y.npy', y)

# Create and save the class label mapping
class_label_mapping = metadata[['classID', 'class']].drop_duplicates().set_index('classID')['class'].to_dict()
with open('class_label_mapping.json', 'w') as f:
    json.dump(class_label_mapping, f)

print(f"Preprocessing complete. Features saved to 'X.npy', labels saved to 'y.npy'.")
print(f"Feature array shape: {X.shape}")
print(f"Labels array shape: {y.shape}")
