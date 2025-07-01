import os
import json
import numpy as np
import librosa
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
mixed_audio_folder = "C:/Users/Desktop/augmented_audios/NEW FOLDER___UPDATED"
json_path = "C:/Users/PycharmProjects/fakeaudiodetection/MIXED#####AUDIOFILES.json"
output_dir = "C:/Users/PycharmProjects/fakeaudiodetection/processed_features"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load ground truth JSON
start_time = time.time()
with open(json_path, "r") as f:
    ground_truth = json.load(f)
logger.info(f"JSON loading time: {time.time() - start_time:.2f} seconds")

# Get all mixed audio files
all_audio_files = sorted(os.listdir(mixed_audio_folder))  # Sorted to ensure consistent split
selected_audio_files = all_audio_files[:5000]  # Ensure we have exactly 5000 files
train_test_files = selected_audio_files[:4000]  # First 4000 for training/testing
unseen_files = selected_audio_files[4000:]  # Last 1000 for evaluation
logger.info(
    f"Total files: {len(selected_audio_files)}, Train/Test: {len(train_test_files)}, Unseen: {len(unseen_files)}")


# Function to extract features from a 2-second segment
def extract_features(y, sr):
    segment_length = sr * 2  # 2 seconds at 16000 Hz
    n_fft = min(512, len(y) // 2)

    # Existing features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft), axis=1)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft), axis=1)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1) if len(
        y) >= 512 else np.zeros(6)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y), axis=1)

    # Additional features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rms = np.mean(librosa.feature.rms(y=y))

    # Concatenate all features
    features = np.concatenate([
        mfcc, mel_spectrogram, chroma, spectral_contrast, tonnetz, zcr,
        [spectral_centroid], [spectral_rolloff], [spectral_flatness], [rms]
    ])
    return features


# Function to process a mixed audio file and extract segment features
def process_audio_file(file_name, mixed_audio_folder, ground_truth, json_keys):
    file_path = os.path.join(mixed_audio_folder, file_name)
    try:
        y, sr = librosa.load(file_path, sr=16000)
        segment_length = sr * 2  # 2 seconds
        num_segments = len(y) // segment_length  # Should be 5 for 10-second files

        segment_features = []
        segment_labels = []

        file_name_cleaned = file_name.strip().lower()
        if file_name_cleaned in json_keys:
            segments = ground_truth[file_name]["segments"][:num_segments]  # Take up to 5 segments
            for i in range(num_segments):
                segment_start = i * segment_length
                segment_end = (i + 1) * segment_length
                segment = y[segment_start:segment_end]
                features = extract_features(segment, sr)
                segment_features.append(features)
                segment_labels.append(segments[i])

        return np.array(segment_features), np.array(segment_labels)
    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")
        return None, None


# Process and save training/testing features
start_time = time.time()
train_test_features = []
train_test_labels = []
train_test_file_mapping = []  # Track which segments belong to which file
current_segment_idx = 0

for idx, file_name in enumerate(train_test_files, start=1):
    features, labels = process_audio_file(file_name, mixed_audio_folder, ground_truth,
                                          set(map(str.lower, ground_truth.keys())))
    if features is not None and labels is not None:
        num_segments = len(features)
        train_test_features.append(features)
        train_test_labels.append(labels)
        # Store mapping of segments to file
        for _ in range(num_segments):
            train_test_file_mapping.append((file_name, current_segment_idx))
            current_segment_idx += 1
    if idx % 100 == 0:
        logger.info(f"Processed {idx}/4000 train/test files")

# Save to .npy files
train_test_features = np.concatenate(train_test_features, axis=0)
train_test_labels = np.concatenate(train_test_labels, axis=0)
np.save(os.path.join(output_dir, "1111111train_test_features.npy"), train_test_features)
np.save(os.path.join(output_dir, "1111111train_test_labels.npy"), train_test_labels)
np.save(os.path.join(output_dir, "1111111train_test_file_mapping.npy"), np.array(train_test_file_mapping, dtype=object))
logger.info(f"Train/Test features shape: {train_test_features.shape}, labels shape: {train_test_labels.shape}")
logger.info(f"Train/Test feature extraction time: {time.time() - start_time:.2f} seconds")

# Process and save unseen features
start_time = time.time()
unseen_features = []
unseen_labels = []
unseen_file_mapping = []  # Track which segments belong to which file
current_segment_idx = 0

for idx, file_name in enumerate(unseen_files, start=1):
    features, labels = process_audio_file(file_name, mixed_audio_folder, ground_truth,
                                          set(map(str.lower, ground_truth.keys())))
    if features is not None and labels is not None:
        num_segments = len(features)
        unseen_features.append(features)
        unseen_labels.append(labels)
        # Store mapping of segments to file
        for _ in range(num_segments):
            unseen_file_mapping.append((file_name, current_segment_idx))
            current_segment_idx += 1
    if idx % 100 == 0:
        logger.info(f"Processed {idx}/1000 unseen files")

# Save to .npy files
unseen_features = np.concatenate(unseen_features, axis=0)
unseen_labels = np.concatenate(unseen_labels, axis=0)
np.save(os.path.join(output_dir, "1111111unseen_features.npy"), unseen_features)
np.save(os.path.join(output_dir, "1111111unseen_labels.npy"), unseen_labels)
np.save(os.path.join(output_dir, "1111111unseen_file_mapping.npy"), np.array(unseen_file_mapping, dtype=object))
logger.info(f"Unseen features shape: {unseen_features.shape}, labels shape: {unseen_labels.shape}")
logger.info(f"Unseen feature extraction time: {time.time() - start_time:.2f} seconds")
