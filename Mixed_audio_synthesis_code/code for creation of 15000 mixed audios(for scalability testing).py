import os
import json
import numpy as np
import librosa
import random
import soundfile as sf

# Paths
REAL_AUDIO_FOLDER = "C:/Users/91826/Desktop/training/real"
FAKE_AUDIO_FOLDER = "C:/Users/91826/Desktop/training/fake"
ORIGINAL_MIXED_AUDIO_FOLDER = "C:/Users/91826/Desktop/augmented_audios/NEW FOLDER___UPDATED"
NEW_MIXED_AUDIO_FOLDER = "C:/Users/91826/Desktop/augmented_audios/15000 files mixed"  # Replace with your specified path
NEW_JSON_FILE_PATH = os.path.join(NEW_MIXED_AUDIO_FOLDER, "NEW_MIXED_AUDIOFILES_15000.json")

# Create output directory if it doesn't exist
if not os.path.exists(NEW_MIXED_AUDIO_FOLDER):
    os.makedirs(NEW_MIXED_AUDIO_FOLDER)

# Load source audio files
real_files = [f for f in os.listdir(REAL_AUDIO_FOLDER) if f.endswith(".wav")]
fake_files = [f for f in os.listdir(FAKE_AUDIO_FOLDER) if f.endswith(".wav")]

if not real_files or not fake_files:
    print("❌ ERROR: No real or fake audio files found. Please check the source folders.")
    exit(1)

# Parameters
SAMPLE_RATE = 22050
SEGMENT_LENGTH = 2
TOTAL_DURATION = 10
NUM_SEGMENTS = 5
NUM_NEW_FILES = 15000  # Total number of new mixed files to create
FIXED_PERCENTAGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_FILES_PER_CYCLE = 5000  # Each cycle mimics the original 5000-file structure (4920 fixed + 80 random)
NUM_CYCLES = NUM_NEW_FILES // NUM_FILES_PER_CYCLE  # 15,000 / 5,000 = 3 cycles

# Function to load and trim/pad audio to 2 seconds
def load_and_adjust_audio(file_path, target_length):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    target_samples = int(target_length * sr)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    else:
        audio = audio[:target_samples]
    return audio

# Create mixed audio and metadata
mixed_metadata = {}
file_count = 5000  # Start from 5000 since original script created 0 to 4999

for cycle in range(NUM_CYCLES):
    # Create 4920 files with fixed percentages per cycle
    for i in range(4920):
        filename = f"MIXED__{file_count:04d}.wav"
        file_path = os.path.join(NEW_MIXED_AUDIO_FOLDER, filename)

        # Choose a fixed real_percentage
        real_percentage = random.choice(FIXED_PERCENTAGES)
        fake_percentage = 1.0 - real_percentage

        # Calculate number of real and fake segments
        num_real_segments = int(NUM_SEGMENTS * real_percentage)
        num_fake_segments = NUM_SEGMENTS - num_real_segments
        segment_labels = [1] * num_real_segments + [0] * num_fake_segments

        # Randomize the order of segments
        random.shuffle(segment_labels)

        # Mix audio by concatenating real and fake segments in random order
        mixed_audio = np.array([])
        for label in segment_labels:
            if label == 1:  # Real segment
                source_file = random.choice(real_files)
                audio = load_and_adjust_audio(os.path.join(REAL_AUDIO_FOLDER, source_file), SEGMENT_LENGTH)
            else:  # Fake segment
                source_file = random.choice(fake_files)
                audio = load_and_adjust_audio(os.path.join(FAKE_AUDIO_FOLDER, source_file), SEGMENT_LENGTH)
            mixed_audio = np.concatenate((mixed_audio, audio))

        # Ensure total length is 10 seconds
        if len(mixed_audio) > int(TOTAL_DURATION * SAMPLE_RATE):
            mixed_audio = mixed_audio[:int(TOTAL_DURATION * SAMPLE_RATE)]
        elif len(mixed_audio) < int(TOTAL_DURATION * SAMPLE_RATE):
            mixed_audio = np.pad(mixed_audio, (0, int(TOTAL_DURATION * SAMPLE_RATE) - len(mixed_audio)), mode='constant')

        # Save the mixed audio file
        sf.write(file_path, mixed_audio, SAMPLE_RATE)
        print(f"Saved mixed audio: {filename}")

        # Store metadata
        mixed_metadata[filename] = {
            "real_percentage": real_percentage,
            "fake_percentage": fake_percentage,
            "segments": segment_labels
        }

        file_count += 1

    # Create 80 files with randomized percentages per cycle
    for i in range(80):
        filename = f"MIXED__{file_count:04d}.wav"
        file_path = os.path.join(NEW_MIXED_AUDIO_FOLDER, filename)

        # Choose a random real_percentage between 0 and 1
        real_percentage = random.uniform(0, 1)
        fake_percentage = 1.0 - real_percentage

        # Calculate number of real and fake segments
        num_real_segments = int(NUM_SEGMENTS * real_percentage)
        num_fake_segments = NUM_SEGMENTS - num_real_segments
        if num_real_segments + num_fake_segments < NUM_SEGMENTS:
            num_real_segments += 1
        segment_labels = [1] * num_real_segments + [0] * num_fake_segments
        segment_labels = segment_labels[:NUM_SEGMENTS]

        # Randomize the order of segments
        random.shuffle(segment_labels)

        # Mix audio by concatenating real and fake segments in random order
        mixed_audio = np.array([])
        for label in segment_labels:
            if label == 1:  # Real segment
                source_file = random.choice(real_files)
                audio = load_and_adjust_audio(os.path.join(REAL_AUDIO_FOLDER, source_file), SEGMENT_LENGTH)
            else:  # Fake segment
                source_file = random.choice(fake_files)
                audio = load_and_adjust_audio(os.path.join(FAKE_AUDIO_FOLDER, source_file), SEGMENT_LENGTH)
            mixed_audio = np.concatenate((mixed_audio, audio))

        # Ensure total length is 10 seconds
        if len(mixed_audio) > int(TOTAL_DURATION * SAMPLE_RATE):
            mixed_audio = mixed_audio[:int(TOTAL_DURATION * SAMPLE_RATE)]
        elif len(mixed_audio) < int(TOTAL_DURATION * SAMPLE_RATE):
            mixed_audio = np.pad(mixed_audio, (0, int(TOTAL_DURATION * SAMPLE_RATE) - len(mixed_audio)), mode='constant')

        # Save the mixed audio file
        sf.write(file_path, mixed_audio, SAMPLE_RATE)
        print(f"Saved mixed audio: {filename}")

        # Store metadata
        mixed_metadata[filename] = {
            "real_percentage": real_percentage,
            "fake_percentage": fake_percentage,
            "segments": segment_labels
        }

        file_count += 1

# Save metadata to JSON file
with open(NEW_JSON_FILE_PATH, "w") as json_file:
    json.dump(mixed_metadata, json_file, indent=2)
print(f"Metadata saved to {NEW_JSON_FILE_PATH}")

# Verify a few entries
print("\nSample metadata entries:")
for key in list(mixed_metadata.keys())[:5]:
    print(f"{key}: {mixed_metadata[key]}")