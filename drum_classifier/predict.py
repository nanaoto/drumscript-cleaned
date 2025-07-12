# DrumScript/drum_classifier/predict_drum_events.py

import os
import numpy as np
import joblib
import json
import librosa
import tensorflow as tf # Required to load the Keras model
from tqdm import tqdm # For progress bar

# --- Configuration (must match model_trainer.py) ---
SAMPLE_RATE = 22050
SEGMENT_LENGTH_SECONDS = 0.2 # The length of segments your model was trained on
HOP_LENGTH_SECONDS = 0.1 # How much to move forward for the next segment (creates overlap)

# Define all UNIQUE drum types - MUST MATCH process_enst_dataset.py and model_trainer.py
ALL_DRUM_TYPES = sorted(['kick', 'snare', 'hi-hat', 'crash', 'ride', 'tom']) 

# --- Feature Extraction Helper (copy from model_trainer.py) ---
def _extract_features(audio_segment, sr, segment_length_seconds):
    """
    Extracts MFCC features from an audio segment (numpy array).
    Ensures the segment has the expected length, padding if necessary.
    """
    try:
        # Pad with zeros if the segment is shorter than expected
        if len(audio_segment) < int(sr * segment_length_seconds):
            pad_length = int(sr * segment_length_seconds) - len(audio_segment)
            audio_segment = np.pad(audio_segment, (0, pad_length), mode='constant')
        elif len(audio_segment) > int(sr * segment_length_seconds):
            # Trim if the segment is longer (shouldn't happen with proper slicing)
            audio_segment = audio_segment[:int(sr * segment_length_seconds)]

        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=40)
        mfccs = mfccs.T # Transpose to (time_frames, n_mfccs)
        return mfccs
    except Exception as e:
        # print(f"Error extracting features from segment: {e}") # Uncomment for verbose debugging
        return None

# --- Prediction Function (modified to take audio array) ---
def predict_drum_type_from_array(audio_array: np.ndarray, model, scaler, label_map):
    """
    Predicts multi-label drum types for a single audio segment (numpy array).
    """
    # 1. Extract features
    features = _extract_features(audio_array, SAMPLE_RATE, SEGMENT_LENGTH_SECONDS)
    
    if features is None:
        return [] # Return empty list if feature extraction failed

    # Ensure features have the correct shape (num_time_steps, num_mfccs)
    # The model expects a batch dimension: (1, num_time_steps, num_mfccs)
    # The time_steps calculation must match model_trainer's _extract_features
    expected_time_steps = int(SEGMENT_LENGTH_SECONDS * SAMPLE_RATE / 512) + 1
    if features.shape != (expected_time_steps, 40):
        # This could happen if the final segment is too short, even after padding.
        # For simplicity, we'll skip it if the shape is fundamentally wrong.
        # A more robust solution might involve further padding/resampling.
        return []

    # Reshape for scaler (flatten time_steps * n_mfcc)
    original_shape = features.shape
    features_reshaped_for_scaler = features.reshape(1, -1) # Reshape to (1, total_features)

    # Scale the features
    scaled_features = scaler.transform(features_reshaped_for_scaler)

    # Reshape back for the CNN model: (num_samples, time_steps, n_mfccs)
    scaled_features_for_cnn = scaled_features.reshape(1, original_shape[0], original_shape[1])

    # 2. Make prediction
    predictions = model.predict(scaled_features_for_cnn, verbose=0) # verbose=0 to suppress Keras output for each segment
    
    # 3. Interpret prediction (multi-label)
    # Apply a threshold (e.g., 0.5) to get binary predictions for each drum type
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)[0] # [0] to get rid of batch dimension
    
    predicted_drums = [
        drum_name for i, drum_name in enumerate(ALL_DRUM_TYPES)
        if binary_predictions[label_map[drum_name]] == 1 # Check if the corresponding label index is 1
    ]

    return predicted_drums

# --- New Function to Process Longer Audio Files ---
def process_long_audio_and_predict(audio_filepath: str, model, scaler, label_map):
    """
    Loads a longer audio file, segments it, and predicts drum types for each segment.
    """
    print(f"\nProcessing long audio file: {audio_filepath}")
    
    y, sr = librosa.load(audio_filepath, sr=SAMPLE_RATE, mono=True)
    
    segment_length_samples = int(SEGMENT_LENGTH_SECONDS * sr)
    hop_length_samples = int(HOP_LENGTH_SECONDS * sr)

    predictions_over_time = []

    # Iterate through the audio, segment by segment
    total_segments = int(np.ceil((len(y) - segment_length_samples + hop_length_samples) / hop_length_samples))
    if total_segments <= 0: # Handle very short files
        print("Audio file too short to create any segments.")
        return []

    for i in tqdm(range(0, len(y) - segment_length_samples + 1, hop_length_samples), total=total_segments, desc="Segmenting & Predicting"):
        start_sample = i
        end_sample = i + segment_length_samples
        
        # Extract segment
        segment = y[start_sample:end_sample]
        
        # Predict for the segment
        predicted_drums = predict_drum_type_from_array(segment, model, scaler, label_map)
        
        if predicted_drums: # Only record if drums were detected
            start_time = librosa.samples_to_time(start_sample, sr=sr)
            predictions_over_time.append({'time': start_time, 'drums': predicted_drums})
            
    if not predictions_over_time:
        print("No drum events detected in the entire audio file.")

    return predictions_over_time


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

    model_load_directory = os.path.join(project_root, "models")
    
    # Paths for your saved model components
    model_file = os.path.join(model_load_directory, "multi_label_drum_classifier_model.h5") 
    scaler_file = os.path.join(model_load_directory, "multi_label_scaler.joblib")
    label_map_file = os.path.join(model_load_directory, "multi_label_label_map.json")

    # Check if model files exist
    if not all(os.path.exists(f) for f in [model_file, scaler_file, label_map_file]):
        print("Error: One or more model files not found. Please ensure model_trainer.py was run successfully.")
        exit(1)

    # Load the trained model, scaler, and label map
    print("Loading trained model, scaler, and label map...")
    loaded_model = tf.keras.models.load_model(model_file)
    loaded_scaler = joblib.load(scaler_file)
    with open(label_map_file, 'r') as f:
        loaded_label_map = json.load(f)
    print("Model components loaded.")

    # --- Example 1: Predict single segment (as before) ---
    print("\n--- Single Segment Drum Prediction Example ---")
    # Replace with a valid path to an actual audio segment from ENST_processed
    single_segment_audio_path = os.path.join(project_root, "training_data", "ENST_processed", "1_rock-prog_125_beat_4-4.wav") 
    
    if not os.path.exists(single_segment_audio_path):
        print(f"Error: Single segment example audio file not found at {single_segment_audio_path}")
        print("Please replace '1_rock-prog_125_beat_4-4.wav' with a valid path to an actual segment for testing.")
    else:
        print(f"Predicting drum types for single segment: {single_segment_audio_path}")
        predicted_drums_single = predict_drum_type_from_array(
            librosa.load(single_segment_audio_path, sr=SAMPLE_RATE, mono=True)[0], 
            loaded_model, loaded_scaler, loaded_label_map
        )
        print(f"Predicted drum types: {predicted_drums_single}")

    # --- Example 2: Process a longer audio file ---
    print("\n--- Longer Audio File Prediction Example ---")
    # IMPORTANT: Replace this with the path to your actual longer audio file
    # This could be an MP3, WAV, etc. (librosa supports various formats)
    long_audio_filepath = os.path.join(project_root, "reference_audio", "test.mp3") 
    
    if not os.path.exists(long_audio_filepath):
        print(f"Error: Long audio file not found at {long_audio_filepath}")
        print("Please place your longer audio file (e.g., 'test.mp3') inside the 'DrumScrip/treference_audio/' directory, or update the path.")
    else:
        results = process_long_audio_and_predict(long_audio_filepath, loaded_model, loaded_scaler, loaded_label_map)
        
        if results:
            print("\n--- Detected Drum Events (Time-stamped) ---")
            for event in results:
                print(f"Time: {event['time']:.2f}s - Drums: {', '.join(event['drums'])}")
        else:
            print("\nNo drum events detected in the longer audio file.")
    print("-------------------------------------------------------------")