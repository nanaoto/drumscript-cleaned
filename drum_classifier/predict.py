# DrumScript/drum_classifier/predict_drum_events.py

import os
import numpy as np
import joblib
import json
import librosa
import tensorflow as tf # Required to load the Keras model

# --- Configuration (must match model_trainer.py) ---
SAMPLE_RATE = 22050
SEGMENT_LENGTH_SECONDS = 0.2
# Define all UNIQUE drum types - MUST MATCH process_enst_dataset.py and model_trainer.py
ALL_DRUM_TYPES = sorted(['kick', 'snare', 'hi-hat', 'crash', 'ride', 'tom']) 

# --- Feature Extraction Helper (copy from model_trainer.py) ---
def _extract_features(audio_path, sr, segment_length_seconds):
    """
    Extracts MFCC features from an audio segment.
    """
    try:
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        if len(audio) < int(sr * segment_length_seconds):
            pad_length = int(sr * segment_length_seconds) - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        elif len(audio) > int(sr * segment_length_seconds):
            audio = audio[:int(sr * segment_length_seconds)]

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs = mfccs.T # Transpose to (time_frames, n_mfccs)
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

# --- Prediction Function ---
def predict_drum_type(audio_path: str, model, scaler, label_map):
    """
    Predicts multi-label drum types for a single audio segment.
    """
    # 1. Extract features
    features = _extract_features(audio_path, SAMPLE_RATE, SEGMENT_LENGTH_SECONDS)
    
    if features is None:
        return "Feature extraction failed."

    # Ensure features have the correct shape (num_time_steps, num_mfccs)
    # The model expects a batch dimension: (1, num_time_steps, num_mfccs)
    if features.shape != (int(SEGMENT_LENGTH_SECONDS * SAMPLE_RATE / 512) + 1, 40):
        print(f"Warning: Expected feature shape (9, 40) but got {features.shape}. Model prediction might be inaccurate.")
        # You might need to resize or pad 'features' if its shape is inconsistent
        # For now, let's assume it's correct or will lead to an error.
        
    # Reshape for scaler (flatten time_steps * n_mfcc)
    original_shape = features.shape
    features_reshaped_for_scaler = features.reshape(1, -1) # Reshape to (1, total_features)

    # Scale the features
    scaled_features = scaler.transform(features_reshaped_for_scaler)

    # Reshape back for the CNN model: (num_samples, time_steps, n_mfccs)
    scaled_features_for_cnn = scaled_features.reshape(1, original_shape[0], original_shape[1])

    # 2. Make prediction
    predictions = model.predict(scaled_features_for_cnn)
    
    # 3. Interpret prediction (multi-label)
    # Apply a threshold (e.g., 0.5) to get binary predictions for each drum type
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)[0] # [0] to get rid of batch dimension
    
    predicted_drums = [
        drum_name for i, drum_name in enumerate(ALL_DRUM_TYPES)
        if binary_predictions[label_map[drum_name]] == 1 # Check if the corresponding label index is 1
    ]

    if not predicted_drums:
        return "No drums detected above threshold."
    return predicted_drums


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

    model_load_directory = os.path.join(project_root, "models")
    
    # Update these paths if you change the saving extension in model_trainer.py
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

    print("\n--- Drum Prediction Example ---")
    # --- IMPORTANT: Provide a path to an actual drum audio segment here ---
    # You can pick one from your training_data/ENST_processed/ directory for testing
    #example_audio_path = os.path.join(project_root, "training_data", "ENST_processed", "your_audio_segment.wav")
    example_audio_path = os.path.join(project_root, "test_audio", "1_rock-prog_125_beat_4-4.wav")

    
    if not os.path.exists(example_audio_path):
        print(f"Error: Example audio file not found at {example_audio_path}")
        print("Please replace 'your_audio_segment.wav' with a valid path to an actual audio segment.")
    else:
        print(f"Predicting drum types for: {example_audio_path}")
        predicted_drums = predict_drum_type(example_audio_path, loaded_model, loaded_scaler, loaded_label_map)
        print(f"Predicted drum types: {predicted_drums}")