import os
import numpy as np
import joblib
import json
import librosa
import tensorflow as tf
import argparse

# --- Configuration (must match training configuration) ---
SAMPLE_RATE = 22050
SEGMENT_LENGTH_SECONDS = 0.2
# This must match the ALL_DRUM_TYPES list used in process_enst_dataset.py and model_trainer.py
ALL_DRUM_TYPES = sorted(['kick', 'snare', 'hi-hat', 'crash', 'ride', 'tom'])

# --- Load Trained Components ---
def load_model_components(models_dir: str):
    """
    Loads the trained Keras model, StandardScaler, and label map.
    """
    model_path = os.path.join(models_dir, "multi_label_drum_classifier_model.h5")
    scaler_path = os.path.join(models_dir, "multi_label_scaler.joblib")
    label_map_path = os.path.join(models_dir, "multi_label_label_map.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model first.")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Label map not found at {label_map_path}. Please train the model first.")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    print(f"Loading label map from: {label_map_path}")
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    return model, scaler, label_map

# --- Feature Extraction (must match model_trainer.py) ---
def _extract_features(audio_segment, sr, segment_length_seconds):
    """
    Extracts MFCC features from an audio segment.
    This function must be identical to the one used in model_trainer.py for consistency.
    """
    # Ensure the segment has the expected length, pad if necessary
    if len(audio_segment) < int(sr * segment_length_seconds):
        pad_length = int(sr * segment_length_seconds) - len(audio_segment)
        audio_segment = np.pad(audio_segment, (0, pad_length), mode='constant')
    elif len(audio_segment) > int(sr * segment_length_seconds):
        audio_segment = audio_segment[:int(sr * segment_length_seconds)]

    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=40)
    mfccs = mfccs.T # Transpose to (time_frames, n_mfccs)
    return mfccs

# --- Inference Function ---
def classify_drum_events(audio_path: str, model, scaler, label_map: dict,
                         sr: int = SAMPLE_RATE, segment_length_seconds: float = SEGMENT_LENGTH_SECONDS,
                         hop_length_seconds: float = 0.05,  # How much to slide the window, e.g., 50ms overlap
                         prediction_threshold: float = 0.5):
    """
    Classifies drum events in a given audio file using the trained multi-label model.
    """
    print(f"Classifying drum events in: {os.path.basename(audio_path)}")
    audio, actual_sr = librosa.load(audio_path, sr=sr, mono=True)
    if actual_sr != sr:
        print(f"  Warning: Audio {os.path.basename(audio_path)} resampled from {actual_sr} to {sr}")

    segment_length_samples = int(segment_length_seconds * sr)
    hop_length_samples = int(hop_length_seconds * sr)

    detected_events = []
    
    # Inverse map for easy lookup of drum names
    # Ensure drum_type_names is sorted by index to match model output order
    idx_to_drum_type = {idx: drum for drum, idx in label_map.items()}
    drum_type_names = [idx_to_drum_type[i] for i in sorted(idx_to_drum_type.keys())]


    # Iterate through the audio using a sliding window
    for i in range(0, len(audio) - segment_length_samples + 1, hop_length_samples):
        segment = audio[i : i + segment_length_samples]
        
        # Ensure segment is not too short if near the end
        if len(segment) < int(sr * 0.05): # Minimum 50ms segment to avoid errors
            continue

        features = _extract_features(segment, sr, segment_length_seconds)
        
        # Check if features were extracted successfully and have correct shape
        if features is None or features.size == 0:
            continue
        if features.shape[0] == 0 or features.shape[1] == 0:
            continue

        # Reshape features for scaler and model: (1, time_frames * n_mfcc) for scaler
        # Then (1, time_frames, n_mfcc) for model
        original_features_shape = features.shape
        features_reshaped_for_scaler = features.reshape(1, -1) # Flatten for scaler
        
        # Apply the same scaler used during training
        features_scaled = scaler.transform(features_reshaped_for_scaler)
        
        # Reshape back to (1, time_frames, n_mfcc) for the CNN model
        features_for_model = features_scaled.reshape(1, original_features_shape[0], original_features_shape[1])

        # Make prediction
        predictions = model.predict(features_for_model, verbose=0)[0] # Get probabilities for this single segment

        # Convert probabilities to binary predictions based on threshold
        binary_predictions = (predictions > prediction_threshold).astype(int)

        # Get the drum types that were predicted as present
        predicted_drum_types = [
            drum_type_names[j] for j, is_present in enumerate(binary_predictions) if is_present == 1
        ]
        
        # If any drums were detected in this segment, record the event
        if predicted_drum_types:
            onset_time = (i + segment_length_samples / 2) / sr # Center time of the segment
            detected_events.append({'time': onset_time, 'drums': predicted_drum_types, 'probabilities': predictions.tolist()})

    print(f"Detected {len(detected_events)} potential drum events.")
    return detected_events

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrumScript: Classify drum events in audio using a trained multi-label CNN.")
    parser.add_argument("audio_file", type=str, nargs='?', 
                        help="Path to the input drum audio file (e.g., .wav). If not provided, an example from ENST_processed will be used.",
                        default=None) # Make it optional

    args = parser.parse_args()

    # Determine project root dynamically
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir)) # Go up from drum_classifier/

    models_directory = os.path.join(project_root, "models")
    
    # --- Step 1: Load Trained Model and Components ---
    try:
        loaded_model, loaded_scaler, loaded_label_map = load_model_components(models_directory)
        # Verify the label map (optional, but good for debugging)
        print(f"Loaded drum types (in order): {list(loaded_label_map.keys())}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run model_trainer.py successfully to generate the model files.")
        exit(1)
    
    print("\n--- Model Loaded Successfully ---")

    # --- Step 2: Define an audio file to test ---
    test_audio_path = args.audio_file
    if test_audio_path is None:
        # If no audio file is provided via argument, try to find an example
        processed_dir = os.path.join(project_root, "training_data", "ENST_processed")
        wav_files = [f for f in os.listdir(processed_dir) if f.endswith('.wav')]
        if wav_files:
            # Use the first WAV file found as an example
            test_audio_path = os.path.join(processed_dir, wav_files[0])
            print(f"No audio file provided. Using example: {os.path.basename(test_audio_path)}")
        else:
            print(f"\nERROR: No audio file provided via argument and no WAV files found in '{processed_dir}' for example.")
            print("Please provide a valid path to an audio file for inference (e.g., python3 -m drum_classifier.main your_audio.wav).")
            print("Or ensure `training_data/ENST_processed/` contains `.wav` files after running `process_enst_dataset.py`.")
            exit(1)

    if not os.path.exists(test_audio_path):
        print(f"\nERROR: Provided audio file not found at: {test_audio_path}")
        exit(1)

    # --- Step 3: Perform Inference ---
    print(f"\n--- Starting Inference on: {os.path.basename(test_audio_path)} ---")
    detected_drum_events = classify_drum_events(
        test_audio_path,
        loaded_model,
        loaded_scaler,
        loaded_label_map,
        sr=SAMPLE_RATE,
        segment_length_seconds=SEGMENT_LENGTH_SECONDS,
        hop_length_seconds=0.05, # Slide window by 50ms
        prediction_threshold=0.5
    )

    # --- Step 4: Display Results ---
    print("\n--- Detected Drum Events ---")
    if detected_drum_events:
        for event in detected_drum_events:
            print(f"Time: {event['time']:.2f}s | Drums: {', '.join(event['drums'])}")
            # Optional: Print raw probabilities for debugging
            # print(f"  Raw Probabilities: {event['probabilities']}")
    else:
        print("No drum events detected in the provided audio file.")

    print("\n--- Inference Complete ---")
    print("\nNext Steps:")
    print("1. You can run this script with your own audio files: `python3 -m drum_classifier.main path/to/your/audio.wav`")
    print("2. The `hop_length_seconds` parameter in `classify_drum_events` determines how often the model makes a prediction. A smaller value (like 0.01-0.02s) might capture more subtle events but will increase processing time. A larger value (like 0.1s) is faster but might miss short events.")
    print("3. The `prediction_threshold` (default 0.5) can be adjusted if you want more or fewer detections (lower threshold = more detections, higher = fewer).")
    print("4. For generating drum sheet music, you will need to adapt the `notation_generator` module to accept these multi-label `detected_events` and map them correctly to notation.")