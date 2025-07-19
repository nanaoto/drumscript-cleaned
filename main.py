# DrumScript/main.py

import os
import numpy as np
import joblib
import json
import librosa
import tensorflow as tf
import argparse
from notation_generator import score_builder
# Import the actual feature extraction logic
from audio_processor.feature_extractor import extract_features

print(f'-------------------------------------------------------------')
# --- Configuration (must match training configuration) ---
SAMPLE_RATE = 22050
SEGMENT_LENGTH_SECONDS = 0.2

# Corrected based on expected CNN input shape (None, 9, 40)
TARGET_N_FRAMES = 9 # Model expects 9 time steps
FEATURES_PER_FRAME = 40 # Model expects 40 features per time step (e.g., 20 MFCCs + 20 Delta MFCCs)

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

# --- Feature Extraction for Prediction (uses the imported extract_features) ---
def _prepare_features_for_prediction(audio_segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Calls the main extract_features function, ensures a fixed number of frames (TARGET_N_FRAMES),
    concatenates features per frame, and flattens them into a single vector of 360 features.
    The order of features in the flattened vector MUST match the order used during training.
    """
    features_dict = extract_features(audio_segment, sr)

    # List to hold processed feature arrays for each frame
    # Each element in this list will be a (FEATURES_PER_FRAME,) array
    features_per_frame_list = []

    # Combine all features for each frame
    # The order here is CRUCIAL and must match how features were combined per frame during training.
    # Assuming MFCCs (20), Delta MFCCs (20) = 40 features per frame, and other features were not used in this specific 40.
    # If other features (spectral_centroid, rolloff, zcr, rms) were also part of the 40,
    # then they should be added here, and FEATURES_PER_FRAME adjusted.
    # Given (9, 40) where 20+20=40, it implies only MFCCs and Delta MFCCs contribute to the 40.

    # Determine the minimum number of frames available across all features
    # This is important because librosa.feature.delta can return one less frame than librosa.feature.mfcc
    # The number of frames is the second dimension of the feature array
    n_actual_frames = min([f.shape[1] for f in features_dict.values() if f.ndim == 2]) if features_dict else 0

    # Handle cases where audio_segment is empty or no frames were extracted
    if n_actual_frames == 0:
        # If no frames, create an array of zeros with the target shape (TARGET_N_FRAMES, FEATURES_PER_FRAME)
        combined_features_across_frames = np.zeros((TARGET_N_FRAMES, FEATURES_PER_FRAME))
    else:
        # Iterate over each frame and concatenate features for that frame
        # Ensure 'mfccs' and 'delta_mfccs' are indeed 20 features each
        for i in range(n_actual_frames):
            frame_features = np.concatenate([
                features_dict['mfccs'][:, i],         # 20 features
                features_dict['delta_mfccs'][:, i]    # 20 features
                # If other 1-dim features were part of the 40 per frame, add them here
                # E.g., features_dict['spectral_centroid'][:, i],
            ])
            features_per_frame_list.append(frame_features)
        
        # Stack the features from all frames into a 2D array
        combined_features_across_frames = np.array(features_per_frame_list) # Shape: (n_actual_frames, FEATURES_PER_FRAME)

        # Pad or truncate to TARGET_N_FRAMES (9 frames)
        if combined_features_across_frames.shape[0] > TARGET_N_FRAMES:
            # Truncate if too many frames
            combined_features_across_frames = combined_features_across_frames[:TARGET_N_FRAMES, :]
        elif combined_features_across_frames.shape[0] < TARGET_N_FRAMES:
            # Pad with zeros if too few frames
            padding_needed = TARGET_N_FRAMES - combined_features_across_frames.shape[0]
            # Pad along the time axis (first axis)
            combined_features_across_frames = np.pad(combined_features_across_frames, 
                                                     ((0, padding_needed), (0, 0)), 
                                                     'constant')
        # Else: Exactly TARGET_N_FRAMES, no action needed

    # Flatten the 2D array (9, 40) into a 1D array (360,) for StandardScaler
    feature_vector_flat = combined_features_across_frames.flatten()
    
    # Assert the final expected shape for debugging
    expected_len = TARGET_N_FRAMES * FEATURES_PER_FRAME
    if feature_vector_flat.shape[0] != expected_len:
        raise ValueError(f"Feature vector has {feature_vector_flat.shape[0]} features, but expected {expected_len} after flattening. Check feature extraction and concatenation logic.")

    return feature_vector_flat

# --- Prediction Helper ---
def predict_drum_type_from_array(audio_segment_array: np.ndarray, loaded_model, loaded_scaler, loaded_label_map) -> tuple[list[str], list[float]]:
    """
    Predicts drum type(s) from a single audio segment (numpy array).
    """
    # Prepare features using the aligned extraction function (outputs 1D, 360 features)
    feature_vector_flat = _prepare_features_for_prediction(audio_segment_array, SAMPLE_RATE)

    # Scale features. The scaler expects a 2D array (1 sample, N features).
    scaled_features_flat = loaded_scaler.transform(feature_vector_flat.reshape(1, -1))

    # Reshape the scaled features for the CNN model: (batch_size, time_steps, features_per_time_step)
    # The CNN expects (1, 9, 40)
    scaled_features_cnn_input = scaled_features_flat.reshape(1, TARGET_N_FRAMES, FEATURES_PER_FRAME)

    # Predict probabilities
    predictions = loaded_model.predict(scaled_features_cnn_input)[0]

    # Map probabilities to drum types
    predicted_drum_types = []
    # Ensure label_map keys are sorted consistently with how the model's output labels were ordered
    labels = sorted(loaded_label_map.keys())
    for i, prob in enumerate(predictions):
        if prob > 0.5: # Example threshold
            predicted_drum_types.append(labels[i])
    return predicted_drum_types, predictions.tolist() # Return raw predictions as list


# The rest of main.py (main function, classify_drum_events, argparse, etc.)
# should remain as it was in the previous correct version, as the changes
# were focused on the _prepare_features_for_prediction and predict_drum_type_from_array.

def classify_drum_events(audio_filepath: str, loaded_model, loaded_scaler, loaded_label_map, sr: int, segment_length_seconds: float, hop_length_seconds: float, prediction_threshold: float) -> list[dict]:
    """
    Processes a long audio file, segments it, and classifies drum events.
    """
    y, sr = librosa.load(audio_filepath, sr=sr, mono=True)
    
    segment_length_samples = int(segment_length_seconds * sr)
    hop_length_samples = int(hop_length_seconds * sr)

    detected_events = []

    for i in range(0, len(y) - segment_length_samples + 1, hop_length_samples):
        segment = y[i : i + segment_length_samples]
        
        # Predict drum types for the current segment
        predicted_drums, probabilities = predict_drum_type_from_array(
            segment, loaded_model, loaded_scaler, loaded_label_map
        )
        
        if predicted_drums:
            event_time = i / sr
            detected_events.append({
                'time': event_time,
                'drums': predicted_drums,
                'probabilities': probabilities
            })
    return detected_events


def main():
    parser = argparse.ArgumentParser(description='Process an audio file to detect drum events and generate drum sheet music.')
    parser.add_argument('audio_file', type=str, help='Path to the input audio file (e.g., .wav, .mp3).')
    parser.add_argument('--output_pdf', type=str, default='drum_score.pdf',
                        help='Path to the output PDF file for the drum sheet music. Defaults to "drum_score.pdf".')
    parser.add_argument('--tempo', type=int, default=120,
                        help='The tempo (BPM) to use for quantizing drum events in the sheet music. Defaults to 120 BPM.')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory containing the trained model, scaler, and label map. Defaults to "models".')

    args = parser.parse_args()

    # --- Step 1: Load Trained Components ---
    models_dir = os.path.join(os.path.dirname(__file__), args.models_dir)
    loaded_model, loaded_scaler, loaded_label_map = load_model_components(models_dir)

    # --- Step 2: Validate Audio File Path ---
    test_audio_path = args.audio_file
    if not os.path.exists(test_audio_path):
        print(f"\nERROR: Provided audio file not found at: {test_audio_path}")
        exit(1)

    # --- Step 3: Perform Inference ---
    print(f"\n--- Starting Inference on: {os.path.basename(test_audio_path)} ---")
    detected_drum_events = classify_drum_events(
        audio_filepath=test_audio_path,
        loaded_model=loaded_model,
        loaded_scaler=loaded_scaler,
        loaded_label_map=loaded_label_map,
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
    else:
        print("No drum events detected in the provided audio file.")

    print("\n--- Inference Complete ---\n")

    # --- Step 5: Generate Drum Sheet Music and Export to PDF ---
    if detected_drum_events: # Only try to generate PDF if events were detected
        print("--- Generating Drum Sheet Music PDF ---")
        score_builder.build_and_export_drum_score(
            detected_drum_events,
            #output_pdf_path=args.output_pdf,
            output_filepath=args.output_pdf,
            #audio_filepath=args.audio_file, # Pass the audio file path
            tempo=args.tempo # Pass the tempo
        )
        print(f"\nDrum sheet music successfully generated to: {args.output_pdf}")
    else:
        print("No drum events detected, so no sheet music PDF will be generated.")
        print(f'-------------------------------------------------------------')


if __name__ == "__main__":
    main()