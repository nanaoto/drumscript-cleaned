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

print(f'#-------------------------------------------------------------')
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

# --- Feature Extraction for Prediction (uses the imported extract_features) ---
def _prepare_features_for_prediction(audio_segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Calls the main extract_features function and flattens its output into a single vector.
    The order of features in the flattened vector MUST match the order used during training.
    """
    features_dict = extract_features(audio_segment, sr)

    # The order here is CRUCIAL and must match how features were concatenated during training.
    # Based on feature_extractor.py, the standard order is MFCCs first, then other features.
    feature_vector = np.concatenate([
        features_dict['mfccs'],
        features_dict['spectral_centroid'],
        features_dict['spectral_rolloff'],
        features_dict['zero_crossing_rate'],
        features_dict['rms']
        # If 'chroma' or other features were used in training, they must be added here
        # and extract_features must also be updated to generate them.
    ])
    return feature_vector

# --- Prediction Helper (adapted to use _prepare_features_for_prediction) ---
def predict_drum_type_from_array(audio_segment_array: np.ndarray, loaded_model, loaded_scaler, loaded_label_map) -> tuple[list[str], list[float]]:
    """
    Predicts drum type(s) from a single audio segment (numpy array).
    """
    # Prepare features using the aligned extraction function
    feature_vector = _prepare_features_for_prediction(audio_segment_array, SAMPLE_RATE)

    # Scale features. The scaler expects a 2D array (1 sample, N features).
    scaled_features = loaded_scaler.transform(feature_vector.reshape(1, -1))

    # Predict probabilities
    predictions = loaded_model.predict(scaled_features)[0]

    # Map probabilities to drum types
    predicted_drum_types = []
    # Ensure label_map keys are sorted consistently with how the model's output labels were ordered
    labels = sorted(loaded_label_map.keys())
    for i, prob in enumerate(predictions):
        if prob > 0.5: # Example threshold
            predicted_drum_types.append(labels[i])
    return predicted_drum_types, predictions.tolist() # Return raw predictions as list

# --- Process Long Audio and Classify Events ---
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
            output_pdf_path=args.output_pdf,
            audio_filepath=args.audio_file, # Pass the audio file path
            tempo=args.tempo # Pass the tempo
        )
        print(f"\nDrum sheet music successfully generated to: {args.output_pdf}")
    else:
        print("No drum events detected, so no sheet music PDF will be generated.")


if __name__ == "__main__":
    main()