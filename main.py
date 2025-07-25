# DrumScript/main.py

import argparse
import os
import sys
import numpy as np
import tensorflow as tf # Added for Keras model loading
import joblib           # Added for scaler loading
import json             # Added for label map loading

# No need for sys.path manipulation here if running as 'python -m DrumScript.main'
# or if DrumScript is installed as a package (e.g., via 'uv pip install -e .').

try:
    from audio_processor import audio_loader, onset_detector, feature_extractor
    # Import the DrumClassifier class directly from its module (if needed for other tasks, but for prediction, we'll use Keras directly)
    # from drum_classifier.drum_model import DrumClassifier # Not strictly needed if handling Keras prediction directly
    from notation_generator import score_builder, pdf_exporter
except ImportError as e:
    print(f"Error importing DrumScript modules: {e}")
    print("Please ensure your project structure is correct and that main.py can access its sub-packages.")
    print("If running with 'uv', ensure you've run 'uv pip install -e .' from the DrumScript project root.")
    sys.exit(1)


def transcribe_audio_to_sheet_music(input_audio_path: str, output_pdf_path: str):
    """
    Orchestrates the entire process from an input drum audio file
    to an output PDF of the sheet music.
    """
    if not os.path.exists(input_audio_path):
        print(f"Error: Input audio file not found at '{input_audio_path}'")
        sys.exit(1)

    print(f"--- Starting DrumScript Transcription ---")
    print(f"Input Audio: {input_audio_path}")

    # Define a segment length for feature extraction (e.g., 200ms)
    # This should match how your model was trained
    segment_length_seconds = 0.2

    try:
        # Step 1: Load and process audio, detect onsets
        print("Step 1/4: Loading and processing audio and detecting onsets...")
        audio_data, sr = audio_loader.load_audio(input_audio_path)
        normalised_audio = audio_loader.normalise_audio(audio_data)
        onsets = onset_detector.detect_onsets(normalised_audio, sr)
        print(f"Detected {len(onsets)} onsets.")

        # Step 2: Extract features for each detected onset
        print("\nStep 2/4: Extracting features from detected onsets...")
        all_features = []
        actual_onset_times = []

        for onset_time in onsets:
            # Create a segment around the onset
            start_sample = max(0, int((onset_time - segment_length_seconds / 2) * sr))
            end_sample = int((onset_time + segment_length_seconds / 2) * sr)
            audio_segment = normalised_audio[start_sample:end_sample]

            if audio_segment.size > 0:
                features = feature_extractor.extract_features(audio_segment, sr)
                # Flatten all features from the dictionary into a single array for the classifier
                # Ensure the order of features is consistent with training data
                flat_features = np.concatenate([f.flatten() for f in features.values()])
                all_features.append(flat_features)
                actual_onset_times.append(onset_time)
        
        if not all_features:
            print("No features extracted, perhaps no onsets detected or segments too short.")
            return # Exit if no features to classify

        all_features = np.array(all_features)
        print(f"Extracted features for {len(all_features)} segments.")

        # Step 3: Load the trained drum classification model and classify events
        print("\n--- Loading Model Components ---")
        # Construct paths to model components relative to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(project_root, "models", "multi_label_drum_classifier_model.h5")
        scaler_path = os.path.join(project_root, "models", "multi_label_scaler.joblib")
        label_map_path = os.path.join(project_root, "models", "multi_label_label_map.json")

        # Load the Keras model
        keras_model = tf.keras.models.load_model(model_path)
        print(f"Loaded Keras model from: {model_path}")
        
        # Load the scaler
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from: {scaler_path}")
        
        # Load the label map
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        print(f"Loaded label map from: {label_map_path}")
        
        # Create an inverted label map for converting prediction indices back to drum names
        idx_to_label = {idx: label for label, idx in label_map.items()}
        print(f"Loaded drum types (in order): {list(label_map.keys())}")
        print("--- Model Components Loaded Successfully ---\n")

        print("Step 3/4: Classifying drum sounds...")
        # Scale features using the loaded scaler
        scaled_features = scaler.transform(all_features)

        # Make predictions using the Keras model (returns probabilities for each drum type)
        predictions_raw = keras_model.predict(scaled_features)

        # Convert probabilities to binary labels (0 or 1) using a threshold (e.g., 0.5)
        # This assumes a multi-label classification where each output is independent.
        predicted_labels_binary = (predictions_raw > 0.5).astype(int)

        all_drum_events = []
        for i, onset_time in enumerate(actual_onset_times):
            detected_drums_for_onset = []
            # Iterate through the binary predictions for this specific onset
            for label_idx, is_present in enumerate(predicted_labels_binary[i]):
                if is_present:
                    # Get the drum type string from the inverted label map
                    drum_type = idx_to_label[label_idx]
                    detected_drums_for_onset.append(drum_type)
            
            # Format the event for score_builder.
            # score_builder expects 'drum_type' to be a list of strings for a chord.
            if detected_drums_for_onset: # Only add if at least one drum type was detected
                all_drum_events.append({
                    'onset_time_seconds': onset_time,
                    'drum_type': detected_drums_for_onset 
                })
        print(f"Classified {len(all_drum_events)} drum events.")


        # Step 4: Quantize events and build/export the score
        print("\nStep 4/4: Building and exporting drum score...")
        # The score_builder.build_and_export_drum_score function handles the
        # rest of the notation generation (quantization, MusicXML creation, and PDF export).
        # It expects detected_events to be a list of dictionaries,
        # where each dictionary contains 'onset_time_seconds' and 'drum_type' (which can be a list of strings).
        score_builder.build_and_export_drum_score(
            detected_events=all_drum_events,
            tempo=65, # This can be made configurable via command line args later if needed
            output_filepath=output_pdf_path,
            quantization_subdivision=16 # This can be made configurable too
        )

        print(f"--- Transcription Complete! ---")
        print(f"Sheet music saved to: {output_pdf_path}")

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)


def main():
    """
    Main entry point for the DrumScript command-line application.
    Parses arguments and calls the transcription function.
    """
    parser = argparse.ArgumentParser(
        description="Convert drum audio into musical sheet music (PDF).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_audio_path",
        type=str,
        help="Path to the input drum audio file (e.g., .wav, .mp3, .flac)."
    )
    parser.add_argument(
        "output_pdf_path",
        type=str,
        help="Path where the generated sheet music PDF will be saved.\\n"
             "Example: my_drum_song.pdf or output/my_sheet.pdf"
    )

    args = parser.parse_args()
    
    transcribe_audio_to_sheet_music(args.input_audio_path, args.output_pdf_path)

if __name__ == "__main__":
    main()