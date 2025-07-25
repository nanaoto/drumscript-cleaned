# DrumScript/main.py

import argparse
import os
import sys
import numpy as np
import joblib    # For loading scaler
import json      # For loading label_map
import tensorflow as tf # For loading Keras models

# No need for sys.path manipulation here.
# When main.py is run directly from the package root (DrumScript/),
# Python's import system will correctly resolve sub-package imports.

try:
    from audio_processor import audio_loader, onset_detector, feature_extractor
    # Import the drum_classifier package itself, not drum_model directly here
    # as classify_events will be a module-level function in drum_classifier/__init__.py
    import drum_classifier
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
    # This should match how your model was trained if feature extraction for individual hits
    segment_length_seconds = 0.2

    try:
        # 1. Load and Process Audio
        print("Step 1/4: Loading and processing audio and detecting onsets...")
        audio_data, sr = audio_loader.load_audio(input_audio_path)

        onsets = onset_detector.detect_onsets(audio_data, sr)
        print(f"Detected {len(onsets)} onsets.")

        # --- Load the pre-trained drum classification model, scaler, and label map ---
        # Ensure these paths are correct relative to where you run main.py
        # Assuming 'models' directory is at the project root level, adjacent to 'DrumScript'
        # Get the project root by going up two levels from main.py's directory if main.py is in DrumScript/
        #project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This goes up two levels
        #models_dir = os.path.join(project_root, "models")
         # Calculate models_dir assuming it's within the DrumScript package,
        # specifically in a 'models' directory at the same level as main.py
        current_dir = os.path.dirname(os.path.abspath(__file__)) # /Users/victoriamckinney/Library/Visual Studio Projects/DrumScript/
        models_dir = os.path.join(current_dir, "models") # /Users/victoriamckinney/Library/Visual Studio Projects/DrumScript/models/

        # Use the multi_label model names from your training log
        model_path = os.path.join(models_dir, "multi_label_drum_classifier_model.h5") # Keras model
        scaler_path = os.path.join(models_dir, "multi_label_scaler.joblib")
        label_map_path = os.path.join(models_dir, "multi_label_label_map.json")

        print(f"\n--- Loading Model Components ---")
        print(f"Loading model from: {model_path}")
        print(f"Loading scaler from: {scaler_path}")
        print(f"Loading label map from: {label_map_path}")

        try:
            model = tf.keras.models.load_model(model_path) # Load Keras model directly
            scaler = joblib.load(scaler_path)
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
            print(f"Loaded drum types (in order): {list(label_map.keys())}")
            print(f"--- Model Components Loaded Successfully ---")
        except Exception as e:
            print(f"Error loading model components: {e}")
            print("Please ensure your models are trained and saved correctly in the 'models/' directory.")
            sys.exit(1)

        if not onsets:
            print("No onsets detected. Cannot proceed with feature extraction and classification.")
            print(f"--- Transcription Complete (No Drums Found)! ---")
            # Ensure score_builder can handle an empty list of events for an empty score
            score_builder.build_and_export_drum_score(
                detected_events=[], # Pass an empty list of events
                tempo=120, # Default tempo
                output_filepath=output_pdf_path,
                quantization_subdivision=16
            )
            return

        # 2. Extract Features from Detected Onsets
        print("Step 2/4: Extracting features from detected onsets...")
        # This function needs to be implemented in feature_extractor.py
        # It should iterate through onsets, extract segments, and then extract features for each segment.
        # It should return a list of dictionaries, where each dictionary contains 'onset_time' and 'features'.
        detected_onsets_features = feature_extractor.extract_features_from_onsets(
            audio_data, onsets, sr, segment_length_seconds
        )
        print(f"Extracted features for {len(detected_onsets_features)} segments.")


        # 3. Classify Drum Sounds
        print(f"Step 3/4: Classifying drum sounds...")
        # This function needs to be implemented in drum_classifier/__init__.py
        # It will take the extracted features, the loaded model, scaler, and label map,
        # and return a list of dictionaries, each with 'time' and 'drums' keys.
        all_drum_events = drum_classifier.classify_events(
            detected_onsets_features,
            model,
            scaler,
            label_map
        )

        print(f"Classified {len(all_drum_events)} drum events.")
        print(f"--- Inference Complete ---")

        # 4. Generate Musical Notation and PDF
        print("Step 4/4: Generating musical notation and PDF...")
        output_dir = os.path.dirname(output_pdf_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        score_builder.build_and_export_drum_score(
            detected_events=all_drum_events,
            tempo=120, # Can be made configurable
            output_filepath=output_pdf_path,
            quantization_subdivision=16 # Can be made configurable
        )

        print(f"--- Transcription Complete! ---")
        print(f"Sheet music saved to: {output_pdf_path}")

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        import traceback
        traceback.print_exc()
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
        help="Path where the generated sheet music PDF will be saved.\n"
             "Example: my_drum_song.pdf or output/my_sheet.pdf"
    )

    args = parser.parse_args()

    transcribe_audio_to_sheet_music(args.input_audio_path, args.output_pdf_path)

if __name__ == "__main__":
    main()