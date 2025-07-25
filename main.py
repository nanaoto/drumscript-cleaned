# DrumScript/main.py

import argparse
import os
import sys
import numpy as np

# No need for sys.path manipulation here if running as 'python -m DrumScript.main'
# or if DrumScript is installed as a package (e.g., via 'uv pip install -e .').

try:
    from audio_processor import audio_loader, onset_detector, feature_extractor
    # Import the DrumClassifier class directly from its module
    from drum_classifier.drum_model import DrumClassifier
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

    # Step 1: Load audio and detect onsets
    print("Step 1/4: Loading and processing audio and detecting onsets...")
    audio_data, sample_rate = audio_loader.load_audio(input_audio_path, sr=22050) # Resample to 22050 Hz
    onsets = onset_detector.detect_onsets(audio_data, sample_rate)
    print(f"Detected {len(onsets)} onsets.")

    # Define model paths (assuming they are in DrumScript/models/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root, then into the 'models' directory
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
    models_dir = os.path.join(project_root, "models")

    model_path = os.path.join(models_dir, "multi_label_drum_classifier_model.h5")
    scaler_path = os.path.join(models_dir, "multi_label_scaler.joblib")
    label_map_path = os.path.join(models_dir, "multi_label_label_map.json")

    # Step 2: Load Model Components
    print("\n--- Loading Model Components ---")
    try:
        # Instantiate DrumClassifier, which will load the model, scaler, and label map
        drum_classifier_instance = DrumClassifier(model_path, scaler_path, label_map_path)
        print("--- Model Components Loaded Successfully ---")
    except Exception as e:
        print(f"Error loading model components: {e}")
        sys.exit(1)

    # Step 3: Extract features from detected onsets
    print("Step 2/4: Extracting features from detected onsets...")
    detected_onsets_features = feature_extractor.extract_features_from_onsets(
        audio_data=audio_data,
        onsets=onsets,
        sr=sample_rate,
        segment_length_seconds=segment_length_seconds
    )
    print(f"Extracted features for {len(detected_onsets_features)} segments.")

    # Step 4: Classify drum sounds
    print("Step 3/4: Classifying drum sounds...")
    # Call the classify_events method on the instantiated drum_classifier_instance
    all_drum_events = drum_classifier_instance.classify_events(
        features_with_onsets=detected_onsets_features
    )
    print(f"Classified {len(all_drum_events)} drum events.")


    # Step 5: Generate and export sheet music PDF
    print("\nStep 4/4: Generating sheet music PDF...")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_pdf_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # The score_builder.build_and_export_drum_score function should be ready
        # to accept the 'all_drum_events' structure (list of dicts with 'onset_time' and 'drum_type')
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