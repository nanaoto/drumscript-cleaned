import argparse
import os # Make sure os is imported for path manipulations
import joblib # Import joblib for loading scaler and model
import json   # Import json for loading label map
import numpy as np # Import numpy for array manipulation


# Corrected imports for main.py located within the DrumScript package directory
from audio_processor.audio_loader import load_audio
from audio_processor.onset_detector import detect_onsets
from audio_processor.feature_extractor import extract_features
from drum_classifier.drum_model import DrumClassifier # Already correct
# from notation_generator.score_builder import quantize_events, create_score_data # map_to_drum_parts might not be needed
from notation_generator.score_builder import quantize_events, map_to_drum_parts, create_score_data
from notation_generator.pdf_exporter import generate_pdf
from utils.config import get_config # This imports get_config from utils/config.py


def run_drumscript(audio_filepath, output_pdf_path, tempo=None):
    config = get_config()

    print(f"Loading audio from {audio_filepath}...")
    audio_data, sr = load_audio(audio_filepath, sr=config['audio']['sample_rate'])

    print("Detecting drum onsets...")
    onset_times = detect_onsets(audio_data, sr)
    print(f"Detected {len(onset_times)} onsets.")

    print("Loading drum classification model, scaler, and label map...")
    # --- START of MODIFIED/NEW LOADING LOGIC ---
    models_dir = config['model']['models_dir'] # Assuming you add this to your config.py
    model_path = os.path.join(models_dir, 'drum_classifier_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    label_map_path = os.path.join(models_dir, 'label_map.json')

    # Load the trained model
    # Initialise the DrumClassifier, then load the actual trained model into its .model attribute
    drum_classifier_instance = DrumClassifier(model_type=config['model']['type']) # Use the model type from config
    drum_classifier_instance.model = joblib.load(model_path)

    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Load the label map
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    inverse_label_map = {v: k for k, v in label_map.items()}
    # --- END of MODIFIED/NEW LOADING LOGIC ---

    classified_events = []
    print("Classifying drum events...")
    # Loop through detected onsets to extract features and classify
    for i, onset_time in enumerate(onset_times):
        # Extract small segment around the onset
        start_sample = int(onset_time * sr)
        # Use segment_length_seconds from config for consistency
        segment_length_samples = int(config['audio']['segment_length_seconds'] * sr) 
        end_sample = min(start_sample + segment_length_samples, len(audio_data))
        audio_segment = audio_data[start_sample:end_sample]

        # Ensure the segment is not empty for feature extraction
        if audio_segment.size == 0:
            print(f"  Warning: Empty segment for onset at {onset_time:.2f}s. Skipping feature extraction.")
            continue

        features_dict = extract_features(audio_segment, sr)
        
        # Flatten the dictionary of features into a single NumPy array
        # Ensure the order of features is consistent with how the model was trained
        all_features = []
        # The `sorted(features_dict.keys())` ensures consistent feature order as in `predict.py`
        for key in sorted(features_dict.keys()): 
            feature_array = features_dict[key]
            # Ensure the feature_array is a 1D array before extending
            if feature_array.ndim > 1:
                feature_array = feature_array.flatten()
            all_features.extend(feature_array)
        
        # Reshape the features for single sample prediction (1 row, N columns)
        X_new = np.array(all_features).reshape(1, -1) 

        # Scale the new features using the loaded scaler
        X_new_scaled = scaler.transform(X_new) # Use the directly loaded 'scaler' object

        # Predict the drum type using the loaded model instance
        predictions_numeric = drum_classifier_instance.predict(X_new_scaled)
        # Get the string label from the inverse label map
        predicted_label = inverse_label_map.get(predictions_numeric[0], f"Unknown Label {predictions_numeric[0]}")
        
        classified_events.append({'time': onset_time, 'drum_type': predicted_label})
        print(f"  Onset {i+1}: {onset_time:.2f}s -> {predicted_label}")

    print("Quantizing and building score data...")
    # If tempo is not provided, use the default from config
    # You might want more sophisticated tempo detection here later
    final_tempo = tempo if tempo else config['notation']['default_tempo']
    """    quantized_events = quantize_events(classified_events, tempo=final_tempo, subdivision=config['notation']['subdivision'])
        
        # map_to_drum_parts is likely called within create_score_data, or directly if needed
        score_data = create_score_data(quantized_events)"""

    quantized_events = quantize_events(classified_events, tempo=final_tempo, subdivision=config['notation']['subdivision'])

    # Map quantized events to drum parts with notation-specific properties
    quantized_and_mapped_events = map_to_drum_parts(quantized_events)

    # Now pass the correctly mapped events to create_score_data
    score_data = create_score_data(quantized_and_mapped_events)

    print(f"Generating PDF sheet music to {output_pdf_path}...")
    generate_pdf(score_data, output_pdf_path)

    print("DrumScript conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrumScript: Convert drum audio to sheet music.")
    parser.add_argument("audio_file", type=str, help="Path to the input drum audio file (e.g., .wav, .mp3).")
    parser.add_argument("output_pdf", type=str, help="Path for the output PDF sheet music file.")
    parser.add_argument("--tempo", type=int, default=None, help="Optional: Specify tempo in BPM. If not provided, a default or detected tempo will be used.")
    args = parser.parse_args()

    # Call the main function to run the drumscript
    run_drumscript(args.audio_file, args.output_pdf, args.tempo)