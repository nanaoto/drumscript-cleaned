# DrumScript/drum_classifier/data_preparer.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd # For easier data handling

# Import functions from your audio_processor module
# Assuming you'll run this from the project root using `python -m DrumScript.drum_classifier.data_preparer`
from audio_processor.audio_loader import load_audio, normalise_audio
from audio_processor.onset_detector import detect_onsets # Might be used to segment if raw files are long
from audio_processor.feature_extractor import extract_features


def prepare_dataset(data_dir: str, sr: int = 22050, segment_length_seconds: float = 0.2) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepares a dataset for drum classification.
    It loads audio files from subdirectories (e.g., 'kick', 'snare', 'hihat'),
    extracts features, and compiles them with labels.

    Args:
        data_dir (str): Path to the directory containing drum sound subfolders (e.g., 'data/').
                        Expected structure: data_dir/kick/, data_dir/snare/, etc.
        sr (int): Target sample rate for audio processing.
        segment_length_seconds (float): Length of the audio segment to extract features from,
                                        centered around the onset.

    Returns:
        tuple[np.ndarray, np.ndarray, StandardScaler]:
            - X (np.ndarray): Feature matrix (samples x features).
            - y (np.ndarray): Label vector (numerical labels corresponding to drum types).
            - scaler (StandardScaler): The fitted scaler for feature normalization.
    """
    X_features = []
    y_labels = []
    label_map = {}
    current_label_id = 0

    print(f"Preparing dataset from: {data_dir}")

    # Iterate through drum type subdirectories (e.g., 'kick', 'snare', 'hihat')
    for drum_type in sorted(os.listdir(data_dir)):
        drum_type_path = os.path.join(data_dir, drum_type)
        if not os.path.isdir(drum_type_path):
            continue # Skip non-directory files

        if drum_type not in label_map:
            label_map[drum_type] = current_label_id
            current_label_id += 1
        label_id = label_map[drum_type]

        print(f"  Processing '{drum_type}' sounds...")
        for audio_file_name in os.listdir(drum_type_path):
            if audio_file_name.lower().endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(drum_type_path, audio_file_name)
                try:
                    audio_data, sample_rate = load_audio(file_path, sr=sr)
                    normalised_audio = normalise_audio(audio_data)

                    # For individual drum samples, assume the "hit" is the whole file or its beginning.
                    # A more robust approach for raw recordings would be to use onset_detector here,
                    # but for pre-cut samples, we can simplify.
                    # For simplicity, if it's a short sample, just use the whole sample.
                    # If it's a longer recording with multiple hits, use the onset detector.
                    
                    # For pre-cut samples, assume the entire sample is "the hit"
                    # We can use the whole audio for feature extraction if it's short,
                    # or the first segment if it's potentially longer.
                    segment_length_samples = int(segment_length_seconds * sr)
                    if len(normalised_audio) > segment_length_samples:
                        audio_segment = normalised_audio[:segment_length_samples]
                    else:
                        audio_segment = normalised_audio


                    features_dict = extract_features(audio_segment, sample_rate)

                    # Flatten all features into a single 1D array
                    # Ensure consistent order of features
                    # This relies on extract_features returning flattened arrays
                    feature_vector = np.concatenate([
                        features_dict["mfccs"],
                        features_dict["spectral_centroid"],
                        features_dict["spectral_rolloff"],
                        features_dict["zero_crossing_rate"],
                        features_dict["rms"]
                        # features_dict["chroma"] # Uncomment if you include chroma features
                    ])
                    
                    # Skip if any feature is empty (e.g., from very short or silent segments)
                    if feature_vector.size == 0 or np.isnan(feature_vector).any():
                        print(f"    Skipping {audio_file_name} due to empty or NaN features.")
                        continue

                    X_features.append(feature_vector)
                    y_labels.append(label_id)

                except Exception as e:
                    print(f"    Error processing {file_path}: {e}")
                    continue

    if not X_features:
        raise ValueError("No features were extracted. Check your data directory and audio files.")

    X = np.array(X_features)
    y = np.array(y_labels)

    print(f"\nFinished data preparation.")
    print(f"Total samples: {X.shape[0]}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Labels processed: {label_map}")

    # Standardize features (important for many ML models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled using StandardScaler.")

    return X_scaled, y, scaler, label_map


if __name__ == "__main__":
    # Example Usage for data_preparer.py
    # This assumes you have a 'data' folder at your project root
    # like DRUMSCRIPT/training_data/kick/, DRUMSCRIPT/training_data/snare/, DRUMSCRIPT/training_data/hihat/

    # Determine project root
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    data_directory = os.path.join(project_root, "training_data") # Adjust if your training data is elsewhere

    print(f"Attempting to prepare data from: {data_directory}")

    # Create dummy data directories and files for testing if they don't exist
    # In a real scenario, you'd replace this with your actual dataset.
    if not os.path.exists(data_directory):
        print(f"Creating dummy data directory: {data_directory}")
        os.makedirs(data_directory)
        dummy_drum_types = ['kick', 'snare', 'hihat']
        sr_dummy = 22050
        duration_dummy = 0.5 # short dummy sample

        for d_type in dummy_drum_types:
            type_dir = os.path.join(data_directory, d_type)
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)
            # Create a few dummy WAV files for each type
            for i in range(3): # 3 dummy files per type
                t_dummy = np.linspace(0, duration_dummy, int(sr_dummy * duration_dummy), endpoint=False)
                if d_type == 'kick':
                    audio_dummy = 0.7 * np.sin(2 * np.pi * 80 * t_dummy) * np.exp(-5 * t_dummy)
                elif d_type == 'snare':
                    audio_dummy = 0.6 * (np.random.randn(len(t_dummy)) + np.sin(2 * np.pi * 2000 * t_dummy)) * np.exp(-8 * t_dummy)
                else: # hihat
                    audio_dummy = 0.5 * np.random.randn(len(t_dummy)) * np.exp(-10 * t_dummy)
                
                dummy_file = os.path.join(type_dir, f"{d_type}_{i+1}.wav")
                try:
                    import soundfile as sf
                    sf.write(dummy_file, audio_dummy, sr_dummy)
                    print(f"  Created dummy file: {dummy_file}")
                except ImportError:
                    print("Skipping dummy file creation: 'soundfile' not installed.")
                    break # Don't try to create more if soundfile is missing

    try:
        features_matrix, labels_vector, scaler, label_mapping = prepare_dataset(data_directory, sr=sr)
        print("\nData preparation successful!")
        print(f"Feature matrix shape (X): {features_matrix.shape}")
        print(f"Labels vector shape (y): {labels_vector.shape}")
        print(f"Label mapping: {label_mapping}")

        # Optional: Save scaler and label mapping for later use
        import joblib
        scaler_path = os.path.join(project_root, "models", "scaler.joblib")
        label_map_path = os.path.join(project_root, "models", "label_map.json")
        
        # Ensure 'models' directory exists
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        import json
        with open(label_map_path, 'w') as f:
            json.dump(label_mapping, f)
        print(f"Label map saved to: {label_map_path}")

    except Exception as e:
        print(f"\nError during data preparation example: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up dummy data if created (careful in real scenarios!)
        if os.path.exists(data_directory) and "dummy" in data_directory: # Added condition to only remove if it's definitely a dummy folder
            print(f"\nCleaning up dummy data directory: {data_directory}")
            import shutil
            try:
                shutil.rmtree(data_directory)
                print("Dummy data cleaned up.")
            except OSError as e:
                print(f"Error removing dummy data directory {data_directory}: {e}")