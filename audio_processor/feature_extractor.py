# DrumScript/audio_processor/feature_extractor.py

"""
This module will extract relevant features from audio segments for drum classification.
"""

import librosa
import numpy as np
import os # Import os for path manipulation

# The extract_features function remains unchanged
def extract_features(audio_segment: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    # ... (Your existing extract_features function code) ...
    features = {}

    if audio_segment.size == 0:
        features["mfccs"] = np.array([])
        features["spectral_centroid"] = np.array([])
        features["spectral_rolloff"] = np.array([])
        features["zero_crossing_rate"] = np.array([])
        features["rms"] = np.array([])
        features["chroma"] = np.array([])
        return features

    n_fft = 2048
    hop_length = 512

    if len(audio_segment) < n_fft:
        padded_segment = np.pad(audio_segment, (0, n_fft - len(audio_segment)), mode='constant')
        y_proc = padded_segment
    else:
        y_proc = audio_segment

    features["mfccs"] = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    features["spectral_centroid"] = librosa.feature.spectral_centroid(y=y_proc, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features["spectral_rolloff"] = librosa.feature.spectral_rolloff(y=y_proc, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(y=y_proc)
    features["rms"] = librosa.feature.rms(y=y_proc)

    flattened_features = {k: v.flatten() for k, v in features.items()}

    return flattened_features


if __name__ == "__main__":
    print("Running feature_extractor.py example with test.mp3...")
    try:
        # Import necessary modules from your package
        #from DrumScript.audio_processor.audio_loader import load_audio, normalise_audio
        #from DrumScript.audio_processor.onset_detector import detect_onsets

        from audio_processor.audio_loader import load_audio, normalise_audio
        from audio_processor.onset_detector import detect_onsets

        sr = 22050 # Target sample rate for processing
        segment_length_seconds = 0.2 # Length of audio segment around each onset (e.g., 200ms)

        # --- Path to your actual drum recording (test.mp3) ---
        # This dynamic path calculation should correctly point to DRUMSCRIPT/tests/test.mp3
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels from audio_processor/feature_extractor.py to the outer DRUMSCRIPT/ folder
        project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
        # Construct the path to test.mp3 within the 'tests' directory
        test_mp3_path = os.path.join(project_root, "DrumScript/tests", "test.mp3")


        print(f"Attempting to load: {test_mp3_path}")
        # Load and normalise the test.mp3 audio
        audio_data, sample_rate = load_audio(test_mp3_path, sr=sr)
        normalised_audio = normalise_audio(audio_data)
        print(f"Loaded audio: Shape={normalised_audio.shape}, Sample Rate={sample_rate}, Duration={len(normalised_audio)/sample_rate:.2f} seconds")

        # Detect onsets
        print("\nDetecting onsets...")
        onsets = detect_onsets(normalised_audio, sample_rate)
        print(f"Detected {len(onsets)} onsets.")

        if onsets:
            # Process features for a few detected onsets (e.g., first 5 or all if fewer)
            num_onsets_to_process = min(len(onsets), 5) # Process up to 5 onsets for brevity
            for i in range(num_onsets_to_process):
                onset_time = onsets[i]
                onset_sample = int(onset_time * sample_rate)
                segment_length_samples = int(segment_length_seconds * sample_rate)

                # Define the segment: from onset to onset + segment_length
                segment_start = onset_sample
                segment_end = min(onset_sample + segment_length_samples, len(normalised_audio))
                
                # Ensure the segment is valid and long enough
                if segment_start >= len(normalised_audio) or segment_end <= segment_start:
                    print(f"  Skipping onset {i+1} at {onset_time:.2f}s: Segment out of bounds or too short.")
                    continue

                audio_segment = normalised_audio[segment_start:segment_end]

                print(f"\n--- Features for Onset {i+1} (at {onset_time:.2f}s) ---")
                if audio_segment.size > 0:
                    features = extract_features(audio_segment, sample_rate)

                    for key, value in features.items():
                        # Handle potential empty arrays from extract_features if segment was too short after padding
                        if value.size > 0:
                            print(f"  {key}: shape={value.shape}, mean={np.mean(value):.4f}")
                        else:
                            print(f"  {key}: (empty)")
                    assert all(f.size > 0 for f in features.values()), f"Some features are empty for onset {i+1}!"
                else:
                    print(f"  Onset {i+1} segment is empty, cannot extract features.")
        else:
            print("No onsets detected in the audio, cannot extract features.")

    except FileNotFoundError:
        print(f"\nERROR: The audio file '{test_mp3_path}' was not found.")
        print("Please ensure you have placed 'test.mp3' inside your 'DrumScript/tests/' directory.")
    except ImportError:
        print("\nERROR: Required modules/libraries might be missing.")
        print("Ensure 'soundfile', 'librosa', 'numpy', and your DrumScript modules are correctly installed and structured.")
        print("For MP3, 'ffmpeg' must also be installed on your system and accessible in PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the example execution: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

    print("\nfeature_extractor.py example finished.")