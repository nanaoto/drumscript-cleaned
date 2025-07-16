# DrumScript/audio_processor/feature_extractor.py

"""
This module will extract relevant features from audio segments for drum classification.
"""

import librosa
import numpy as np
import os # Import os for path manipulation

def extract_features(audio_segment: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """
    Extracts various audio features from a short audio segment.
    Returns the mean of each feature over the segment to ensure fixed-length output.

    Args:
        audio_segment (np.ndarray): A short audio time series segment containing a drum hit.
        sr (int): The sample rate of the audio segment.

    Returns:
        dict[str, np.ndarray]: A dictionary containing various extracted features.
                              Each feature is returned as a NumPy array (mean over time).
    """
    features = {}

    if audio_segment.size == 0:
        # Return zeros for all features if segment is empty to maintain consistent shape.
        # These sizes correspond to the expected output after averaging (e.g., 20 MFCCs, 1 for spectral centroid).
        features["mfccs"] = np.zeros(20) # Common n_mfcc value
        features["spectral_centroid"] = np.array([0.0])
        features["spectral_rolloff"] = np.array([0.0])
        features["zero_crossing_rate"] = np.array([0.0])
        features["rms"] = np.array([0.0])
        # features["chroma"] = np.zeros(12) # Common n_chroma value - COMMENT OUT OR DELETE THIS LINE
        return features
    
    
    # Define common FFT parameters for percussive sounds
    # Adjust n_fft and hop_length to be more suitable for short transients
    N_FFT = 1024 # A common choice for transient analysis, allows for shorter segments
    HOP_LENGTH = 512 # Standard hop_length for N_FFT=1024

    # MFCCs (Mel-frequency cepstral coefficients)
    # We take the mean across the time axis (axis=1) to get a single vector of 20 coefficients.
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=20)
    features["mfccs"] = np.mean(mfccs, axis=1) if mfccs.size > 0 else np.zeros(20)

    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
    features["spectral_centroid"] = np.array([np.mean(spectral_centroids)] if spectral_centroids.size > 0 else [0.0])

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)[0]
    features["spectral_rolloff"] = np.array([np.mean(spectral_rolloff)] if spectral_rolloff.size > 0 else [0.0])

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio_segment)[0] # [0] to get the 1D array
    features["zero_crossing_rate"] = np.array([np.mean(zcr)] if zcr.size > 0 else [0.0])

    # RMS Energy
    rms = librosa.feature.rms(y=audio_segment)[0] # [0] to get the 1D array
    features["rms"] = np.array([np.mean(rms)] if rms.size > 0 else [0.0])

# Chroma features (optional, but ensure it's handled consistently if used)
    # We take the mean across the time axis (axis=1) to get a single vector of 12 chroma values.
    # COMMENT OUT OR DELETE THESE TWO LINES:
    # chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sr)
    # features["chroma"] = np.mean(chroma, axis=1) if chroma.size > 0 else np.zeros(12)

    return features


if __name__ == "__main__":
    print("Running feature_extractor.py example with test.mp3...")
    try:
        # Import necessary modules
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
        print("Please ensure you have placed 'test.mp3' inside your 'DrumScript/tests/' directory, or updated the relative file_path.")
    except ImportError:
        print("\nERROR: Required modules/libraries might be missing.")
        print("Ensure 'soundfile', 'librosa', 'numpy', and your DrumScript modules are correctly installed and structured.")
        print("For MP3, 'ffmpeg' must also be installed on your system and accessible in PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the example execution: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

    print("\nfeature_extractor.py example finished.")