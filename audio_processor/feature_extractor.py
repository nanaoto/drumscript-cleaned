# DrumScript/audio_processor/feature_extractor.py

"""
This module will extract relevant features from audio segments for drum classification.
"""

import numpy as np
import librosa
import os
import soundfile
import math # Added for math.floor to calculate EXPECTED_N_FRAMES

# --- Configuration (Mirroring model_trainer.py for consistency) ---
# These should ideally be imported from a central 'constants.py'
# but for now, we'll keep them consistent by defining them here.
SAMPLE_RATE = 22050
SEGMENT_LENGTH_SECONDS = 0.2
N_FFT = 1024
HOP_LENGTH = 512

# Calculate the expected number of frames (timesteps) per segment
# This calculation needs to be robust to ensure consistency with librosa's output.
# librosa.stft typically returns 1 + (len(y) - n_fft) // hop_length frames.
# So, we first determine the expected audio length that corresponds to SEGMENT_LENGTH_SECONDS
EXPECTED_AUDIO_LEN_SAMPLES = int(SEGMENT_LENGTH_SECONDS * SAMPLE_RATE)
EXPECTED_N_FRAMES = 1 + (EXPECTED_AUDIO_LEN_SAMPLES - N_FFT) // HOP_LENGTH
if EXPECTED_N_FRAMES < 1:
    EXPECTED_N_FRAMES = 1 # Ensure at least one frame, even for very short segments

# previous Define TOTAL_FEATURES_PER_FRAME globally so it's accessible in __main__ block
# previous Number of MFCCs + Spectral Centroid + Spectral Rolloff + ZCR + RMS = 20 + 1 + 1 + 1 + 1 = 24
# previous TOTAL_FEATURES_PER_FRAME = 20 + 1 + 1 + 1 + 1

# current TOTAL_FEATURES_PER_FRAME = 
# current TOTAL_FEATRUES_PER_FRAME = 40 (MFCCs) + 1 (Centroid) + 1 (Rolloff) + 1 (RMS) + 1 (Zero-Crossing Rate) = 44

TOTAL_FEATURES_PER_FRAME = 40 + 1 + 1 + 1 + 1 # 44


# Drumscript/audio_processor/feature_extractor.py

def extract_features(audio_segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts a combined feature vector from an audio segment.

    Features:
    - 40 MFCCs (mean over time)
    - Spectral Centroid (mean)
    - Spectral Rolloff (mean)
    - RMS Energy (mean)

    Returns a single, flattened numpy array.
    """
    if audio_segment.size == 0:
        return None # Return None if segment is empty

    try:
        # 1. MFCCs
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=TOTAL_FEATURES_PER_FRAME) # ENSURE n_Mfcc = TOTAL_FEATURES_PER_FRAME, ensure hardcoded line ~37 above matches the number of features extracted.
        # note - the number of features is not hard binded, ie you can define a subset, that is n_mfcc = 40
        mfccs_mean = np.mean(mfccs, axis=1)

        # 2. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)

        # 3. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)

        # 4. RMS Energy
        rms = librosa.feature.rms(y=audio_segment)
        rms_mean = np.mean(rms)

        # 5. Combine all features into a single vector
        combined_features = np.hstack([
            mfccs_mean,
            spectral_centroid_mean,
            spectral_rolloff_mean,
            rms_mean
        ])

        return combined_features

    except Exception as e:
        print(f"Error extracting features from segment: {e}")
        return None


if __name__ == '__main__':
    # This block is for testing the feature_extractor.py script directly.
    import os
    from audio_loader import load_audio

    print("--- Running feature_extractor.py example ---")

    # Use a relative path to make the script more portable
    try:
        # Construct path to the test audio file
        script_dir = os.path.dirname(__file__)
        test_audio_path = os.path.join(script_dir, '..', 'test_audio', 'test.wav')
        
        print(f"Attempting to load: {test_audio_path}")
        audio_data, sample_rate = load_audio(test_audio_path, sr=22050)
        
        print(f"Loaded audio: Shape={audio_data.shape}, Sample Rate={sample_rate}")

        # Create a single, short segment for testing (e.g., 200ms)
        segment_duration_ms = 200
        segment_samples = int(sample_rate * (segment_duration_ms / 1000.0))
        test_segment = audio_data[sample_rate : sample_rate + segment_samples] # Take a slice 1s in

        print(f"\n--- Testing extract_features function ---")
        
        # Call the new, correct function
        features = extract_features(test_segment, sample_rate)
        
        if features is not None:
            print(f"  ✅ Function executed successfully.")
            print(f"  Combined features shape: {features.shape}")
            print(f"  Expected features shape: ({TOTAL_FEATURES_PER_FRAME},)")
            
            # --- VERIFICATION ---
            assert features.shape == (TOTAL_FEATURES_PER_FRAME,), "Output shape mismatch!"
            print("  ✅ SUCCESS: The output shape is correct!")
            
            print(f"\n  Sample of features (first 5): {features[:5]}")
        else:
            print("  ❌ FAILURE: Feature extraction returned None.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the example execution: {e}")

    print("\n--- feature_extractor.py example finished ---")
    print("\n-----------------------------------------------------")
