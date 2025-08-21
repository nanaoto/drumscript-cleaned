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



if __name__ == "__main__":
    print("Running feature_extractor.py example with test.mp3/test.wav...")
    try:
        from audio_loader import load_audio, normalise_audio
        from onset_detector import detect_onsets

        # --- Path to your actual drum recording (test.wav) ---
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
        test_audio_path = os.path.join(project_root, "DrumScript/test_audio", "test.wav")

        print(f"Attempting to load: {test_audio_path}")
        audio_data, sample_rate = load_audio(test_audio_path, sr=SAMPLE_RATE) # Use global SAMPLE_RATE
        normalised_audio = normalise_audio(audio_data)
        print(f"Loaded audio: Shape={normalised_audio.shape}, Sample Rate={sample_rate}, Duration={len(normalised_audio)/sample_rate:.2f} seconds")

        print("\nDetecting onsets...")
        onsets = detect_onsets(normalised_audio, sample_rate)
        print(f"Detected {len(onsets)} onsets.")

        if onsets:
            num_onsets_to_process = min(len(onsets), 5)
            for i in range(num_onsets_to_process):
                onset_time = onsets[i]
                onset_sample = int(onset_time * sample_rate)
                segment_length_samples_target = EXPECTED_AUDIO_LEN_SAMPLES # Use the consistent length

                # Define the segment: from onset, with padding/truncation
                segment_start = onset_sample
                segment_end = onset_sample + segment_length_samples_target

                # Ensure segment extraction handles boundaries and then pass to extract_features
                # For this example, we'll extract the segment and then let extract_features pad/truncate if needed
                # For consistency with model_trainer, it's better to ensure the raw audio segment itself
                # passed to extract_features is already the EXPECTED_AUDIO_LEN_SAMPLES.
                # Here, we'll demonstrate that by padding the *input* audio_segment to extract_features
                
                temp_audio_segment = normalised_audio[segment_start:segment_end]
                
                # Manual padding/truncation to ensure the segment fed to extract_features has correct length
                if len(temp_audio_segment) < segment_length_samples_target:
                    padding = segment_length_samples_target - len(temp_audio_segment)
                    temp_audio_segment = np.pad(temp_audio_segment, (0, padding), mode='constant')
                elif len(temp_audio_segment) > segment_length_samples_target:
                    temp_audio_segment = temp_audio_segment[:segment_length_samples_target]

                print(f"\n--- Features for Onset {i+1} (at {onset_time:.2f}s) ---")
                if temp_audio_segment.size > 0:
                    features_array = extract_features(temp_audio_segment, sample_rate)

                    print(f"  Combined features shape: {features_array.shape}")
                    print(f"  Combined features shape: {features_array}") # print the features_array in console for reviewing
                    print(f"  Expected features shape: ({EXPECTED_N_FRAMES}, {TOTAL_FEATURES_PER_FRAME})")
                    assert features_array.shape == (EXPECTED_N_FRAMES, TOTAL_FEATURES_PER_FRAME), "Output shape mismatch!"
                    
                    # You can print statistics of the features_array here if desired
                    # For example, print mean of first few feature dimensions
                    print(f"  Mean of first 5 feature dimensions: {np.mean(features_array[:, :5], axis=0)}")
                else:
                    print(f"  Onset {i+1} segment is empty, cannot extract features.")
        else:
            print("No onsets detected in the audio, cannot extract features.")

    except FileNotFoundError:
        print(f"\nERROR: The audio file '{test_audio_path}' was not found.")
        print("Please ensure you have placed 'test.mp3/test.wav' inside your 'DrumScript/test_audio/' directory.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the example execution: {e}")
        import traceback
        traceback.print_exc()

    print("\nfeature_extractor.py example finished.")
    print("\n-----------------------------------------------------")