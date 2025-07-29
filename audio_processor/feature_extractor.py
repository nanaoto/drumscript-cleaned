# DrumScript/audio_processor/feature_extractor.py

"""
This module will extract relevant features from audio segments for drum classification.
"""

import librosa
import numpy as np
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


def extract_features(audio_segment: np.ndarray, sr: int) -> np.ndarray: # Changed return type to np.ndarray
    """
    Extracts various audio features from a short audio segment,
    ensuring a consistent number of frames (timesteps) for CNN input.

    Args:
        audio_segment (np.ndarray): A short audio time series segment containing a drum hit.
                                    Expected to be already padded/truncated to EXPECTED_AUDIO_LEN_SAMPLES
                                    if called from a pipeline that handles this (like model_trainer).
        sr (int): The sample rate of the audio segment.

    Returns:
        np.ndarray: A 2D NumPy array containing the stacked and shaped features.
                    Shape: (EXPECTED_N_FRAMES, total_number_of_features_per_frame).
    """
    # Number of MFCCs + Spectral Centroid + Spectral Rolloff + ZCR + RMS = 20 + 1 + 1 + 1 + 1 = 24
    TOTAL_FEATURES_PER_FRAME = 20 + 1 + 1 + 1 + 1
    if audio_segment.size == 0:
        # Return zeros for all features if segment is empty to maintain consistent shape.
        return np.zeros((EXPECTED_N_FRAMES, TOTAL_FEATURES_PER_FRAME))


    # --- Feature Extraction without time-averaging ---

    # MFCCs (Mel-frequency cepstral coefficients)
    # n_mfcc=20 is standard. librosa returns (n_mfcc, n_frames). Transpose to (n_frames, n_mfcc).
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfccs = mfccs.T # Shape: (n_frames, n_mfcc)

    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    spectral_centroids = spectral_centroids.reshape(-1, 1) # Shape: (n_frames, 1)

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    spectral_rolloff = spectral_rolloff.reshape(-1, 1) # Shape: (n_frames, 1)

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio_segment, hop_length=HOP_LENGTH)[0]
    zcr = zcr.reshape(-1, 1) # Shape: (n_frames, 1)

    # RMS Energy
    rms = librosa.feature.rms(y=audio_segment, hop_length=HOP_LENGTH)[0]
    rms = rms.reshape(-1, 1) # Shape: (n_frames, 1)

    # Combine all features. Ensure all have the same number of frames.
    # librosa.feature functions with same n_fft/hop_length will produce same n_frames.
    combined_features = np.hstack((mfccs, spectral_centroids, spectral_rolloff, zcr, rms))

    # --- Pad/Truncate features to EXPECTED_N_FRAMES ---
    # This is crucial for consistent input shape to the CNN
    current_n_frames = combined_features.shape[0]

    if current_n_frames < EXPECTED_N_FRAMES:
        padding_frames = EXPECTED_N_FRAMES - current_n_frames
        # Pad with zeros along the frames dimension, maintaining the feature dimension size
        padded_features = np.pad(combined_features, ((0, padding_frames), (0, 0)), mode='constant')
        return padded_features
    elif current_n_frames > EXPECTED_N_FRAMES:
        # Truncate if too many frames
        truncated_features = combined_features[:EXPECTED_N_FRAMES, :]
        return truncated_features
    else:
        return combined_features


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
                    print(f"  Expected features shape: ({EXPECTED_N_FRAMES}, {TOTAL_FEATURES_PER_FRAME})")
                    assert features_array.shape == (EXPECTED_N_FRAMES, TOTAL_FEATURES_PER_FRAME), "Output shape mismatch!"
                    
                    # You can print statistics of the features_array here if desired
                    # For example, print mean of first few feature dimensions
                    # print(f"  Mean of first 5 feature dimensions: {np.mean(features_array[:, :5], axis=0)}")
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