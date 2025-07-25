# DrumScript/audio_processor/feature_extractor.py

import librosa
import numpy as np
import os
from typing import List, Dict, Any

def extract_features(audio_segment: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """
    Extracts various audio features from a short audio segment.
    Returns the time-series of each feature (not the mean) for CNN input.
    Includes base MFCCs and their first derivatives (deltas).

    Args:
        audio_segment (np.ndarray): A short audio time series segment containing a drum hit.
        sr (int): The sample rate of the audio segment.

    Returns:
        dict[str, np.ndarray]: A dictionary containing various extracted features.
                              Each feature is returned as a NumPy array (feature_dim x n_frames).
    """
    features = {}

    # Define common FFT parameters for percussive sounds
    N_FFT = 1024 # A common choice for transient analysis
    HOP_LENGTH = 512 # Standard hop_length for N_FFT=1024

    # Handle empty segment case: Return arrays with appropriate dimensions, filled with zeros.
    if audio_segment.size == 0:
        # MFCCs and Delta MFCCs
        features["mfccs"] = np.empty((20, 0)) # n_mfcc x n_frames
        features["delta_mfccs"] = np.empty((20, 0)) # n_mfcc x n_frames
        # Other features (1 dimension per frame)
        features["spectral_centroid"] = np.empty((1, 0))
        features["spectral_rolloff"] = np.empty((1, 0))
        features["chroma"] = np.empty((12, 0)) # 12 chroma bins
        features["tonnetz"] = np.empty((6, 0))  # 6 tonnetz dimensions
        features["zero_crossing_rate"] = np.empty((1, 0))

        return features

    # MFCCs
    # n_mfcc=20 is standard for many audio tasks
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["mfccs"] = mfccs

    # Delta MFCCs (first derivative)
    delta_mfccs = librosa.feature.delta(mfccs)
    features["delta_mfccs"] = delta_mfccs

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["spectral_centroid"] = spectral_centroid

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["spectral_rolloff"] = spectral_rolloff

    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["chroma"] = chroma

    # Tonnetz Features (Harmonic features derived from chroma)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_segment), sr=sr)
    features["tonnetz"] = tonnetz

    # Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_segment, frame_length=N_FFT, hop_length=HOP_LENGTH)
    features["zero_crossing_rate"] = zero_crossing_rate

    return features


def extract_features_from_onsets(
    audio_data: np.ndarray,
    onsets: List[float],
    sr: int,
    segment_length_seconds: float
) -> List[Dict[str, Any]]:
    """
    Extracts features for each detected onset in the audio.

    Args:
        audio_data (np.ndarray): The full audio time series.
        onsets (List[float]): List of onset times in seconds.
        sr (int): The sample rate of the audio data.
        segment_length_seconds (float): The length of the audio segment
                                        to extract features from, centered
                                        around the onset.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              contains 'onset_time' and 'features' for a segment.
                              'features' is a dictionary of feature arrays.
    """
    all_onset_features = []
    segment_length_samples = int(segment_length_seconds * sr)

    for i, onset_time in enumerate(onsets):
        onset_sample = int(onset_time * sr)

        # Define segment: from onset_sample to onset_sample + segment_length_samples
        # Ensure segment does not go beyond audio data bounds
        segment_start = onset_sample
        segment_end = min(onset_sample + segment_length_samples, len(audio_data))

        audio_segment = audio_data[segment_start:segment_end]

        if audio_segment.size == 0:
            print(f"  Warning: Onset {i+1} at {onset_time:.2f}s has an empty audio segment. Skipping feature extraction.")
            continue

        features_for_segment = extract_features(audio_segment, sr)
        all_onset_features.append({
            "onset_time": onset_time,
            "features": features_for_segment
        })
        # print(f"  Extracted features for onset at {onset_time:.2f}s. Feature shapes: {[f'{k}: {v.shape}' for k,v in features_for_segment.items()]}")

    return all_onset_features

# The __main__ block for example usage should be kept at the end
if __name__ == "__main__":
    print("Running feature_extractor.py example with test_audio/8_rock-groove8_65_beat_4-4.wav...")
    try:
        from audio_loader import load_audio
        from onset_detector import detect_onsets

        sr = 22050 # Target sample rate for processing
        segment_length_seconds = 0.2

        # --- Path to your actual drum recording ---
        # This dynamic path calculation should correctly point to DRUMSCRIPT/test_audio/test.mp3
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
        test_audio_path = os.path.join(project_root, "test_audio", "8_rock-groove8_65_beat_4-4.wav")

        print(f"Loading audio from: {test_audio_path}")
        audio_data, sample_rate = load_audio(test_audio_path, sr=sr)
        print(f"Loaded audio with sample rate: {sample_rate}, duration: {len(audio_data)/sample_rate:.2f} seconds")

        # Detect onsets
        print("\nDetecting onsets...")
        onsets = detect_onsets(audio_data, sample_rate)
        print(f"Detected {len(onsets)} onsets.")

        # Extract features for all onsets
        print("\nExtracting features from onsets...")
        all_extracted_features = extract_features_from_onsets(audio_data, onsets, sample_rate, segment_length_seconds)
        print(f"Successfully extracted features for {len(all_extracted_features)} onsets.")

        if all_extracted_features:
            print("\nFirst onset's features (shapes and first few values):")
            first_onset = all_extracted_features[0]
            print(f"  Onset time: {first_onset['onset_time']:.2f}s")
            for key, value in first_onset['features'].items():
                print(f"    {key}: shape={value.shape}")
                if value.size > 0:
                    print(f"    {key} (first few values if 1D): {value.flatten()[:5]}")
                else:
                    print(f"    {key}: (empty array with shape {value.shape})")
        else:
            print("No features extracted (possibly no onsets detected or segments were empty).")

    except FileNotFoundError:
        print(f"\nERROR: The audio file '{test_audio_path}' was not found.")
        print("Please ensure you have placed '8_rock-groove8_65_beat_4-4.wav' inside your 'DrumScript/test_audio/' directory.")
    except ImportError:
        print("\nERROR: Required audio backend libraries might be missing.")
        print("Ensure 'soundfile' and 'pydub' are installed via 'uv pip install -r requirements.txt'.")
        print("For MP3, 'ffmpeg' must also be installed on your system and accessible in PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the example execution: {e}")
        import traceback
        traceback.print_exc()

    print("\nfeature_extractor.py example finished.")