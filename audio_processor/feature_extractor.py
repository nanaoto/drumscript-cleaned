# DrumScript/audio_processor/feature_extractor.py

"""
This module will extract relevant features from audio segments for drum classification.
"""

import librosa
import numpy as np

def extract_features(audio_segment: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """
    Extracts various audio features from a short audio segment.

    These features are crucial for distinguishing between different drum sounds.

    Args:
        audio_segment (np.ndarray): A short audio time series segment containing a drum hit.
        sr (int): The sample rate of the audio segment.

    Returns:
        dict[str, np.ndarray]: A dictionary containing various extracted features.
                              Each feature is returned as a NumPy array.
                              Common features include MFCCs, spectral centroid,
                              spectral rolloff, zero-crossing rate, and RMS energy.
    """
    features = {}

    if audio_segment.size == 0:
        # Return empty arrays for all features if segment is empty
        # This prevents errors in subsequent processing steps
        features["mfccs"] = np.array([])
        features["spectral_centroid"] = np.array([])
        features["spectral_rolloff"] = np.array([])
        features["zero_crossing_rate"] = np.array([])
        features["rms"] = np.array([])
        features["chroma"] = np.array([]) # Chroma might be less useful for drums but included for completeness
        return features

    # Ensure segment is at least N_FFT long for STFT-based features
    # Default N_FFT for librosa is 2048, check if segment is long enough
    n_fft = 2048 # Common FFT window size
    hop_length = 512 # Common hop length

    if len(audio_segment) < n_fft:
        # Pad the segment if it's too short for FFT
        padded_segment = np.pad(audio_segment, (0, n_fft - len(audio_segment)), mode='constant')
        y_proc = padded_segment
    else:
        y_proc = audio_segment

    # MFCCs (Mel-Frequency Cepstral Coefficients)
    # n_mfcc: number of MFCCs to compute. 13-20 are common.
    features["mfccs"] = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)

    # Spectral Centroid
    # Represents the "center of mass" of the spectrum; higher for brighter sounds.
    features["spectral_centroid"] = librosa.feature.spectral_centroid(y=y_proc, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Spectral Rolloff
    # The frequency below which a specified percentage of the total spectral energy lies.
    # Useful for distinguishing between noisy and tonal sounds, or bright vs. dull.
    features["spectral_rolloff"] = librosa.feature.spectral_rolloff(y=y_proc, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Zero-Crossing Rate
    # The rate at which the signal changes sign. Higher for noisy/percussive sounds.
    features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(y=y_proc)

    # RMS Energy (Root Mean Square)
    # Measures the overall energy or loudness of the segment.
    features["rms"] = librosa.feature.rms(y=y_proc)

    # Chroma features (less crucial for unpitched drums, but sometimes used)
    # Represents the 12 different pitch classes.
    # features["chroma"] = librosa.feature.chroma_stft(y=y_proc, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Flatten features if they are 2D (time-varying) to make them suitable for traditional ML models
    # For deep learning (CNNs), you might keep them 2D (like spectrograms)
    flattened_features = {k: v.flatten() for k, v in features.items()}

    return flattened_features


if __name__ == "__main__":
    print("Running feature_extractor.py example...")
    try:
        # Requires soundfile for dummy .wav creation
        import soundfile as sf
        from DrumScript.audio_processor.audio_loader import load_audio
        from DrumScript.audio_processor.onset_detector import detect_onsets

        sr = 22050
        duration = 1.0 # second
        # Simulate a simple 'kick' sound (low frequency)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        kick_audio = 0.8 * np.sin(2 * np.pi * 60 * t) * np.exp(-5 * t) # 60 Hz decaying sine
        kick_filepath = "dummy_kick.wav"
        sf.write(kick_filepath, kick_audio, sr)

        # Simulate a simple 'hi-hat' sound (high frequency noise)
        hihat_audio = 0.5 * np.random.randn(int(sr * duration)) * np.exp(-10 * t) # White noise decaying
        hihat_filepath = "dummy_hihat.wav"
        sf.write(hihat_filepath, hihat_audio, sr)

        print(f"Created dummy audio files: {kick_filepath}, {hihat_filepath}")

        # --- Test with Kick Drum ---
        kick_data, kick_sr = load_audio(kick_filepath, sr=sr)
        kick_onsets = detect_onsets(kick_data, kick_sr)
        print(f"\n--- Kick Drum Features ---")
        if kick_onsets:
            # Take a small segment around the first onset
            onset_sample = int(kick_onsets[0] * kick_sr)
            segment_length = int(0.2 * kick_sr) # 200ms segment
            kick_segment = kick_data[onset_sample : onset_sample + segment_length]
            kick_features = extract_features(kick_segment, kick_sr)

            for key, value in kick_features.items():
                print(f"  {key}: shape={value.shape}, mean={np.mean(value):.4f}")
                # Expect MFCCs, spectral centroid, etc. to have values.
                assert value.size > 0, f"{key} feature is empty for kick!"
        else:
            print("No onsets detected for kick drum, cannot extract features.")


        # --- Test with Hi-Hat ---
        hihat_data, hihat_sr = load_audio(hihat_filepath, sr=sr)
        hihat_onsets = detect_onsets(hihat_data, hihat_sr)
        print(f"\n--- Hi-Hat Features ---")
        if hihat_onsets:
            onset_sample = int(hihat_onsets[0] * hihat_sr)
            segment_length = int(0.2 * hihat_sr) # 200ms segment
            hihat_segment = hihat_data[onset_sample : onset_sample + segment_length]
            hihat_features = extract_features(hihat_segment, hihat_sr)

            for key, value in hihat_features.items():
                print(f"  {key}: shape={value.shape}, mean={np.mean(value):.4f}")
                assert value.size > 0, f"{key} feature is empty for hi-hat!"
        else:
            print("No onsets detected for hi-hat, cannot extract features.")

        # Clean up dummy files
        import os
        os.remove(kick_filepath)
        os.remove(hihat_filepath)
        print(f"\nCleaned up dummy audio files.")

    except ImportError:
        print("Install 'soundfile' (pip install soundfile) to run this example fully.")
        print("Skipping dummy file creation/deletion.")
    except Exception as e:
        print(f"An error occurred during the example execution: {e}")

    print("feature_extractor.py example finished.")