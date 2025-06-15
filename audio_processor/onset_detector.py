# DrumScript/audio_processor/onset_detector.py

"""
This module will detect the onset (start) times of drum hits in the audio.
"""

import librosa
import numpy as np

def detect_onsets(audio_data: np.ndarray, sr: int) -> list[float]:
    """
    Detects the onset (start) times of percussive events in an audio signal.

    This function uses librosa's built-in onset detection algorithms, which
    typically rely on spectral flux or other energy-based methods to identify
    sudden changes in the audio signal characteristic of percussive hits.

    Args:
        audio_data (np.ndarray): The input audio time series.
        sr (int): The sample rate of the audio data.

    Returns:
        list[float]: A list of detected onset times in seconds.
    """
    if audio_data.size == 0:
        return []

    # Compute onset strength envelope
    # D = librosa.stft(audio_data) # Could compute STFT explicitly if needed
    onset_env = librosa.onset.onset_detect(y=audio_data, sr=sr, units='frames')

    # Convert onset frames to time in seconds
    onset_times = librosa.frames_to_time(onset_env, sr=sr)

    return onset_times.tolist()


if __name__ == "__main__":
    print("Running onset_detector.py example...")
    try:
        # Create a dummy audio with some simulated "hits"
        # Requires soundfile for dummy .wav creation
        import soundfile as sf

        sr = 22050
        duration = 5 # seconds
        # Generate some synthetic drum-like pulses
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio_with_hits = np.zeros_like(t)

        # Simulate a few impulses at specific times
        hit_times = [0.5, 1.2, 1.8, 2.5, 3.1, 3.8, 4.4]
        for hit_time in hit_times:
            start_sample = int(hit_time * sr)
            end_sample = min(start_sample + int(0.05 * sr), len(audio_with_hits)) # 50ms pulse
            if start_sample < len(audio_with_hits):
                # A simple decaying impulse
                audio_with_hits[start_sample:end_sample] += np.exp(-np.linspace(0, 5, end_sample - start_sample)) * 0.8

        dummy_filepath = "dummy_audio_with_hits.wav"
        sf.write(dummy_filepath, audio_with_hits, sr)
        print(f"Created dummy audio file with hits: {dummy_filepath}")

        # Load the dummy audio
        loaded_audio, loaded_sr = librosa.load(dummy_filepath, sr=sr)

        # Detect onsets
        detected_onsets = detect_onsets(loaded_audio, loaded_sr)
        print(f"Original hit times: {hit_times}")
        print(f"Detected onsets (seconds): {[f'{t:.2f}' for t in detected_onsets]}")

        # Basic assertion: check if number of detected onsets is reasonable
        assert len(detected_onsets) > 0, "No onsets detected in dummy audio!"
        # More robust testing would compare detected times to original hit_times with a tolerance

        # Clean up dummy file
        import os
        os.remove(dummy_filepath)
        print(f"Cleaned up dummy audio file: {dummy_filepath}")

    except ImportError:
        print("Install 'soundfile' (pip install soundfile) to run this example fully.")
        print("Skipping dummy file creation/deletion.")
    except Exception as e:
        print(f"An error occurred during the example execution: {e}")

    print("onset_detector.py example finished.")