#.audio_processor/audio_loader.py

"""
This module will handle loading and basic normalisation of audio files. It also offers automatic playback of the audio once loaded.
"""

# DrumScript/audio_processor/audio_loader.py

import librosa
import numpy as np
import os
import sounddevice as sd
import time # for pausing listenable audio

def load_audio(file_path: str, sr: int = None) -> tuple[np.ndarray, int]:
    """
    Loads an audio file and optionally resamples it.

    Args:
        file_path (str): The path to the audio file.
        sr (int, optional): The target sample rate. If None, the original
                            sample rate of the audio file is used. Defaults to None.

    Returns:
        tuple[np.ndarray, int]: A tuple containing:
                                - audio_data (np.ndarray): The loaded audio time series.
                                - sample_rate (int): The sample rate of the loaded audio.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        Exception: For other errors during audio loading (e.g., corrupted file).
    """
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=sr) # The liborsa.load_audio() fct handles wide variety of audio formats, including .mp3, .wav, .flac, .ogg, etc.
        return audio_data, sample_rate
    except FileNotFoundError:
        print(f"Error: Audio file not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        raise

def normalise_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Normalises the audio data to a range between -1.0 and 1.0.

    This helps in standardising the amplitude across different recordings
    and prevents issues with varying loudness levels during processing.

    Args:
        audio_data (np.ndarray): The input audio time series.

    Returns:
        np.ndarray: The normalised audio time series.
    """
    if audio_data.size == 0:
        return audio_data # Return empty array if input is empty

    max_abs_val = np.max(np.abs(audio_data))
    if max_abs_val > 0:
        normalised_data = audio_data / max_abs_val
    else:
        normalised_data = audio_data # Already zero or empty
    return normalised_data


if __name__ == "__main__":
    #print("Running audio_loader.py example with actual MP3/WAV...")
    print("Running audio_loader.py example with actual MP3/WAV...")

    # --- Path to your actual drum recording ---
    # Assuming your recording is in a 'data' folder at the project root
    # e.g., 'DrumScript/test_audio/test.wav'
    # e.g., 'DrumScript/test_audio/test.mp3'
    #
    # To correctly get the path:
    # 1. Get the directory of the current script (audio_loader.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Go up to the project root (from audio_processor to DRUMSCRIPT)
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    # 3. Construct the path to your audio file
    #actual_drum_recording_path = os.path.join(project_root,"DrumScript/test_audio","test.mp3") # Change .mp3 to .wav if using WAV, or other audio format
    actual_drum_recording_path = os.path.join(project_root,"DrumScript/test_audio","test.wav") # Change .wav to .mp3 if using MP3, or other audio format

    # --- IMPORTANT: Create a 'data' folder in your project root and place an MP3/WAV there ---
    # For testing, you must have 'my_drum_recording.mp3' (or .wav) inside 'DrumScript/test_audio/'
    # before running this example.

    try:
        print(f"Attempting to load: {actual_drum_recording_path}")
        audio, sr = load_audio(actual_drum_recording_path, sr=44100)
        print(f"Loaded audio: Shape={audio.shape}, Sample Rate={sr}, Duration={len(audio)/sr:.2f} seconds")

        # Test normalise_audio
        original_max = np.max(np.abs(audio))
        normalised_audio = normalise_audio(audio)
        normalised_max = np.max(np.abs(normalised_audio))
        print(f"Original max amplitude: {original_max:.4f}")
        print(f"Normalised max amplitude: {normalised_max:.4f}")
        assert np.isclose(normalised_max, 1.0) or np.isclose(normalised_max, 0.0), "Normalisation failed!"

        # Playback the loaded and normalised audio:
        print("\nPlaying loaded (and normalised) audio...")
        # sd.play takes the audio array and sample rate
        sd.play(normalised_audio, sr)
        # sd.wait() blocks until playback is complete
        sd.wait()
        print("Audio playback finished.")

        # Optional: You can also save the loaded audio to a new file to verify
        # import soundfile as sf
        # output_wav_path = "output_test_audio.wav"
        # sf.write(output_wav_path, normalised_audio, sr)
        # print(f"Saved loaded audio to: {output_wav_path}")
        # time.sleep(1) # Give it a moment before cleaning up if playing

    except FileNotFoundError:
        # ... (your existing error handling) ...
        print(f"\nERROR: The audio file '{actual_drum_recording_path}' was not found.")
        print("Please ensure you have placed an audio file (e.g., 'test.mp3' or '.wav') inside test_audio/.")
    except ImportError:
        print("\nERROR: Required audio backend libraries might be missing.")
        print("Ensure 'soundfile' and 'pydub' are installed via 'uv pip install -r requirements.txt'.")
        print("For MP3, 'ffmpeg' must also be installed on your system and accessible in PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the example execution: {e}")

    print("audio_loader.py example finished.")

