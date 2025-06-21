#.audio_processor/audio_loader.py

"""
This module will handle loading and basic normalisation of audio files.
"""

# DrumScript/audio_processor/audio_loader.py

import librosa
import numpy as np

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
    if audio_data.s == 0:
        return audio_data # Return empty array if input is empty

    max_abs_val = np.max(np.abs(audio_data))
    if max_abs_val > 0:
        normalised_data = audio_data / max_abs_val
    else:
        normalised_data = audio_data # Already zero or empty
    return normalised_data


if __name__ == "__main__":
    # This is a simple example of how to use the functions.
    # In a real scenario, you'd likely have test audio files.
    print("Running audio_loader.py example...")
    try:
        # Create a dummy audio file for testing (or use a real one)
        # This part requires soundfile to create a dummy .wav
        import soundfile as sf
        dummy_sr = 44100
        dummy_duration = 3 # seconds
        # A simple sine wave as dummy audio
        t = np.linspace(0, dummy_duration, int(dummy_sr * dummy_duration), endpoint=False)
        dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t) # 440 Hz sine wave
        dummy_filepath = "dummy_audio.wav"
        sf.write(dummy_filepath, dummy_audio, dummy_sr)
        print(f"Created dummy audio file: {dummy_filepath}")

        # Test load_audio
        audio, sr = load_audio(dummy_filepath, sr=22050) # Load and resample
        print(f"Loaded audio: Shape={audio.shape}, Sample Rate={sr}")

        # Test normalise_audio
        original_max = np.max(np.abs(audio))
        normalised_audio = normalise_audio(audio)
        normalised_max = np.max(np.abs(normalised_audio))
        print(f"Original max amplitude: {original_max:.4f}")
        print(f"normalised max amplitude: {normalised_max:.4f}")
        assert np.isclose(normalised_max, 1.0) or np.isclose(normalised_max, 0.0), "Normalisation failed!"

        # Clean up dummy file
        import os
        os.remove(dummy_filepath)
        print(f"Cleaned up dummy audio file: {dummy_filepath}")

    except ImportError:
        print("Install 'soundfile' (pip install soundfile) to run this example fully.")
        print("Skipping dummy file creation/deletion, attempting with non-existent file.")
        try:
            load_audio("non_existent_file.wav")
        except FileNotFoundError:
            print("Successfully caught FileNotFoundError for non_existent_file.wav")
        except Exception as e:
            print(f"Caught unexpected error: {e}")
    except FileNotFoundError:
        print("Test file not found, skipping specific tests.")
    except Exception as e:
        print(f"An error occurred during the example execution: {e}")

    print("audio_loader.py example finished.")