# DrumScript/audio_processor/onset_detector.py

"""
This module will detect the onset (start) times of drum hits in the audio.
"""

import librosa
import numpy as np
import os # Import os for path manipulation

def detect_onsets(audio_data: np.ndarray, sr: int) -> list[float]:
    # ... (Your existing detect_onsets function code remains unchanged) ...
    if audio_data.size == 0:
        return []

    onset_env = librosa.onset.onset_detect(y=audio_data, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_env, sr=sr)
    return onset_times.tolist()


if __name__ == "__main__":
    print("Running onset_detector.py example with test.mp3...")
    try:
        # Import necessary modules from your package
        # Note: You might need 'from DrumScript.audio_processor.audio_loader import ...'
        # if running this script directly and 'audio_processor' is not in the Python path.
        # However, for 'python -m' style execution, 'from audio_processor.audio_loader import ...' is usually correct.
        from audio_processor.audio_loader import load_audio, normalise_audio

        sr = 22050 # Target sample rate for processing

        # --- Path to your actual drum recording (test.mp3) ---
        # This dynamic path calculation should correctly point to DRUMSCRIPT/tests/test.mp3
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels from audio_processor/onset_detector.py to the outer DRUMSCRIPT/ folder
        project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
        # Construct the path to test.mp3 within the 'tests' directory
        test_mp3_path = os.path.join(project_root, "DrumScript/tests", "test.mp3")


        print(f"Attempting to load: {test_mp3_path}")
        # Load and normalise the test.mp3 audio
        audio_data, sample_rate = load_audio(test_mp3_path, sr=sr)
        normalised_audio = normalise_audio(audio_data)
        print(f"Loaded audio: Shape={normalised_audio.shape}, Sample Rate={sample_rate}, Duration={len(normalised_audio)/sample_rate:.2f} seconds")

        # Detect onsets from test.mp3
        print("\nDetecting onsets from test.mp3...")
        onsets = detect_onsets(normalised_audio, sample_rate)
        print(f"Detected {len(onsets)} onsets.")

        if onsets:
            # Print the first few detected onsets for verification
            print("\nFirst 10 detected onsets (seconds):")
            for i, onset_time in enumerate(onsets[:10]):
                print(f"  Onset {i+1}: {onset_time:.2f}s")
            if len(onsets) > 10:
                print(f"  ...and {len(onsets) - 10} more onsets.")
        else:
            print("No onsets detected in test.mp3.")

    except FileNotFoundError:
        print(f"\nERROR: The audio file '{test_mp3_path}' was not found.")
        print("Please ensure you have placed 'test.mp3' inside your 'DrumScript/tests/' directory.")
    except ImportError as e:
        print(f"\nERROR: Required modules/libraries might be missing or imports are incorrect: {e}")
        print("Ensure 'soundfile', 'librosa', 'numpy', and your DrumScript modules are correctly installed and structured.")
        print("For MP3, 'ffmpeg' must also be installed on your system and accessible in PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the example execution: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

    print("\nonset_detector.py example finished.")