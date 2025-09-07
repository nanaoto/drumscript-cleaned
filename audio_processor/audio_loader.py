# DrumScript/audio_processor/audio_loader.py

"""
This module will handle loading and basic normalisation of audio files. It also offers automatic playback of the audio once loaded.
"""


import librosa
import numpy as np
import os
import sounddevice as sd
import time # for pausing listenable audio
import scipy
import threading
import matplotlib
matplotlib.use('Agg') # Set the backend to 'Agg' to prevent conflicts between plotting and threading
from scipy.signal import find_peaks
from scipy.stats import norm
import matplotlib.pyplot as plt
import librosa.display

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
        audio_data, sample_rate = librosa.load(file_path, sr=sr) # The librosa.load_audio() fct handles wide variety of audio formats, including .mp3, .wav, .flac, .ogg, etc.
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
"""

# --- REFINED TEMPO DETECTION LOGIC --- MIGHT REMOVE LATER

def detect_onsets_high_resolution(audio_data: np.ndarray, sr: int) -> np.ndarray:
"""
#Detects onsets with a higher temporal resolution suitable for fast music.
"""
# hop_length=256 is smaller than the default 512, giving us more detail.
onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr, units='frames', hop_length=256)
return librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)

def estimate_tempo_from_onsets(onset_times: np.ndarray, sr: int) -> float:
"""
#Estimates tempo from a list of onset timestamps.
"""
if len(onset_times) < 2:
    #return 120.0 # Default tempo
    return print('\n Not enough onsets to detect tempo automatically!')
    
tempo = librosa.beat.tempo(onset_envelope=None, sr=sr, onset_events=onset_times)
return tempo[0]
"""


# Tempo logic function 1: ------------------------------------------------------------------------

# --- adding function for Automatic Tempo Detection:
# --- replacing beat detection algorithm

def estimate_tempo(audio_data, sr):
    """
    Estimates tempo by finding the most musically plausible peak in the
    histogram of inter-onset intervals.
    """
    if audio_data.size == 0:
        return 0.0

    # Step 1: Detect onsets with high resolution (unchanged)
    onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr, units='frames', hop_length=256)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)

    if len(onset_times) < 2:
        #return 120.0
        return print(f'Not enough intervals to detect tempo automatically!')

    # Step 2: Calculate inter-onset intervals (unchanged)
    iois = np.diff(onset_times)

    # Step 3: Create a histogram of the IOIs to find common rhythmic values
    # We use a large number of bins for high precision.
    counts, bin_edges = np.histogram(iois, bins=100, range=(0.0, 2.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Step 4: Find the prominent peaks in the histogram
    peaks, _ = find_peaks(counts, height=np.max(counts)*0.1) # Find peaks > 10% of max height

    if len(peaks) == 0:
        return 60.0 / np.median(iois) # Fallback to simple median if no peaks found

    # Step 5: Score the peaks based on musical plausibility
    # We create a "prior" that prefers tempos around 120 BPM.
    peak_iois = bin_centers[peaks]
    candidate_tempos = 60 / peak_iois
    
    # A Gaussian curve centered at 120 BPM with a standard deviation of 30
    # This gives higher scores to more "typical" tempos.
    prior = norm(loc=130, scale=40)
    scores = prior.pdf(candidate_tempos)

    # Step 6: Choose the best tempo
    # The best tempo is the candidate with the highest score.
    best_tempo = candidate_tempos[np.argmax(scores)]
    
    return best_tempo


# Tempo logic function 2: ------------------------------------------------------------------------

def _estimate_tempo(audio_data, sr):
    """
    Estimates tempo using beat_track and then uses a tempogram as a
    "sanity check" to correct for common triplet errors.
    """
    if audio_data.size == 0:
        return 0.0
    
    # 1. Get initial tempo guess from beat_track
    initial_tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
    initial_tempo = initial_tempo[0]

    # 2. Compute tempogram to find the strongest overall tempo pulse
    oenv = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=256)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=256)
    
    # Sum energy across time to get a static tempo "spectrum"
    tempo_spectrum = np.sum(tempogram, axis=1)
    
    # Find the tempo with the most energy
    strongest_peak_idx = np.argmax(tempo_spectrum)
    tempo_freqs = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=256)
    strongest_tempo = tempo_freqs[strongest_peak_idx]

    # 3. The heuristic: If beat_track's guess is 2/3 of the strongest pulse,
    #    it's likely a triplet error. We correct it to the stronger pulse.
    #    (We check within a tolerance of +/- 5 BPM)
    if np.isclose(initial_tempo, strongest_tempo * 2/3, atol=5):
        return strongest_tempo
    
    # 4. Otherwise, we trust beat_track's initial guess.
    return initial_tempo

# Tempo logic function 3: ------------------------------------------------------------------------

def __estimate_tempo(audio_data, sr):
    """
    Estimates tempo from the tempogram, but restricted to a plausible range.
    (Corrected to avoid INF and extreme BPM errors, ie 10500 BPM).
    """
    if audio_data.size == 0:
        return 0.0
    oenv = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=256)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=256)
    tempo_spectrum = np.sum(tempogram, axis=1)
    tempo_freqs = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=256)
    
    # --- FIX for extreme BPM error ---
    # Create a mask to only consider tempos in a plausible musical range (e.g., 60-240 BPM)
    plausible_tempos_mask = (tempo_freqs >= 60) & (tempo_freqs <= 240)
    
    # Find the index of the peak within the plausible range
    plausible_spectrum = tempo_spectrum[plausible_tempos_mask]
    if plausible_spectrum.size == 0:
        return 120.0 # Return default if no energy in plausible range
        
    peak_idx_in_plausible_range = np.argmax(plausible_spectrum)
    
    # Convert that index back to a BPM value
    plausible_tempo_freqs = tempo_freqs[plausible_tempos_mask]
    estimated_bpm = plausible_tempo_freqs[peak_idx_in_plausible_range]
    
    return estimated_bpm
## --- NOTE: Review keep/discard/move to own script once finished using for testing of automatic tempo detection functions 
# --- New function for visualising the tempo across audio (while testing) ------------------------

def visualise_tempogram(audio_data, sr, hop_length=256, output_path="tempogram.png"):
    """
    Calculates and saves a tempogram visualization for the given audio.
    """
    oenv = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    global_tempo = __estimate_tempo(audio_data, sr)

    fig, ax = plt.subplots(figsize=(12, 6))
    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, 
                             x_axis='time', y_axis='tempo', cmap='magma', ax=ax)
    ax.axhline(global_tempo, color='w', linestyle='--', alpha=0.8, label=f'Global Tempo: {global_tempo:.2f} BPM')
    ax.set_title('Tempogram')
    ax.legend(loc='upper right')
    fig.colorbar(ax.get_children()[0], ax=ax, label='Energy')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    print(f"✅ Tempogram saved to: {output_path}")
    
# -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # We need the threading module to listen for input in the background
    import threading

    print("\n#============================================================================================================")

    print("Running audio_loader.py example with actual MP3/WAV...")

    # --- Path to your audio file ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    #actual_drum_recording_path = os.path.join(project_root,"DrumScript/test_audio","test.wav")
    actual_drum_recording_path = os.path.join(project_root,"DrumScript/test_audio","test5__195bpm.mp3")

    try:
        print(f"Attempting to load: {actual_drum_recording_path}")
        audio, sr = load_audio(actual_drum_recording_path, sr=44100)
        
        print(f"Loaded audio: Shape={audio.shape}, Sample Rate={sr}, Duration={len(audio)/sr:.2f} seconds")

        # Normalise and check the audio
        normalised_audio = normalise_audio(audio)
        normalised_max = np.max(np.abs(normalised_audio))
        assert np.isclose(normalised_max, 1.0) or np.isclose(normalised_max, 0.0), "Normalisation failed!"

       # Call all three tempo estimation functions
        bpm1 = estimate_tempo(normalised_audio, sr) # Filtered Median, estimate_tempo function above
        bpm2 = _estimate_tempo(normalised_audio, sr) # Smart Heuristic, _estimate_tempo function above
        bpm3 = __estimate_tempo(normalised_audio, sr) # Tempogram-First, __estimate_tempo function above

        print(f"       1. Estimated Tempo (Filtered Median): {bpm1:.2f} BPM")
        print(f"       2. Estimated Tempo (Smart Heuristic): {bpm2:.2f} BPM")
        print(f"       3. Estimated Tempo (Tempogram-First): {bpm3:.2f} BPM")


        ## --- NOTE: Review keep/discard/move to own script once finished using for testing of automatic tempo detection functions 
        # --- New function for visualising the tempo across audio (while testing) ------------------------
        # --- Call the visualisation function ------------------------------------------------------------
        print("\nGenerating and saving tempogram visualization...")
        # We save the image in the same directory as the script
        output_image_path = os.path.join(current_script_dir, "tempogram.png")
        visualise_tempogram(normalised_audio, sr, output_path=output_image_path)

        # --- Keyboard Interruption Logic -----------------------------------------------------------------
        # 1. Define a function that waits for Enter and then stops the audio
        def stop_playback_on_enter():
            input("Audio is playing. Press Enter to stop...\n")
            sd.stop()
            print("Playback stopped by user.")

        # 2. Start the audio playback (this is non-blocking)
        print("\nPlaying loaded (and normalised) audio...")
        sd.play(normalised_audio, sr)

        # 3. Start the listener function in a background thread
        #    'daemon=True' means the thread won't prevent the script from exiting
        listener_thread = threading.Thread(target=stop_playback_on_enter, daemon=True)
        listener_thread.start()

        # 4. Wait for playback to finish (either naturally or by being stopped)
        sd.wait()
        
        print("Audio playback finished.")
        # --------------------------------------------------------------------------------------------------

    except Exception as e:
        print(f"\nAn unexpected error occurred during the example execution: {e}")

    print("audio_loader.py example finished.")
    print("\n#============================================================================================================")