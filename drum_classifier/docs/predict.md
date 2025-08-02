### Using `predict.py` to test the model

The following documentation outlines how to use the `predict.py` script in the `.drum_classifier/` module to test the outputs of a trained model

<!--date_created:mon-14-jul-2025-->
<!--date_updated:sat-02-aug-2025-->
-----

### `predict.py` - Multi-Label Drum Event Prediction (CNN Model)

This script is designed to perform multi-label drum event classification on audio files using your trained Convolutional Neural Network (CNN) model. It processes audio segments, identifies multiple drum types within each segment, and reports them with detailed metadata including MIDI pitches, staff positions, and display names.

### Location

Based on your execution command `python3 drum_classifier/predict.py`, this script is located within the `drum_classifier` module: `drum_classifier/predict.py`

### Prerequisites

Before running `predict.py`, ensure you have:

1.  **Processed ENST Dataset:** You must have successfully run `process_enst_data.py` to generate the segmented audio files and the `multi_label_events.csv` metadata in `training_data/ENST_processed/`.
2.  **Trained Model:** You must have successfully run `model_trainer.py` to train the multi-label CNN model and save the following files in the `models/` directory:
      * `multi_label_drum_classifier_model.h5` (the Keras CNN model)
      * `multi_label_scaler.joblib` (the fitted `StandardScaler`)
      * `multi_label_label_map.json` (the mapping from drum names to numeric labels)

### Usage

Run `predict.py` from the root directory of your `DrumScript` project.

The script currently includes two example sections: one for a single 0.2-second segment and one for a longer audio file. You will need to uncomment and/or modify the `long_audio_filepath` variable to point to your desired audio file.

#### Running the script

```bash
python3 drum_classifier/predict.py
```

#### Modifying the Test Audio

Open `drum_classifier/predict.py` and look for the `if __name__ == "__main__":` block.
You'll see lines similar to these for specifying the audio files:

```python
    # --- Example 1: Predict single segment ---
    single_segment_audio_path = os.path.join(project_root, "training_data", "ENST_processed", "1_rock-prog_125_beat_4-4.wav")

    # --- Example 2: Process a longer audio file ---
    long_audio_filepath = os.path.join(project_root, "test_audio", "test.wav")
    # You can change this path to your specific WAV or MP3 file
    # Example: long_audio_filepath = os.path.join(project_root, "your_custom_audio", "my_song.mp3")
```

**To test with a specific WAV or MP3 file:**

1.  Locate the line `long_audio_filepath = os.path.join(project_root, "test_audio", "test.wav")`
2.  Change `"test.wav"` and/or `"test_audio"` to the actual path of your audio file. For instance, if your file is `my_song.mp3` directly in your `DrumScript/` project root, you might change it to:
    `long_audio_filepath = os.path.join(project_root, "my_song.mp3")`

### How it Works

The script performs the following steps:

1.  **Loads Components:** It loads the pre-trained Keras CNN model (`.h5`), the `StandardScaler` (`.joblib`), and the `label_map` (`.json`) from the `models/` directory.
2.  **Loads Audio:** For the "Longer Audio File Prediction Example," it loads the specified audio file (`long_audio_filepath`), resampling it to `SAMPLE_RATE` (22050 Hz) if necessary.
3.  **Segments Audio:** It divides the long audio file into 0.2-second segments (`SEGMENT_LENGTH_SECONDS`), using a `HOP_LENGTH_SECONDS` (default: 0.1 seconds) to create overlapping windows. This ensures comprehensive analysis.
4.  **Extracts Features:** For each segment, it computes 40 MFCC (Mel-frequency cepstral coefficients) features, which are then scaled using the loaded `StandardScaler`.
5.  **Predicts Drum Events:** The scaled MFCCs are fed into the CNN model, which outputs probabilities for each of the 6 drum types (`kick`, `snare`, `hi-hat`, `crash`, `ride`, `tom`).
6.  **Thresholding:** Any drum type with a probability above a hardcoded threshold (e.g., 0.5) is considered "present" in that segment.
7.  **Creates Detailed Events:** For each detected drum type, the script creates a detailed event object containing:
   - `drum_type`: The detected drum type (e.g., "kick", "snare", "hi-hat")
   - `onset_time_seconds`: The timestamp when the drum event occurs
   - `midi_pitch`: The corresponding MIDI note number for the drum
   - `note_head_type`: The type of note head for music notation ("normal" or "x")
   - `staff_position`: The staff position for music notation (e.g., "F2", "C3")
   - `display_name`: A human-readable name for the drum (e.g., "Bass Drum", "Hi-Hat (Closed)")
8.  **Reports Results:** It prints a summary of detected drum events and exports the complete results to `prediction_output.json`.

### Output Format

The script generates a single output file called `prediction_output.json` containing detailed drum event information. Each event in the JSON array includes complete metadata for music notation and MIDI applications.

#### Example Output Structure

```json
[
    {
        "drum_type": "hi-hat",
        "onset_time_seconds": 0.0,
        "midi_pitch": 42,
        "note_head_type": "x",
        "staff_position": "F#3",
        "display_name": "Hi-Hat (Closed)"
    },
    {
        "drum_type": "snare",
        "onset_time_seconds": 0.3,
        "midi_pitch": 38,
        "note_head_type": "normal",
        "staff_position": "C3",
        "display_name": "Snare Drum"
    },
    {
        "drum_type": "kick",
        "onset_time_seconds": 0.5,
        "midi_pitch": 36,
        "note_head_type": "normal",
        "staff_position": "F2",
        "display_name": "Bass Drum"
    }
]
```

### Drum Type Mappings

The script includes comprehensive metadata for all supported drum types:

| Drum Type | MIDI Pitch | Note Head | Staff Position | Display Name |
|-----------|------------|-----------|----------------|--------------|
| kick      | 36         | normal    | F2             | Bass Drum    |
| snare     | 38         | normal    | C3             | Snare Drum   |
| hi-hat    | 42         | x         | F#3            | Hi-Hat (Closed) |
| crash     | 49         | x         | C#4            | Crash Cymbal |
| ride      | 51         | x         | D#4            | Ride Cymbal  |
| tom       | 45         | normal    | A3             | Tom-Tom      |

### Example Console Output

```
Loading trained model, scaler, and label map...
Model components loaded.

--- Single Segment Drum Prediction Example ---
Predicting drum types for single segment: .../1_rock-prog_125_beat_4-4.wav
Predicted drum types: ['hi-hat']

--- Longer Audio File Prediction Example (Detailed Output) ---
Processing long audio file: .../test.wav
Segmenting & Predicting: 100%|██████████| 589/589 [00:02<00:00, 294.50it/s]

--- Detected Drum Events (Detailed Format) ---
Event 0: hi-hat at 0.00s - Hi-Hat (Closed) (MIDI: 42)
Event 1: hi-hat at 0.10s - Hi-Hat (Closed) (MIDI: 42)
Event 2: snare at 0.30s - Snare Drum (MIDI: 38)
Event 3: hi-hat at 0.40s - Hi-Hat (Closed) (MIDI: 42)
... and 585 more events

Successfully exported detailed drum events to: .../prediction_output.json

--- Sample Detailed Events ---
Event 0: {
  "drum_type": "hi-hat",
  "onset_time_seconds": 0.0,
  "midi_pitch": 42,
  "note_head_type": "x",
  "staff_position": "F#3",
  "display_name": "Hi-Hat (Closed)"
}
```

---

# Using `predict.py` to Test Your Drum Classification Model

This guide explains how to use the enhanced `predict.py` script to test your trained DrumScript drum classification model on actual audio files. This script now provides detailed metadata suitable for music notation and MIDI applications.

### How to Use `predict.py`

The `predict.py` script is designed to load your trained model components and process audio files to detect drum events with comprehensive metadata.

#### 1\. Ensure Model Files Exist

Before running `predict.py`, confirm that the following trained model files are present in your `DrumScript/models/` directory:

  * `multi_label_drum_classifier_model.h5`
  * `multi_label_scaler.joblib`
  * `multi_label_label_map.json`

The `predict.py` script includes a built-in check for these files and will print an error message if any are missing.

#### 2\. Place Your Test Audio File

The script is configured by default to look for a test audio file at `DrumScript/test_audio/test.wav`.

  * If you have an audio file you wish to test, place it in the `DrumScript/test_audio/` directory.

  * If your file has a different name or format (e.g., `my_drum_beat.mp3`), you must update the `long_audio_filepath` variable within the `if __name__ == "__main__":` block of `predict.py`.

    **Example:**

    ```python
    long_audio_filepath = os.path.join(project_root, "test_audio", "my_drum_beat.mp3")
    ```

#### 3\. Run the Script

Navigate to the `DrumScript/drum_classifier/` directory in your terminal and execute the script using Python:

```bash
cd DrumScript/drum_classifier
python3 predict.py
```

### What the Script Does When You Run It

Upon execution, `predict.py` performs the following actions:

1.  **Load Components**: It loads your saved Keras model (`.h5`), the feature scaler (`.joblib`), and the drum label mapping (`.json`).

2.  **Single Segment Example (Optional)**: The script first attempts to run a prediction on a single, short audio segment. This example is configured to use `1_rock-prog_125_beat_4-4.wav` from your `training_data/ENST_processed/` directory. If this file isn't found, the script will prompt you to update the path or skip this part. This section is useful for quickly verifying basic functionality on a known, small input.

3.  **Longer Audio File Prediction**: This is the primary function for comprehensive testing.

      * It loads the audio from the `long_audio_filepath` you specified (e.g., `test.wav`).
      * The longer audio file is then **segmented** into small, overlapping chunks (0.2 seconds per segment, with a hop of 0.1 seconds).
      * For each segment, the script:
          * Extracts **MFCC features**.
          * **Scales** these features using the `multi_label_scaler.joblib`.
          * Feeds the scaled features into your **trained CNN model** (`multi_label_drum_classifier_model.h5`) to generate probability predictions for each drum type.
          * Applies a **threshold (0.5)** to these probabilities. If a drum type's predicted probability is above 0.5 for a given segment, that drum is considered present.
          * Creates **detailed event objects** with complete metadata for each detected drum type.
      * Finally, the script exports all detected drum events to `prediction_output.json` and prints a summary.

### Output File: `prediction_output.json`

The script generates a single comprehensive output file containing detailed information about each detected drum event. This file includes:

- **Complete timestamps** for each drum event
- **MIDI pitch numbers** for use with digital audio workstations
- **Music notation metadata** including note head types and staff positions
- **Human-readable display names** for each drum type

This detailed format makes the output suitable for:
- Music notation software
- MIDI sequencers and digital audio workstations
- Drum transcription applications
- Music analysis and research

The enhanced output provides everything needed to recreate the detected drum pattern in both musical notation and digital audio formats.

---

<!--END-->
