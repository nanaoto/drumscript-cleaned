## `DrumScript`  Python Package Structure


<!--date_created: sun-15-june2025-->

<!--date_updated: weds-24-june-2025-->

---
### **Modular Structure**

```markdown
# Repository Structure

This document outlines the directory and file structure of the `DrumScript` project.


DrumScript/
├── .venv/                         # Python virtual environment (created by `uv venv`)
├── .gitignore                     # Specifies intentionally untracked files to ignore
├── pyproject.toml                 # Project metadata and dependencies (managed by `uv`)
├── README.md                      # Project overview and main documentation
├── requirements.txt               # Direct dependencies (generated/used by `uv pip install -r`)
├── LICENSE                        # Project licensing information (e.g., MIT License)
├── main.py                        # (Optional) Main entry point for the application
├── repository_structure.md        # This document
├── tests/                         # Unit and integration tests
│   └── test.mp3                   # Example audio file for testing modules
│   └── ...
├── DrumScript/                    # The main Python package
│   ├── __init__.py                # Makes 'DrumScript' a Python package
│   ├── audio_processor/           # Handles audio loading, normalisation, onset detection, and feature extraction
│   │   ├── __init__.py
│   │   ├── audio_loader.py        # Loads and normalises audio files
│   │   ├── onset_detector.py      # Detects drum hit onsets
│   │   └── feature_extractor.py   # Extracts features for classification
│   ├── drum_classifier/           # Machine learning module for drum sound classification
│   │   ├── __init__.py
│   │   ├── data_preparer.py       # Prepares audio features into a dataset for training
│   │   ├── drum_model.py          # Defines the drum classification model architecture (e.g., scikit-learn based)
│   │   └── model_trainer.py       # Script for training, evaluating, and saving the drum classification model
│   ├── training_data/             # Contains labelled audio files for model training
│   │   ├── kick/                  # Subdirectory for kick drum samples
│   │   │   ├── kick_001.wav
│   │   │   └── ...
│   │   ├── snare/                 # Subdirectory for snare drum samples
│   │   │   ├── snare_001.wav
│   │   │   ├── snare_002.mp3
│   │   │   └── ...
│   │   └── hihat/                 # Subdirectory for hi-hat samples
│   │       ├── hihat_001.wav
│   │       └── ...
│   ├── sheet_music_generator/     # (Future) Generates musical notation
│   │   └── __init__.py
│   │   └── ...
│   ├── utils/                     # Utility functions (e.g., for file handling, configuration)
│   │   └── __init__.py
│   │   └── ...
│   ├── how_it_works.md            # Documentation that explains how music theory and `DrumScript` interact
<!--│   └── ...
├── models/                        # **NEW:** Directory to store trained machine learning models, scalers, etc.
│   ├── drum_classifier_model.joblib  # Example: Saved trained classifier model
│   ├── scaler.joblib               # Example: Saved feature scaler
│   └── label_map.json              # Example: Saved mapping of numerical labels to drum types
└── documentation/                 # (Optional) Detailed documentation, tutorials, etc.
└── user_guide.md
└── ...--# commenting out for now--->
 
```
```
```


---
### **Modular Descriptions**

#### `audio_processor/`

This module handles all the raw audio manipulation and initial feature extraction.

<!--|Name|Purpose|
|-|-|
|**`audio_loader.py`**|Load audio files (`.wav`, `.mp3`, etc.).|
|**`onset_detector.py`**||
 **`feature_extractor.py`**||-->

* **`audio_loader.py`**:
    * **Purpose:** Load audio files (`.wav`, `.mp3`, etc.).
    * **Scientific Principle:** Digital audio files are sampled representations of analog sound waves. Loading involves reading these samples into a numerical array (e.g., NumPy array).
    * **Functions:**
        * `load_audio(file_path, sr=None)`: Loads audio, optionally resamples to a target sample rate (`sr`).
        * `normalise_audio(audio_data)`: Normalizes amplitude to a standard range (-1.0 to 1.0).
    * 

* **`onset_detector.py`**:
    * **Purpose:** Identify the precise time points where drum hits occur.
    * **Scientific Principle:** Onsets are characterized by rapid changes in spectral energy or amplitude over time (high spectral flux).
    * **Functions:**
        * `detect_onsets(audio_data, sr)`: Uses algorithms like `librosa.onset.onset_detect` (which might use spectral flux, peak picking, etc.) to return a list of onset times in seconds.
    * **Considerations:** Thresholding is critical. Too low, many false positives; too high, miss subtle hits (ghost notes).

* **`feature_extractor.py`**:
    * **Purpose:** Extract relevant features from audio segments around detected onsets that characterize drum sounds.
    * **Scientific Principle:** Timbre differences are encoded in the frequency content (spectrum) and the temporal evolution of the sound.
    * **Functions:**
        * `extract_features(audio_segment, sr)`: Takes a small audio segment (e.g., 200-500ms around an onset) and extracts features.
        * **Common Features (using `librosa`):**
            * **MFCCs (Mel-Frequency Cepstral Coefficients):** Excellent for timbre.
            * **Spectral Centroid:** Average frequency.
            * **Spectral Rolloff:** Skewness of the spectrum.
            * **Zero-Crossing Rate:** Roughness/noisiness.
            * **RMS Energy:** Loudness.
            * **Spectral Bandwidth:** Spread of frequencies.
            * **Pitch (for pitched elements like toms, or to *disregard* for unpitched ones):** While drums are unpitched, low-frequency content can indicate bass drum.
            * **Attack/Decay Characteristics:** Shape of the amplitude envelope.

#### `drum_classifier/`

This is the "brain" of the package, identifying which drum was hit.

* **`data_preparer.py`**:
    * **Purpose:** Prepare a dataset for training the drum classifier.
    * **Functions:**
        * `build_dataset(audio_files, annotations_files)`: Takes raw drum audio (ideally isolated hits or labeled full tracks) and corresponding labels (e.g., "kick," "snare," "hi-hat"). Extracts features for each labeled hit and stores them.
        * `split_data(features, labels)`: Splits data into training, validation, and test sets.
    * **Challenge:** Creating a robust, labeled dataset is the hardest part.  might need to manually label some initial drum samples or use existing datasets (e.g., Open Drum Kit, ENST-Drums).

* **`drum_model.py`**:
    * **Purpose:** Define the machine learning model architecture.
    * **Scientific Principle:** Machine learning models learn complex patterns and relationships between input features (spectral data) and output classes (drum type).
    * **Classes/Functions:**
        * `DrumClassifierModel()`: A class to encapsulate the ML model.
        * **Possible Models:**
            * **Traditional ML (Scikit-learn):** `RandomForestClassifier`, `SVC` (Support Vector Classifier). Simpler, good baseline.
            * **Deep Learning (TensorFlow/PyTorch):**
                * **CNN (Convolutional Neural Network):** Excellent for processing spectral images (spectrograms generated from features). Can learn hierarchical features directly from frequency-time representations.
                * **RNN/LSTM:** If  want to consider sequential context (e.g., hi-hat patterns).
                * **Transformer:** Emerging as powerful for sequential data like audio.

* **`model_trainer.py`**:
    * **Purpose:** Train and evaluate the drum classification model.
    * **Functions:**
        * `train_model(model, train_data, val_data, epochs, batch_size)`: Trains the `DrumClassifierModel`.
        * `evaluate_model(model, test_data)`: Evaluates performance (accuracy, precision, recall, F1-score per class).
        * `save_model(model, path)` / `load_model(path)`: For persistence.

#### `notation_generator/`

This module handles converting the identified drum events into a visual score.

* **`score_builder.py`**:
    * **Purpose:** Map detected and classified drum events to musical notation data.
    * **Scientific Principle:** Musical quantization (aligning events to a beat grid) and standard drum notation rules.
    * **Functions:**
        * `quantize_events(onset_times, tempo=120, subdivision=16)`: Takes raw onset times and snaps them to the nearest musical subdivision (e.g., 16th note). This is critical for readable sheet music. Requires tempo detection (or user input).
        * `map_to_drum_parts(classified_events)`: Converts classified drum hits ("kick", "snare") into standard drum notation elements (e.g., MIDI note numbers for notation software, or custom symbols).
        * `create_score_data(quantized_events)`: Assembles the raw data for the score (e.g., a list of `(time, drum_part, velocity)` tuples).
    * **Notation Mapping (Example):**
        * Kick: Lowest space/line
        * Snare: Middle space (often C or E)
        * Hi-Hat (closed): Top space (X-head)
        * Hi-Hat (open): Top space (X-head with circle)
        * Crash: Above top staff line
        * Ride: Top space (X-head)
        * Toms: Various spaces/lines

* **`pdf_exporter.py`**:
    * **Purpose:** Render the musical score data into a PDF.
    * **Libraries:**
        * `music21`: A powerful Python library for computer-aided musicology. Can represent musical objects and export to MusicXML, which can then be converted to PDF.
        * `LilyPond` (external): A powerful music engraving program. `music21` can often interface with it.
        * `PyPDF2` (for direct PDF manipulation if rendering manually, less common for complex scores).
        * `reportlab` (for drawing shapes, if  ant to draw the staff and notes manually, more complex).
    * **Functions:**
        * `generate_pdf(score_data, output_path)`: Takes the structured score data and generates a PDF. This will be the most visually challenging part.
    * **Considerations:** Staff lines, note heads, stems, flags, rests, clefs, time signatures, tempo markings, dynamic markings. `music21` simplifies this greatly.

#### `utils/`

Helper functions and configuration.

* **`config.py`**: Stores configuration parameters (e.g., default sample rate, model paths, feature extraction parameters).
* **`helpers.py`**: Small utility functions (e.g., `milliseconds_to_seconds`).
* **`constants.py`**: Defines constant values (e.g., drum instrument mappings to MIDI notes or custom IDs).

---

#### `main.py` (*`The Orchestrator`*)

<!--This script brings all the modules together.-->

 `main.py`serves as the **main entry point** or **orchestration script** for the entire `DrumScript` application. Specifically,  `main.py` is used for:

1.  **Orchestrating the full pipeline:** 
It brings together all the individual components of  `DrumScript`:

    - **Audio loading:** It loads the input drum audio file.
    - **Onset detection:** It detects when drum hits occur in the audio.
    - **Drum classification:** It loads previously trained drum classification model (from `drum_classifier.drum_model` and `model_trainer`), extracts features from the detected drum hits, and then classifies each hit (e.g., as a kick, snare, hi-hat).
    - **Notation generation:** It takes the classified drum events, quantizes them (ie. aligns them to a musical grid), and structures them into data suitable for sheet music.
    - **PDF generation:** Finally, it generates a PDF file of the drum sheet music.
    - **User Interface via Command Line:** It uses `argparse` to allow users **to specify the input audio file**, **the desired output PDF file path**, and (optionally) **the tempo**, directly from the **command line**.
>
In short, `main.py` **makes it easy** for someone to **run the entire conversion process** with a **single command**.

> `main.py` is the script  would run when you **want to convert a complete drum audio recording** into a drum **sheet music PD**F, utilising all the preceding steps (like data preparation and model training). It's the final piece that connects everything together for the end-user functionality.

> However, as the comments in `main.py` script point out, for `main.py` to work correctly,  **must first have a trained drum classification model** saved (using `model_trainer.py`)

---
<!--END--->