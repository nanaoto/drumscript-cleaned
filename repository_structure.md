## `DrumScript`  Python Package Structure


<!--date_created: sun-15-june2025-->

<!--date_updated: sun-15-june2025-->

---
### **Modular Structure**
```
DrumScript/
├── __init__.py
├── audio_processor/ # This module handles raw audio manipulation and initial feature extraction.
│   ├── __init__.py
│   ├── audio_loader.py
│   ├── feature_extractor.py
│   └── onset_detector.py
├── drum_classifier/ # determines model used to trascribe events
│   ├── __init__.py
│   ├── model_trainer.py
│   ├── drum_model.py
│   └── data_preparer.py
├── notation_generator/ # This module handles converting the identified drum events into a visual score
│   ├── __init__.py
│   ├── score_builder.py
│   └── pdf_exporter.py
├── utils/ # This module contains helper functions and configuration.
│   ├── __init__.py
│   ├── config.py
│   ├── helpers.py
│   └── constants.py
├── main.py                # The main entry point for running the conversion
├── requirements.txt
├── requirements.in
├── repository_structure.md
├── setup.py
└── README.md
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
    * **Challenge:** Creating a robust, labeled dataset is the hardest part. You might need to manually label some initial drum samples or use existing datasets (e.g., Open Drum Kit, ENST-Drums).

* **`drum_model.py`**:
    * **Purpose:** Define the machine learning model architecture.
    * **Scientific Principle:** Machine learning models learn complex patterns and relationships between input features (spectral data) and output classes (drum type).
    * **Classes/Functions:**
        * `DrumClassifierModel()`: A class to encapsulate the ML model.
        * **Possible Models:**
            * **Traditional ML (Scikit-learn):** `RandomForestClassifier`, `SVC` (Support Vector Classifier). Simpler, good baseline.
            * **Deep Learning (TensorFlow/PyTorch):**
                * **CNN (Convolutional Neural Network):** Excellent for processing spectral images (spectrograms generated from features). Can learn hierarchical features directly from frequency-time representations.
                * **RNN/LSTM:** If you want to consider sequential context (e.g., hi-hat patterns).
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
        * `reportlab` (for drawing shapes, if you want to draw the staff and notes manually, more complex).
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

This script brings all the modules together.



---
<!--END--->