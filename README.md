## DrumScript

<!--date_created: sun-15-june-2025-->
<!--date_updated: sun-15-june-2025-->

`DrumScript` is a `Python package` that **converts drum audio recordings into sheet music (drum notation) in PDF format**. 

It leverages **advanced audio signal processing** and **machine learning** to detect individual drum hits (kick, snare, hi-hat, etc.) and translate them into a musical score.

**[Package Structure](repository_structure.md)**

<!-- **NOTE:** This package requires `Python 3.12+`-BLOT-OUT-FOR-NOW-AS-MIGHT-CHANGE-->

---

### Features

* **Audio Input:** Supports common audio formats like WAV.
* **Drum Hit Detection:** Identifies the precise timing of drum strikes.
* **Drum Classification:** Differentiates between various drum kit elements (e.g., kick, snare, hi-hat).
* **Musical Quantization:** Aligns detected drum hits to a musical grid for accurate notation.
* **PDF Sheet Music Output:** Generates clear and readable drum notation.


---

### Installation

`DrumScript` uses `uv` for efficient dependency management and project setup.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/DrumScript.git
    cd DrumScript
    ```

2.  **Create a virtual environment and install dependencies using `uv`:**

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    uv pip install -r requirements.txt
    ```

> **NOTE:** 
> Other `Python` package managers, ie `pip`, `conda`, `pyenv`, `hatch`, `poetry` may be used interchangeably with the commands above.

---

### Usage

Once installed, you can run `DrumScript` from the command line.

```bash
python main.py <input_audio_file.wav> <output_sheet_music.pdf> [OPTIONS]

```

#### Arguments

* `<input_audio_file.wav>`: The path to your drum audio recording.
* `<output_sheet_music.pdf>`: The desired path for the generated PDF sheet music.

#### Options

* `--tempo <BPM>`: (Optional) Specify the tempo of the drum performance in Beats Per Minute (BPM). If not provided, `DrumScript` will attempt to detect the tempo or use a default.

**Example**

```bash
python main.py my_drum_track.wav drum_score.pdf --tempo 120
```

---

### Model Training

`DrumScript` relies on a machine learning model to classify drum sounds. For optimal performance, you might need to train or fine-tune this model with your own data.

Instructions for training the model will be provided in a dedicated section (e.g., `drum_classifier/README.md`) within the `drum_classifier/` directory. This typically involves:

1.  Preparing a labeled dataset of drum sounds.
2.  Running a training script (e.g., `python drum_classifier/train_model.py`).

---

### Contributing

We welcome contributions to `DrumScript`! 

If you have ideas for improvements, bug fixes, or new features, please **[open an issue]()** or submit a **[pull request]()**.

---

### License

This project is licensed under the **MIT Licens**e. See the **[`LICENSE` file]()** for details.

---

### Contact

For questions or support, please **[open an issue]()** on the **[GitHub repository]()**.


---

<!--END -->
