## DrumScript

<!--date_created: sun-15-june-2025-->
<!--date_updated: sat-21-june-2025-->

###### Python 3.12.10

`DrumScript` is a `Python package` that **converts drum audio recordings into sheet music (drum notation) in PDF format**. 

It leverages **advanced audio signal processing** and **machine learning** to detect individual drum hits (kick, snare, hi-hat, etc.) and translate them into a musical score.

**[Package Structure](repository_structure.md)**

<!-- **NOTE:** This package requires `Python 3.12+`-BLOT-OUT-FOR-NOW-AS-MIGHT-CHANGE-->


 - **[Features](#features-1)**
 - **[Installation](#installation)**
 - **[Usage](#usage)**
   - **[Arguments](#arguments)**
   - **[Options](#options)**
   - **[Example](#example)**
 - **[Model Training](#model-training)**
 - **[Contributing](#contributing)**
 <!--- **[License](#license)**-->
 - **[Contact](#contact)**
 - **[FAQ's](#faqs)**

---

### Features

* **Audio Input:** Supports common audio formats such as `.mp3`
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
### Dependencies
>
A list of dependencies can be found in the **[requirements](requirements.txt)** and in **[pyproject.toml](pyproject.toml)**


---
### Important note for `.mp3` users

**See also **[FAQs](#faqs)****

In order to use `.mp3` files, users must **first install** [`FFmpeg`](), a free command-line tool downloaded via [`Homebrew`](https://brew.sh/) designed for processing video and audio files. 

#### How to install [`FFmpeg`](https://ffmpeg.org/download.html) via `DrumScript`

```python

# In your main.py, or any script where you want to ensure FFmpeg is present
import drumscript

# sudo on Linux/macOS: The automated installation attempts for Linux and macOS will prompt the user for their sudo password. This is necessary for system-wide package installations. The user must explicitly type 'y' to agree.

# Call the utility function
drumscript.install_ffmpeg()

#  automating PATH modification on Windows is significantly more complex and error-prone from a Python script, especially persistently for all users or without administrator privileges. Therefore, for Windows, the function provides clear manual instructions only.

# Now proceed with loading audio, etc.
# audio_data, sr = drumscript.audio_processor.audio_loader.load_audio("your_drum_track.mp3")
```

#### How to install `FFmpeg` manually
- **macOS (using [Homebrew](https://brew.sh/))**

```Bash
brew install ffmpeg
```

 - **Linux (using apt for Debian/Ubuntu)**

``` Bash
sudo apt update
sudo apt install ffmpeg
```

 - **Windows:**
  > This is a bit more involved. You typically download the `FFmpeg` binaries (as a `.zip` file) from the **[official website](ffmpeg.org/download.html)**, extract them, and then add the `bin` directory of the extracted `FFmmpeg` folder to your system's `PATH` environment variable. There are many tutorials online for this specific step on Windows.


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

#### Example

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

If you have ideas for improvements, bug fixes, or new features, please **[open an issue](https://github.com/victoria-mckinney/DrumScript/issues/new)** or submit a **[pull request](https://github.com/victoria-mckinney/DrumScript/pulls)**.


<!--
---### License # Comment out again once license is chosen/added

This project is licensed under the **MIT License**. See the **[`LICENSE` file]()** for details.-->

---

### Contact

For questions or support, please **[open an issue](https://github.com/victoria-mckinney/DrumScript/issues)** on the **[GitHub repository](https://github.com/victoria-mckinney/DrumScript)**.


---

### FAQs

 - #### Why don't you just include `FFmpeg` as a dependency rather than a specific download requirement?
    
    We cannot directly bundle or automatically install [`FFmpeg`](https://ffmpeg.org/download.html) at the **Python project dependency level** (via `uv pip install` or or in `pyproject.toml`) for all operating systems.

 - #### Is it safe to install system-wide dependencies using `DrumScript`, ie. `drumscript.install_ffmpeg()`?

    Utility functions like `drumscript.install_ffmpeg()` are wrappers that serve **as an aid**. They are helpers than **run on every OS**.  However, usage is **optional**.Users are welcome to follow **[manual instructions for installing `FFmpeg`](#how-to-install-ffmpeg-manually)** provided.
    >
    **Note for `Linux`/`macOS`**: For `sudo` commands, `shell=True` is often used. It's important to be cautious, ie with `shell=True` as it can introduce security risks if the command being executed includes untrusted input. In this case, the commands are **hardcoded** and **safe**.


---

<!--END -->
