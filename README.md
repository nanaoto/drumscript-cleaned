## **DrumScript**

<!--date_created: sun-15-june-2025-->
<!--date_updated: sat-23-july-2025-->

> **Python 3.12.10**

`DrumScript` is a `Python package` that converts **drum audio** into **sheet music (drum notation)** in PDF format. 

It leverages **advanced audio signal processing** and **machine learning** to detect individual drum hits (*kick*, *snare*, *hi-hat*, etc.) and translate them into a **musical score**, either as `.XML` or `.pdf`.

***[hello.drumscript@gmail.com](hello.drumscript@gmail.com)***

---

 - **[Features](#features-1)**
 - **[Installation](#installation)**
 - **[Usage](#usage)**
   - **[Arguments](#arguments)**
   - **[Options](#options)**
   - **[Example](#example)**
 - **[Model Training](#model-training)**
 
 <!--- **[License](#license)**-->
 - **[Contributing](#contributing)**
 - **[Contact](#contact)**
 - **[FAQs](#faqs)**

> **NOTE:** See **[how_it_works.md](how_it_works.md)** if you are interested in the link between **music theory** and `DrumScript` structure. 

> **NOTE:**  For simplicity, `DrumScript` uses the generic spelling of the word ***quantize***, rather than British English spelling ***quantise***. 

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
>
**LilyPond Installation**

To generate high-quality PDF sheet music, DrumScript utilises `music21` which, in turn, relies on an external music engraving program called **LilyPond**. You'll need to install LilyPond separately on your system for PDF export functionality to work.

  * **Download & Install LilyPond:**

      * Visit the official LilyPond website: [https://lilypond.org/](https://lilypond.org/) and download the appropriate version for your operating system.
      * **macOS Users:** LilyPond can be easily installed via [Homebrew](https://brew.sh/) by running:
        ```bash
        brew install lilypond
        ```

  * **Verify Installation:**
    You can verify LilyPond is correctly installed and accessible by opening your terminal/command prompt and typing:

    ```bash
    lilypond --version
    ```

    This should display the installed LilyPond version information.

`music21` will automatically try to locate your LilyPond installation. If you encounter errors during PDF generation, you might need to manually configure the LilyPond path within `music21` (refer to the `DrumScript/notation_generator/pdf_exporter.py` file for guidance on this).

---
### Dependencies
A list of dependencies can be found in **[`requirements.txt`](requirements.txt)** and in **[`pyproject.toml`](pyproject.toml)**


---
> **Important  note for audio playback [`sounddevice`](https://python-sounddevice.readthedocs.io/en/0.5.1/):** 
        >
        > **`sounddevice`** relies on a system-level library called **[PortAudio](https://www.portaudio.com/). You might need to install this separately**:
        >
        > * **macOS:** `brew install portaudio` (if you have Homebrew)
        > * **Ubuntu/Debian:** `sudo apt-get install libportaudio2`
        > * **Windows:** `PortAudio` usually comes bundled with `sounddevice` wheels, but you might need to find specific installation instructions for your Python distribution.
        > 
        > **[`PortAudio` source code](https://github.com/PortAudio/portaudio)**



---
> **Important note for `.mp3` users**

**See also **[FAQs](#faqs)****

In order to use `.mp3` files, users must **first install** [`FFmpeg`](), a free command-line tool downloaded via [`Homebrew`](https://brew.sh/) designed for processing video and audio files. 

##### Installing [`FFmpeg`](https://ffmpeg.org/download.html)

**How to install [`FFmpeg`](https://ffmpeg.org/download.html) via `DrumScript`**

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
**How to install `FFmpeg` manually**
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

The initial model provided with this package was trained using the **ENST-Drums database**.

#### Training Data: ENST-Drums Dataset

The model was trained on the ENST-Drums dataset, a comprehensive audio-visual database of drum performances created for research purposes. This extensive collection features recordings from three professional drummers on their own kits, using various sticks (sticks, rods, brushes, and mallets) to capture a wide diversity of sounds.

The dataset is fully annotated, providing precise timings for each drum event, which makes it an excellent resource for training machine learning models for drum transcription.

**Link to the original paper**:

>**[***ENST-Drums: an extensive audio-visual database for drum signals processing***. Olivier Gillet and Gaël Richard. ***In Proc of the 7th International Society for Music Information Retrieval Conference (ISMIR'06), Victoria, Canada, 2006.***](https://archives.ismir.net/ismir2006/paper/000027.pdf)**

> You can find more information and download the dataset from the official webpage: ***[http://www.enst.fr/\~grichard/ENST-drums/](http://www.enst.fr/~grichard/ENST-drums/)***.

--

Instructions for training the model with your own data will be provided in a dedicated section (e.g., `drum_classifier/README.md`) within the `drum_classifier/` directory. This typically involves:

1.  Preparing a labeled dataset of drum sounds.
2.  Running a training script (e.g., `python drum_classifier/train_model.py`).

---

### Contributing

We welcome contributions to `DrumScript`! 

If you have ideas for improvements, bug fixes, or new features, please **[open an issue](https://github.com/victoria-mckinney/DrumScript/issues/new)** or submit a **[pull request](https://github.com/victoria-mckinney/DrumScript/pulls)**.

####  Training the Model
Instructions for training the model with your own data will be provided in a dedicated section (e.g., `drum_classifier/README.md`) within the `drum_classifier/` directory. This typically involves:

1.  Preparing a labeled dataset of drum sounds.
2.  Running a training script (e.g., `python drum_classifier/train_model.py`).



<!--
---### License # Comment out again once license is chosen/added

This project is licensed under the **MIT License**. See the **[`LICENSE` file]()** for details.-->

---

### Contact

For questions or support, please **[open an issue](https://github.com/victoria-mckinney/DrumScript/issues)** on the **[GitHub repository](https://github.com/victoria-mckinney/DrumScript)**, or email the code maintainers:

***[hello.drumscript@gmail.com](hello.drumscript@gmail.com)***

---

### FAQs

 - #### Why don't you just include `FFmpeg` as a dependency rather than a specific download requirement?
    
    We cannot directly bundle or automatically install [`FFmpeg`](https://ffmpeg.org/download.html) at the **Python project dependency level** (via `uv pip install` or or in `pyproject.toml`) for all operating systems.

 - #### Is it safe to install system-wide dependencies using `DrumScript`, ie. `drumscript.install_ffmpeg()`?

    Utility functions like `drumscript.install_ffmpeg()` are wrappers that serve **as an aid**. They are helpers than **run on every OS**.  However, usage is **optional**.Users are welcome to follow **[manual instructions for installing `FFmpeg`](#how-to-install-ffmpeg-manually)** provided.
    >
    **Note for `Linux`/`macOS`**: For `sudo` commands, `shell=True` is often used. It's important to be cautious, ie with `shell=True` as it can introduce security risks if the command being executed includes untrusted input. In this case, the commands are **hardcoded** and **safe*

 - #### What normalisation is applied to loaded audio?

    Audio is loaded into `DrumScript` using the `.audio_processor/audio_loader.py` script. The `audio_loader.py` script applies **peak normalisation** to the audio after loading it. The process is straightforward and happens in two main steps inside the `load_audio` function:
>  1.  First, the script loads the audio file and converts it to a mono signal.
> 2.  It then immediately passes this audio data to the `normalise_audio` function, which uses the `librosa.util.normalise(y)` command to perform the normalisation.


 - #### **What is `peak normalisation`? 🔊**

    **Peak normalisation** is a process that adjusts the volume of an audio file so that its loudest point—the "peak"—is set to a maximum level (in this case, 1.0) without clipping or distortion.
    >
    Think of it like adjusting the brightness of a group photo. The software finds the single brightest spot in the entire image and adjusts the overall brightness of the photo so that this one spot is pure white. Every other part of the image is brightened by the same amount, preserving all the relative differences in light and shadow.
    >
    Peak normalisation does the same for audio. It finds the loudest drum hit in the entire track and boosts the volume of the whole file so that this single hit is at maximum loudness. The relative volume between all other hits is perfectly preserved. 

 - #### Why is `peak_normalisation` used in `DrumScript`?

    Applying peak normalisation is a crucial step for preparing data for a machine learning model. By ensuring that every audio file has a consistent peak volume, it prevents the model from being biased by how loud or quiet the original recording was.
    >
    This forces the model to learn the **timbre** and **texture** of a snare drum, for example, rather than just learning that "loud sounds are snare drums." It ensures the model makes classifications based on the sonic character of the instruments, n
 - ####  Why normalise imported audio at all? 

    The reason we normalise is to make the machine learning model robust to differences in recording volume.
    >
    Without normalisation, our model might learn that a snare drum is a sound with a high RMS (loudness) value. But what happens when it analyses a track that was recorded very quietly? The snare's RMS value might be lower than a kick drum's from a different, louder recording. The model would get confused.
    >
    By normalising every audio file, we remove the variable of recording level. This forces the model to learn the true sonic fingerprint of each drum—its timbre, its frequency content, its attack—which remains consistent regardless of how loud it is.

 - ####  Don't you lose detail by normalising imported audio?

    The peak normalsation used in our `audio_loader.py` script is a **linear process**. This means it multiplies **every single audio sample** by the **exact same constant value**. It's the digital equivalent of turning a volume knob.
    >
    **Analogy:** Think of it like changing the brightness of a photograph. You make every pixel brighter by the same amount, but you don't lose any detail. The shapes, textures, and relative differences between light and dark areas remain perfectly intact. The photo is just brighter. The same is true for your audio. The "shape" of the sound wave—which determines all the features like `MFCCs`, `spectral centroid`, and `rhythm`—is completely preserved. The audio is just made louder or quieter to fit a standard level.

 - #### What is is `hop_length`? 

      **`hop_length`** is the **number of audio samples** between the **start of one analysis window** and the **start of the nex**t.

      When `librosa` analyses audio, it doesn't look at the whole file at once; instead, it slides a small ***window* across the audio** and **analyses each chunk.** The `hop_length` is the distance that window ***hops*** forward for each step. 
      >
      * A **large** `hop_length` means fewer, bigger jumps, which is faster but less detailed.
      >
      * A **small** `hop_length` means more, smaller steps with more overlap, giving a much more detailed and precise analysis of time.
      >
      For example, if the `hop_length` has a value of **`256`**, this corresponds to **256 audio samples**. We can translate this into seconds with the designated `SAMPLE_RATE(sr)` in `DrumScript`: **`sample_rate=44100`**
      >
      $$\frac{256 \text{ samples}}{44100 \text{ samples per second}} \approx 0.0058 \text{ seconds (or 5.8 milliseconds)}$$
      >
      If, however, we were to change **`sample_rate=22050`**, this would become:
      >
      $$\frac{256 \text{ samples}}{22050 \text{ samples per second}} \approx 0.01160 \text{ seconds (or 11.6 milliseconds)}$$
      >

      

---

<!--END -->
