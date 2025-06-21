## Testing Documentation

<!--date_created: sat-21-june-2025-->
<!--date_updated: sat-21-june-2025-->

`DrumScript` is a `Python package` that **converts drum audio recordings into sheet music (drum notation) in PDF format**. 

It leverages **advanced audio signal processing** and **machine learning** to detect individual drum hits (kick, snare, hi-hat, etc.) and translate them into a musical score.

**[Package Structure](repository_structure.md)** **[~ README](README.md)**

---
### Modules

#### **`audio_processor/`**

#####  `audio_loader.py`


```python 
python3 audio_processor/audio_loader.py
```


**expected output:**

    Running audio_loader.py example with actual MP3/WAV...
    Attempting to load: ~DrumScript/tests/test.mp3
    Loaded audio: Shape=(324288,), Sample Rate=22050, Duration=14.71 seconds
    Original max amplitude: 0.9660
    Normalised max amplitude: 1.0000
        
    Playing loaded (and normalised) audio...
        
    # test.mp3 will play


    Audio playback finished.
    audio_loader.py example finished.

```python
python3 audio_processor/feature_extractor.py
```

##### `feature_extractor.py`
**expected output:**

    Running feature_extractor.py example...
    Install 'soundfile' (pip install soundfile) to run this example fully.
    Skipping dummy file creation/deletion.
    feature_extractor.py example finished.

##### `onset_detector.py`
```python
python3 audio_processor/onset_detector.py
```

**expected output:**

    Running onset_detector.py example...
    Created dummy audio file with hits: dummy_audio_with_hits.wav
    Original hit times: [0.5, 1.2, 1.8, 2.5, 3.1, 3.8, 4.4]
    Detected onsets (seconds): ['0.51', '1.21', '1.81', '2.51', '3.11', '3.81', '4.41']
    Cleaned up dummy audio file: dummy_audio_with_hits.wav
    onset_detector.py example finished.


---
<!--END-->