# **`DrumScript`: How it works**


<!--date_created: tues-24-june-2025-->
<!--date_updated: thurs-03-july-2025-->
**Description:** This document serves dual purpose:

1) It outlines the purpose of **specific modules** in `DrumScript`, object-wise.
2) It outlines key definitions to **clarify the methodology used**.

---
## `DrumScript`: How it works
## `audio_processor/`

**Module overview** 

 - Find where the **drum hits actually occur** (ie. `onset_events`)
 - Extract **numerical features** from these *events*

---

## `drum_classifier/`

**Module overview** 

- Prepares the audio features imported using `audio_processor` module **into a format that a machine-learning model** can recognise.
- Uses the **transformed, structured training data** to **train** a chosen machine-learning model.
- Compiles an **entirely new** model based on model training (`model_trainer.py`) that can make **predictive** judgements on **new audio inputs**

### `data_preparer.py`

This is the **raw material processor**. It's main job is to take your **raw audio files** (like `.mp3`) and turn them into a **structured dataset ready for machine-learning**. Specifically, `data_preparer.py`:

-  Organise these **features**  - and their corresponding drum labels **per the training data** (e.g. *snare*, *hi-hat*) - into a **format that a machine-learning model can understand**)
  >
 - Scales these features and **saves the scaling information and label mapping** for later use, ie in the **[`model_trainer.py`](#model_trainerpy) and [`drum_model.py`](#drum_modelpy)** modules.

<!--**Depends on:**--maybe add in later :)-->

### `model_trainer.py`

This is where we **train the machine-learning model** using the **transformed training data** from `data_preparer.py` . It uses this data **teach a machine-learning model** how to **recognise different drum sounds**. Specifically, `model_trainer.py`:

- Feeds the **features and labels** into a **chosen machine-learning model** (for now, this is standard **RandomForestClassifier**, but could equally be a **Support Vector** or **Neural Network** (aka. *Deep Learning*) models.
>
- During training, the model learns patterns that distinguish one **drum sound from another**.
>
- Once **model training is complete**, it saves the trained mo**del, ready for use in **identifying new drum sounds.

### `drum_model.py`

This is the **trained model** applied to new audio recordings. It uses the outputs from `model_trainer.py` to make statistical inference-based judgment on new audio recordings or uploads. Specifically, `drum_model.py`:

 - Loads the **chosen machine learning model**, created by `model_trainer.py`
 - Takes new audio (imported using the `audio_processor` module) to **predict which part of the drums is being played**
> 

---
<!--END---> 