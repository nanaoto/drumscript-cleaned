import argparse
from DrumScript.audio_processor.audio_loader import load_audio
from DrumScript.audio_processor.onset_detector import detect_onsets
from DrumScript.audio_processor.feature_extractor import extract_features
from DrumScript.drum_classifier.drum_model import DrumClassifierModel
from DrumScript.drum_classifier.model_trainer import load_model # Assuming a pre-trained model
from DrumScript.notation_generator.score_builder import quantize_events, map_to_drum_parts, create_score_data
from DrumScript.notation_generator.pdf_exporter import generate_pdf
from DrumScript.utils.config import get_config

def run_drumscript(audio_filepath, output_pdf_path, tempo=None):
    config = get_config()

    print(f"Loading audio from {audio_filepath}...")
    audio_data, sr = load_audio(audio_filepath, sr=config['audio']['sample_rate'])

    print("Detecting drum onsets...")
    onset_times = detect_onsets(audio_data, sr)
    print(f"Detected {len(onset_times)} onsets.")

    print("Loading drum classification model...")
    # Load your pre-trained model here
    # For a real project, you'd train this separately and save it.
    drum_classifier = load_model(config['model']['path'])

    classified_events = []
    print("Classifying drum events...")
    for i, onset_time in enumerate(onset_times):
        # Extract small segment around the onset
        start_sample = int(onset_time * sr)
        end_sample = min(start_sample + int(config['audio']['segment_length_samples']), len(audio_data))
        segment = audio_data[start_sample:end_sample]

        if len(segment) > 0:
            features = extract_features(segment, sr)
            # Reshape features for prediction if needed (e.g., for single prediction)
            prediction = drum_classifier.predict([features])[0] # Assuming single prediction
            # Map prediction index to drum type string
            drum_type = config['model']['class_labels'][prediction]
            classified_events.append({'time': onset_time, 'drum_type': drum_type})
            print(f"  Onset {i+1}: {onset_time:.2f}s -> {drum_type}")

    print("Quantizing and building score data...")
    # If tempo is not provided, you might need a tempo detection step here
    # For simplicity, let's assume a default or user-provided tempo for now
    quantized_events = quantize_events(classified_events, tempo=tempo if tempo else config['notation']['default_tempo'], subdivision=config['notation']['subdivision'])
    score_data = create_score_data(quantized_events)

    print(f"Generating PDF sheet music to {output_pdf_path}...")
    generate_pdf(score_data, output_pdf_path)

    print("DrumScript conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrumScript: Convert drum audio to sheet music.")
    parser.add_argument("audio_file", type=str, help="Path to the input drum audio file (e.g., .wav, .mp3).")
    parser.add_argument("output_pdf", type=str, help="Path for the output PDF sheet music file.")
    parser.add_argument("--tempo", type=int, default=None, help="Optional: Specify tempo in BPM. If not provided, a default or detected tempo will be used.")
    args = parser.parse_args()

    # Example of a missing piece: training a model initially.
    # For first use, you'd need to run a separate script to train the model
    # and save it before you can load it here.

    run_drumscript(args.audio_file, args.output_pdf, args.tempo)