# DrumScript/drum_classifier/generate_score.py

import os
import sys
import json
import music21

def generate_midi_and_xml_from_json(input_json_path: str, midi_output_path: str, xml_output_path: str):
    """
    Reads a JSON file of drum events and generates a MIDI and MusicXML file
    in separate, specified directories using the music21 library.

    Args:
        input_json_path (str): The path to the input JSON file containing drum event data.
        midi_output_path (str): The directory where the output MIDI file will be saved.
        xml_output_path (str): The directory where the output XML file will be saved.
    """
    if not os.path.exists(input_json_path):
        print(f"Error: Input JSON file not found at '{input_json_path}'.")
        return

    # Create the output directories if they don't exist
    os.makedirs(midi_output_path, exist_ok=True)
    os.makedirs(xml_output_path, exist_ok=True)

    print(f"Reading drum events from: {input_json_path}")
    try:
        with open(input_json_path, 'r') as f:
            drum_events = json.load(f)
        print(f"Successfully loaded {len(drum_events)} drum events.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return

    # Create a new music21 score and percussion part
    score = music21.stream.Score()
    drum_part = music21.stream.Part()
    drum_part.insert(0, music21.instrument.Percussion())

    # Sort events by onset time to ensure they are in chronological order
    sorted_events = sorted(drum_events, key=lambda x: x['onset_time_seconds'])

    last_onset_time = 0.0
    for event in sorted_events:
        onset_time = event['onset_time_seconds']
        
        # Add rests for gaps in time
        if onset_time > last_onset_time:
            rest_duration = onset_time - last_onset_time
            rest_note = music21.note.Rest(duration=music21.duration.Duration(rest_duration))
            drum_part.append(rest_note)

        # Create an unpitched note for the drum event
        drum_note = music21.note.Unpitched()
        drum_note.midi = event['midi_pitch']
        drum_note.notehead = event['note_head_type']
        
        # music21's Unpitched note uses the 'midi' attribute to determine its
        # vertical position on the staff and for MIDI playback.
        
        drum_part.append(drum_note)
        last_onset_time = onset_time

    score.insert(0, drum_part)

    # Generate output filenames
    output_filename = os.path.splitext(os.path.basename(input_json_path))[0]
    
    # Write the score to MIDI and MusicXML files in their respective directories
    print("\nGenerating MIDI and MusicXML files...")
    midi_filepath = os.path.join(midi_output_path, f"{output_filename}.mid")
    score.write('midi', fp=midi_filepath)
    
    xml_filepath = os.path.join(xml_output_path, f"{output_filename}.xml")
    score.write('musicxml', fp=xml_filepath)
    
    print(f"✅ Successfully generated MIDI file: {midi_filepath}")
    print(f"✅ Successfully generated MusicXML file: {xml_filepath}")


if __name__ == "__main__":
    # Get the project root by going up one level from the current script's directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

    # Define paths
    input_json = os.path.join(current_script_dir, "prediction_output.json")
    midi_output_dir = os.path.join(project_root, "outputs", "drum_classifier", "midi")
    xml_output_dir = os.path.join(project_root, "outputs", "drum_classifier", "xml")

    generate_midi_and_xml_from_json(input_json, midi_output_dir, xml_output_dir)