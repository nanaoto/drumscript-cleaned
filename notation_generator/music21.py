# Filename: notation_generator/drum_notation_tester.py
# Purpose: Test script to generate drum sheet music from a list of drum events
#          using music21 and LilyPond.

import json
from music21 import stream, note, chord, meter, clef, instrument

def generate_drum_score(events, tempo=120, output_filename="drum_test_score"):
    """
    Generates a drum score PDF and MusicXML file from a list of timed events.

    Args:
        events (list): A list of dictionaries, where each dict represents a drum hit.
                       Expected keys: 'onset_time_seconds', 'midi_pitch'.
        tempo (int): The tempo of the piece in beats per minute (BPM).
        output_filename (str): The base name for the output files.
    """
    # 1. Initialize Score and Drum Part
    score = stream.Score()
    drum_part = stream.Part()

    # Add metadata and setup for a drum staff
    drum_part.insert(0, instrument.Percussion())
    drum_part.insert(0, clef.PercussionClef())
    drum_part.insert(0, meter.TimeSignature('4/4'))
    
    # 2. Group events that happen at the same time (chords)
    # A small tolerance to group notes that are extremely close together
    TOLERANCE = 0.02  
    
    grouped_events = []
    if not events:
        print("No events to process.")
        return

    # Sort events by onset time to ensure correct processing order
    events.sort(key=lambda x: x['onset_time_seconds'])
    
    current_group = [events[0]]
    for i in range(1, len(events)):
        if events[i]['onset_time_seconds'] - current_group[-1]['onset_time_seconds'] <= TOLERANCE:
            current_group.append(events[i])
        else:
            grouped_events.append(current_group)
            current_group = [events[i]]
    grouped_events.append(current_group)

    # 3. Process grouped events and add them to the drum part
    for group in grouped_events:
        onset_time_sec = group[0]['onset_time_seconds']
        
        # Convert seconds to a music21 quarter note offset
        # Offset is based on the tempo (beats per second)
        offset = onset_time_sec * (tempo / 60.0)

        # A simple fixed duration for each note/chord (e.g., 16th note)
        duration_qn = 0.25 

        if len(group) == 1:
            # It's a single note
            event = group[0]
            n = note.Note(event['midi_pitch'])
            n.duration.quarterLength = duration_qn
            # music21 automatically handles drum note heads (e.g., 'x' for cymbals)
            # based on the MIDI pitch and the Percussion instrument context.
            drum_part.insert(offset, n)
        else:
            # It's a chord (multiple drum hits at once)
            midi_pitches = [event['midi_pitch'] for event in group]
            c = chord.Chord(midi_pitches)
            c.duration.quarterLength = duration_qn
            drum_part.insert(offset, c)

    # 4. Assemble the score and generate output files
    score.append(drum_part)
    
    # Make the score more readable by creating measures
    score.makeMeasures(inPlace=True)
    
    print(f"✅ Score created successfully. Generating output files...")
    
    # Generate MusicXML (a primary goal of your project)
    xml_path = f"{output_filename}.musicxml"
    score.write('musicxml', fp=xml_path)
    print(f"📄 MusicXML file saved to: {xml_path}")
    
    # Generate PDF using the configured backend (ideally LilyPond)
    pdf_path = f"{output_filename}.pdf"
    score.write('musicxml.pdf', fp=pdf_path)
    print(f"🎵 PDF file saved to: {pdf_path}")


if __name__ == '__main__':
    # --- MOCK DATA ---
    # This simulates the kind of data your `prediction_output.json` would provide.
    # It represents a simple 4/4 rock beat.
    # MIDI Pitches for a standard drum map:
    # 36: Kick Drum
    # 38: Snare Drum
    # 42: Closed Hi-Hat
    # 49: Crash Cymbal 1
    
    mock_drum_hits = [
        # Measure 1
        {'onset_time_seconds': 0.0, 'midi_pitch': 36},  # Kick on beat 1
        {'onset_time_seconds': 0.0, 'midi_pitch': 49},  # Crash on beat 1
        {'onset_time_seconds': 0.5, 'midi_pitch': 42},  # Hi-hat on the '+' of 1
        {'onset_time_seconds': 1.0, 'midi_pitch': 38},  # Snare on beat 2
        {'onset_time_seconds': 1.0, 'midi_pitch': 42},  # Hi-hat on beat 2
        {'onset_time_seconds': 1.5, 'midi_pitch': 42},  # Hi-hat on the '+' of 2
        {'onset_time_seconds': 2.0, 'midi_pitch': 36},  # Kick on beat 3
        {'onset_time_seconds': 2.0, 'midi_pitch': 42},  # Hi-hat on beat 3
        {'onset_time_seconds': 2.5, 'midi_pitch': 42},  # Hi-hat on the '+' of 3
        {'onset_time_seconds': 3.0, 'midi_pitch': 38},  # Snare on beat 4
        {'onset_time_seconds': 3.0, 'midi_pitch': 42},  # Hi-hat on beat 4
        {'onset_time_seconds': 3.5, 'midi_pitch': 42},  # Hi-hat on the '+' of 4
    ]
    
    generate_drum_score(mock_drum_hits, tempo=120)