# Filename: notation_generator/drum_notation_tester.py
# Purpose: Test script to generate drum sheet music from a list of drum events
#          using music21 and a direct call to MuseScore for PDF conversion.

import json
import subprocess # Import the subprocess module
import music21
from music21 import stream, note, chord, meter, clef, instrument

# NOTE: We have removed the 'import musescore' line and the entire 
# 'env = ...' section as they were incorrect and causing errors.

def generate_drum_score(events, tempo=120, output_filename="drum_test_score"):
    """
    Generates a drum score PDF and MusicXML file from a list of timed events.
    """
    # 1. Initialize Score and Drum Part
    score = stream.Score()
    drum_part = stream.Part()
    drum_part.insert(0, instrument.Percussion())
    drum_part.insert(0, clef.PercussionClef())
    drum_part.insert(0, meter.TimeSignature('4/4'))

    # 2. Group events that happen at the same time
    TOLERANCE = 0.02
    grouped_events = []
    if not events:
        print("No events to process.")
        return
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
        offset = onset_time_sec * (tempo / 60.0)
        duration_qn = 0.25
        if len(group) == 1:
            event = group[0]
            n = note.Note(event['midi_pitch'])
            n.duration.quarterLength = duration_qn
            drum_part.insert(offset, n)
        else:
            midi_pitches = [event['midi_pitch'] for event in group]
            c = chord.Chord(midi_pitches)
            c.duration.quarterLength = duration_qn
            drum_part.insert(offset, c)

    # 4. Assemble the score and generate output files
    score.append(drum_part)
    score.makeMeasures(inPlace=True)
    print("✅ Score created successfully. Generating output files...")

    # Step 4a: Generate the MusicXML file.
    xml_path = f"{output_filename}.musicxml"
    score.write('musicxml', fp=xml_path)
    print(f"📄 MusicXML file saved to: {xml_path}")

    # Step 4b: Directly call MuseScore to convert the MusicXML to PDF.
    musescore_path = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
    pdf_path = f"{output_filename}.pdf"

    try:
        print("🎵 Attempting to generate PDF using MuseScore...")
        command = [
            musescore_path,
            '-o',
            pdf_path,
            xml_path,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"🎉 PDF file successfully generated at: {pdf_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: MuseScore executable not found at '{musescore_path}'. Please check the path.")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: MuseScore failed to convert the file.")
        print(f"   Error Output: {e.stderr}")


if __name__ == '__main__':
    mock_drum_hits = [
        {'onset_time_seconds': 0.0, 'midi_pitch': 36}, {'onset_time_seconds': 0.0, 'midi_pitch': 49},
        {'onset_time_seconds': 0.5, 'midi_pitch': 42}, {'onset_time_seconds': 1.0, 'midi_pitch': 38},
        {'onset_time_seconds': 1.0, 'midi_pitch': 42}, {'onset_time_seconds': 1.5, 'midi_pitch': 42},
        {'onset_time_seconds': 2.0, 'midi_pitch': 36}, {'onset_time_seconds': 2.0, 'midi_pitch': 42},
        {'onset_time_seconds': 2.5, 'midi_pitch': 42}, {'onset_time_seconds': 3.0, 'midi_pitch': 38},
        {'onset_time_seconds': 3.0, 'midi_pitch': 42}, {'onset_time_seconds': 3.5, 'midi_pitch': 42},
    ]
    
    generate_drum_score(mock_drum_hits, tempo=120)