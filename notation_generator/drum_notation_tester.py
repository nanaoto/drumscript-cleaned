# Filename: notation_generator/drum_notation_tester.py
# Purpose: Test script to generate drum sheet music from a list of drum events
#          using music21 and a direct call to MuseScore for PDF conversion.
import json
import subprocess # Import the subprocess module
import music21
from music21 import stream, note, chord, meter, clef, instrument
import musescore


# --- ADD THIS SECTION ---
# Manually set the path to your MuseScore 4 executable.
# This overrides the default and ensures music21 finds the correct version.
env = music21.environment.Environment()
#env['musicxmlPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
env['musescore'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
# ----------------------

def generate_drum_score(events, tempo=120, output_filename="drum_test_score"):
    """
    Generates a drum score PDF and MusicXML file from a list of timed events.
    """
    # 1. Initialize Score and Drum Part (This section is unchanged)
    score = stream.Score()
    drum_part = stream.Part()
    drum_part.insert(0, instrument.Percussion())
    drum_part.insert(0, clef.PercussionClef())
    drum_part.insert(0, meter.TimeSignature('4/4'))

    # 2. Group events that happen at the same time (This section is unchanged)
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

    # 3. Process grouped events and add them to the drum part (This section is unchanged)
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

    # Step 4a: Generate the MusicXML file. This part works correctly.
    xml_path = f"{output_filename}.musicxml"
    score.write('musicxml', fp=xml_path)
    print(f"📄 MusicXML file saved to: {xml_path}")

    # --- REVISED SECTION: PDF GENERATION ---
    # Step 4b: Directly call MuseScore to convert the MusicXML to PDF.
    # This is a more reliable method than using music21's internal converter.
    musescore_path = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
    pdf_path = f"{output_filename}.pdf"

    try:
        print("🎵 Attempting to generate PDF using MuseScore...")
        command = [
            musescore_path,
            '-o',  # Specify the output file
            pdf_path,
            xml_path, # Specify the input file
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"🎉 PDF file successfully generated at: {pdf_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: MuseScore executable not found at '{musescore_path}'. Please check the path.")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: MuseScore failed to convert the file.")
        print(f"   Return Code: {e.returncode}")
        print(f"   Output: {e.stdout}")
        print(f"   Error Output: {e.stderr}")
    # --- END REVISED SECTION ---



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