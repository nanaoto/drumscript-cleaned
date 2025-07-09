# DrumScript/notation_generator/pdf_exporter.py

from typing import Dict, Any
import os
import music21 
import subprocess # Import subprocess for calling external commands

# Optional: Maybe want to import config for default settings
# from . import config 

def generate_pdf(score_data: Dict[str, Any], output_filepath: str):
    """
    Generates a PDF document of the drum sheet music from the structured score data
    by first creating MusicXML and then converting it to PDF using MuseScore.

    Args:
        score_data (Dict[str, Any]): A dictionary containing the structured score information.
        output_filepath (str): The full path where the PDF file should be saved.
    """
    if not score_data or not score_data.get('parts', {}).get('drums'):
        print("No score data provided or no drum events to generate PDF.")
        return
    print(score_data) # inspect data applied for building score, useful for comparing outputs later
    print(f"Generating PDF for '{score_data.get('title', 'Untitled')}' to {output_filepath} using music21 and MuseScore...")

    try:
        # --- Diagnostic: music21 recognised user settings keys (useful for reference) ---
        # print("DEBUG: music21 recognised user settings keys:")
        # for key in music21.environment.UserSettings().keys():
          #   print(f"  - {key}")
        # --- End Diagnostic ---

        # --- music21 Environment Setup (Keeping for robustness, though less critical for subprocess call) ---
        # Can comment out the lilypondPath setting maybe, if it's not needed
        # music21.environment.UserSettings()['lilypondPath'] = '/opt/homebrew/bin/lilypond'
        # music21.environment.UserSettings().write() 
        # current_lilypond_path = music21.environment.UserSettings()['lilypondPath']
        # print(f"DEBUG: music21's lilypondPath setting: {current_lilypond_path}")

        # 1. Create a new Score object
        score = music21.stream.Score()

        # 2. Create a Part for drums
        part = music21.stream.Part()
        part.partName = 'Drums'
        part.clef = music21.clef.PercussionClef() # Set percussion clef

        # Add initial metadata (time signature, tempo)
        time_signature_str = score_data.get('time_signature', '4/4')
        ts = music21.meter.TimeSignature(time_signature_str)
        part.append(ts)

        tempo_bpm = score_data.get('tempo', 120)
        metronome = music21.tempo.MetronomeMark(number=tempo_bpm)
        part.append(metronome)

        sorted_events = sorted(score_data['parts']['drums'], key=lambda x: x['time_beats'])
    # 3. Add notes to the part
        for event_idx, note_data in enumerate(sorted_events):
            beat_time = note_data['time_beats']
            midi_pitch = note_data['midi_pitch']
            drum_type = note_data['drum_type'] # Get the drum type (e.g., 'kick', 'snare')
            note_head_type = note_data['note_head_type']
            duration_beats = note_data['duration_beats']

            n = music21.note.Note()
            n.pitch = music21.pitch.Pitch(midi=midi_pitch)
            n.quarterLength = duration_beats

        # --- UPDATED CODE TO SET STAFF LINE FOR ALL DRUM TYPES ---
        # These staffLine numbers are common conventions for a 5-line percussion staff:
        # Line 1: bottom line
        # Line 3: middle line
        # Line 5: top line
        # Beyond Line 5 are ledger lines above the staff.
        if drum_type == 'kick':
            n.pitch.staffLine = 1 # Place kick on the bottom line
        elif drum_type == 'snare':
            n.pitch.staffLine = 3 # Place snare on the middle line
        elif drum_type == 'hi-hat':
            n.pitch.staffLine = 5 # Place hi-hat on the top line
        elif drum_type == 'crash':
            n.pitch.staffLine = 6 # Place crash cymbal on a ledger line above staff
        elif drum_type == 'ride':
            n.pitch.staffLine = 7 # Place ride cymbal on a ledger line higher than crash
        else:
            # Fallback for any unmapped drum types to avoid them being off-stave
            print(f"Warning: No specific staffLine mapping for drum type '{drum_type}'. Defaulting to snare position.")
            n.pitch.staffLine = 3 # Default to snare position

        # This ensures x-noteheads are applied based on the note_head_type from constants.py
        if note_head_type == 'x':
            n.notehead = note_head_type

        part.insert(beat_time, n)


        # Let music21 automatically create and fill measures for the entire part
        # part.makeMeasures(inPlace=True, finalBarline=True)
        # Create a specific final barline object
        final_barline_obj = music21.bar.Barline('final') 
        part.makeMeasures(inPlace=True, finalBarline=final_barline_obj)

        # 4. Add the part to the score
        score.append(part)

        # 5. Export to MusicXML first (intermediate step)
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        xml_filepath = output_filepath.replace('.pdf', '.xml')
        score.write('musicxml', fp=xml_filepath)
        print(f"Successfully generated MusicXML to {xml_filepath}")

        # 6. Convert MusicXML to PDF using MuseScore via subprocess
        # This path was confirmed from previous outputs:
        musescore_path = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'

        if not os.path.exists(musescore_path):
            raise FileNotFoundError(f"MuseScore executable not found at: {musescore_path}. Please check the path.")

        print(f"Attempting to convert MusicXML to PDF using MuseScore from: {musescore_path}")

        try:
            # MuseScore command: -o for output, input file, -f for force overwrite
            command = [musescore_path, xml_filepath, '-o', output_filepath, '-f']
            print(f"Executing MuseScore command: {' '.join(command)}")

            # Run the command, capture output for debugging, and raise an error if it fails
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            if result.stdout:
                print(f"MuseScore STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"MuseScore STDERR:\n{result.stderr}")

            print(f"Successfully converted MusicXML to PDF: {output_filepath}")

        except FileNotFoundError:
            print(f"Error: MuseScore command not found. Please ensure MuseScore is installed and the path '{musescore_path}' is correct in pdf_exporter.py.")
            raise # Re-raise to indicate failure
        except subprocess.CalledProcessError as e:
            print(f"Error during MuseScore conversion (exit code {e.returncode}):")
            print(f"Command: {e.cmd}")
            print(f"STDOUT:\n{e.stdout}")
            print(f"STDERR:\n{e.stderr}")
            print("Please ensure MuseScore can convert MusicXML files from the command line and that the XML file is valid.")
            raise # Re-raise to indicate failure
        except Exception as e:
            print(f"An unexpected error occurred during MuseScore conversion: {e}")
            raise # Re-raise

    except music21.converter.ConverterException as e:
        print(f"Error during music21 score creation/MusicXML export: {e}")
        print("This indicates an issue with the music21 score object itself or its internal MusicXML converter.")
    except Exception as e:
        print(f"An unexpected error occurred during PDF generation: {e}")
        import traceback
        traceback.print_exc()