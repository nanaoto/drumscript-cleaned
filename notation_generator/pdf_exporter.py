# DrumScript/notation_generator/pdf_exporter.py

from typing import Dict, Any
import os
import music21 # Import the music21 library

# Optional: You might want to import config for default settings
# from . import config 

def generate_pdf(score_data: Dict[str, Any], output_filepath: str):
    """
    Generates a PDF document of the drum sheet music from the structured score data
    using music21 and an external music engraving software (like MuseScore or LilyPond).

    Args:
        score_data (Dict[str, Any]): A dictionary containing the structured score information.
        output_filepath (str): The full path where the PDF file should be saved.
    """
    if not score_data or not score_data.get('parts', {}).get('drums'):
        print("No score data provided or no drum events to generate PDF.")
        return

    print(f"Generating PDF for '{score_data.get('title', 'Untitled')}' to {output_filepath} using music21...")

    try:
        # --- Music21 Environment Setup (Important for LilyPond) ---
        # music21 tries to auto-locate LilyPond. If it can't find it,
        # you might need to manually specify the path to your LilyPond executable.
        # Uncomment and adjust the line below if you get errors like "Couldn't find LilyPond"
        # after running the script.
        #
        # Example for macOS (adjust path based on where you installed LilyPond):
        # music21.environment.UserSettings()['lilypondPath'] = '/Applications/LilyPond.app/Contents/Resources/bin/lilypond'
        #
        # You can set this once in a Python console, or include it in your script.
        # Setting it once is generally preferred if you don't want to hardcode in every script.
        # For temporary testing, putting it here is fine.
        
        # 1. Create a new music21 Score object
        score = music21.stream.Score()
        
        # 2. Add metadata
        score.metadata = music21.metadata.Metadata()
        score.metadata.title = score_data.get('title', 'Drum Transcription')
        score.metadata.composer = score_data.get('composer', 'AI Generated')

        # 3. Create a percussion part
        part = music21.stream.Part()
        part.id = 'Drums'
        part.instrument = music21.instrument.Percussion()
        score.append(part)

        # 4. Add initial musical context (tempo, time signature)
        score.insert(0, music21.tempo.MetronomeMark(number=score_data['tempo']))
        numerator_str, denominator_str = score_data['time_signature'].split('/')
        score.insert(0, music21.meter.TimeSignature(f"{numerator_str}/{denominator_str}"))

        # 5. Populate measures with notes
        current_measure_number = 1
        current_measure_start_beat = 0.0
        beats_per_measure = float(numerator_str) # Assuming quarter note is the beat unit
        
        # Start the first measure
        current_measure = music21.stream.Measure(number=current_measure_number)
        part.append(current_measure)

        # Sort events by time to ensure chronological order
        sorted_events = sorted(score_data['parts']['drums'], key=lambda x: x['time_beats'])

        for event_idx, note_data in enumerate(sorted_events):
            beat_time = note_data['time_beats']
            midi_pitch = note_data['midi_pitch']
            note_head_type = note_data['note_head_type']
            duration_beats = note_data['duration_beats'] # This is the quantized duration from score_builder

            # Calculate offset within the current measure
            offset_in_current_measure = beat_time - current_measure_start_beat

            # Check if the note belongs in the current measure or if a new measure is needed
            # This is a simplified approach; music21's `makeMeasures` can automate much of this.
            if offset_in_current_measure >= beats_per_measure:
                # Fill the rest of the current measure with rests if needed
                current_measure.fillEmptyMeasures(inPlace=True)
                
                # Start a new measure
                current_measure_number += 1
                current_measure_start_beat = (current_measure_number - 1) * beats_per_measure
                current_measure = music21.stream.Measure(number=current_measure_number)
                part.append(current_measure)
                offset_in_current_measure = beat_time - current_measure_start_beat
            
            # Create a music21 note object
            n = music21.note.Note()
            n.pitch = music21.pitch.Pitch(midi=midi_pitch)
            n.quarterLength = duration_beats # Use the determined beat duration

            # Set notehead style for percussion (e.g., 'x' for cymbals/hi-hat)
            if note_head_type == 'x':
                n.notehead = music21.note.Notehead('x')
            
            # Add to the current measure at the correct offset
            current_measure.insert(offset_in_current_measure, n)

        # Fill any remaining empty space in the last measure with rests
        current_measure.fillEmptyMeasures(inPlace=True)

        # 6. Export to PDF
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # The 'write' method will try to find LilyPond and convert MusicXML to PDF.
        score.write('pdf', fp=output_filepath)
        
        print(f"Successfully generated PDF to {output_filepath}")

    except music21.converter.ConverterException as e:
        print(f"Error during music21 conversion/PDF generation. "
              f"This often means LilyPond (or MuseScore) is not installed or music21 can't find it. Error: {e}")
        print("Please ensure LilyPond is installed and its executable is in your system's PATH.")
        print("Alternatively, you might need to manually configure music21's environment path. For example, in a Python console (once):")
        print(f">>> import music21")
        print(f">>> music21.environment.UserSettings()['lilypondPath'] = '/Applications/LilyPond.app/Contents/Resources/bin/lilypond' # Adjust this path to your LilyPond executable")
        print(f">>> music21.environment.UserSettings().write() # Save the setting")
    except Exception as e:
        print(f"An unexpected error occurred during PDF generation: {e}")
        import traceback
        traceback.print_exc()