# DrumScript/notation_generator/pdf_exporter.py

from typing import Dict, Any
import os
import music21
import subprocess

def generate_pdf(music21_score: music21.stream.Score, output_filepath: str):
    """
    Generates a PDF document of the drum sheet music from a music21 Score object
    by first creating MusicXML and then converting it to PDF using MuseScore.

    Args:
        music21_score (music21.stream.Score): The music21 Score object containing the drum notation.
        output_filepath (str): The full path where the PDF file should be saved.
    """
    if not music21_score:
        print("No music21 score provided to generate PDF.")
        return

    print(f"Generating PDF for '{music21_score.metadata.title}' to {output_filepath} using music21 and MuseScore...")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    # Temporarily save as MusicXML
    musicxml_filepath = output_filepath.replace('.pdf', '.musicxml')
    try:
        music21_score.write('musicxml', fp=musicxml_filepath)
        print(f"MusicXML saved to: {musicxml_filepath}")

        # MuseScore command (adjust path if MuseScore is not in system PATH)
        # You might need to configure this path in a central config.py or an environment variable.
        # For macOS, it might be '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
        # For Windows, it might be 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'
        musescore_path = "musescore" # Assumes musescore is in your PATH

        # Attempt to find MuseScore if not directly in PATH (example for macOS)
        if os.path.exists("/Applications/MuseScore 4.app/Contents/MacOS/mscore"):
            musescore_path = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
        elif os.path.exists("/Applications/MuseScore 3.app/Contents/MacOS/mscore"): # for MuseScore 3
            musescore_path = "/Applications/MuseScore 3.app/Contents/MacOS/mscore"
        # Add Windows/Linux paths as needed, or instruct user to add to PATH

        print(f"Using MuseScore executable: {musescore_path}")

        # Run MuseScore to convert MusicXML to PDF
        command = [
            musescore_path,
            musicxml_filepath,
            '-o', output_filepath,
            '-r', '200' # Resolution (optional)
        ]
        
        # Use subprocess.run for better error handling and capturing output
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            if result.stdout:
                print(f"MuseScore STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"MuseScore STDERR:\n{result.stderr}")

            print(f"Successfully converted MusicXML to PDF: {output_filepath}")

        except FileNotFoundError:
            print(f"Error: MuseScore command '{musescore_path}' not found. Please ensure MuseScore is installed and the path is correct or it's in your system PATH.")
            raise # Re-raise to indicate failure
        except subprocess.CalledProcessError as e:
            print(f"Error during MuseScore conversion (exit code {e.returncode}):")
            print(f"Command: {' '.join(e.cmd)}") # Print the full command executed
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
        raise
    except Exception as e:
        print(f"An unexpected error occurred during PDF generation: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up the temporary MusicXML file
        if os.path.exists(musicxml_filepath):
            os.remove(musicxml_filepath)
            print(f"Cleaned up temporary MusicXML file: {musicxml_filepath}")