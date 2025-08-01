import os
import sys
import json
from notation_generator.score_builder import build_and_export_drum_score_to_pdf

def main():
    """
    Loads drum event data from a JSON file and generates a PDF drum score.
    """
    # Get the project root directory
    project_root = os.path.abspath(os.path.dirname(__file__))

    # Define the input and output file paths
    # The JSON file is in the drum_classifier directory
    input_json_filepath = os.path.join(project_root, "drum_classifier", "prediction_output.json")
    output_pdf_directory = os.path.join(project_root, "out")
    output_pdf_filepath = os.path.join(output_pdf_directory, "drum_score.pdf")

    # --- Step 1: Load the parsed drum events from the JSON file ---
    try:
        with open(input_json_filepath, 'r') as f:
            parsed_drum_events = json.load(f)
        print(f"Successfully loaded {len(parsed_drum_events)} drum events from {input_json_filepath}")
    except FileNotFoundError:
        print(f"Error: The file {input_json_filepath} was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file {input_json_filepath} is not a valid JSON file.")
        sys.exit(1)

    # --- Step 2: Define score parameters and generate the PDF ---
    # Ensure the output directory exists
    os.makedirs(output_pdf_directory, exist_ok=True)

    # Define tempo and quantization subdivision
    # You may need to adjust these values based on your music
    tempo_bpm = 120  # Beats per minute
    quantization_subdivision = 16 # e.g., 16 for sixteenth notes

    print(f"\nAttempting to generate drum score to: {output_pdf_filepath}")
    
    try:
        build_and_export_drum_score_to_pdf(
            drum_events=parsed_drum_events,
            output_filepath=output_pdf_filepath,
            tempo_bpm=tempo_bpm,
            quantization_subdivision=quantization_subdivision
        )
        print(f"✅ Successfully generated drum score PDF to: {output_pdf_filepath}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: A required program or file was not found. Please check:")
        print(f" - {e}")
        print(" - **Is LilyPond installed and its path correctly configured?**")
        print("   You can test this by typing `lilypond --version` in your terminal.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during PDF generation: {e}")
        print("   Please ensure Music21 is correctly installed (`pip install music21`).")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()