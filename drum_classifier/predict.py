import re
import os
import sys
import json


# Get the directory of the current script (predict.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach the 'DrumScript' root directory
# From drum_classifier/predict.py -> drum_classifier/ -> DrumScript/
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

# Add the 'DrumScript' root directory to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Assume the current working directory is DrumScript/
# Add the 'DrumScript' root to sys.path to allow imports from notation_generator
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary functions from your notation_generator module
from notation_generator.score_builder import build_and_export_drum_score
from notation_generator.helpers import format_drum_event

# The raw text output from the predict.py script
predict_output_raw = """
Time: 0.00s - Drums: hi-hat
Time: 0.10s - Drums: hi-hat
Time: 0.30s - Drums: snare
Time: 0.40s - Drums: hi-hat
Time: 0.50s - Drums: hi-hat
Time: 0.60s - Drums: hi-hat
Time: 0.70s - Drums: hi-hat
Time: 0.80s - Drums: hi-hat
Time: 0.90s - Drums: snare
Time: 1.10s - Drums: snare
Time: 1.20s - Drums: snare
Time: 1.40s - Drums: hi-hat
Time: 1.90s - Drums: hi-hat
Time: 2.00s - Drums: hi-hat
Time: 2.10s - Drums: snare
Time: 4.50s - Drums: hi-hat
Time: 5.20s - Drums: kick
Time: 10.90s - Drums: hi-hat, snare
# ... (rest of your predict.py output)
"""

# Process the output line by line
parsed_drum_events = []
# Regex to capture time and drum types, including multiple drums separated by ', '
# It handles cases like "hi-hat" or "hi-hat, snare"
line_pattern = re.compile(r"Time: (\d+\.\d+)s - Drums: (.+)")

for line in predict_output_raw.strip().split('\n'):
    match = line_pattern.match(line)
    if match:
        time_str, drums_str = match.groups()
        onset_time = float(time_str)
        # Split multiple drum types if present
        drum_types = [dt.strip() for dt in drums_str.split(',')]

        for drum_type in drum_types:
            # Use the helper function to get full notation info
            formatted_event = format_drum_event(drum_type, onset_time)
            parsed_drum_events.append(formatted_event)

# Now, `parsed_drum_events` is in the correct format for `build_and_export_drum_score`

print(f"Parsed {len(parsed_drum_events)} drum events.")
# You can print the first few to verify:
for i, event in enumerate(parsed_drum_events[:]):
     print(f"Event {i}: {event}")
print("-------------------------------------------------------------")

# --- ADD THE NEW CODE HERE ---
# Define the output file path within the drum_classifier directory
output_filename = "prediction_output.json"
output_filepath = os.path.join(current_script_dir, output_filename)

try:
    with open(output_filepath, 'w') as f:
        json.dump(parsed_drum_events, f, indent=4) # Use indent=4 for pretty printing
    print(f"\nSuccessfully exported parsed drum events to: {output_filepath}")
except IOError as e:
    print(f"\nError writing prediction output to file: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred during file export: {e}")
