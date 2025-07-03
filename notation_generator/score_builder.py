# DrumScript/notation_generator/score_builder.py

from typing import List, Dict, Any
#from . import config
#from . import constants
#from .helpers import seconds_to_beats, round_to_nearest_subdivision, format_event_for_notation_library
import utils.config as config
#import constants
from . import constants
import math
#from helpers import seconds_to_beats, round_to_nearest_subdivision, format_event_for_notation_library
from .helpers import seconds_to_beats, round_to_nearest_subdivision, format_event_for_notation_library

def quantize_events(classified_events: List[Dict[str, Any]], tempo: int, subdivision: int) -> List[Dict[str, Any]]:
    """
    Quantizes the timing of classified drum events to the nearest rhythmic subdivision.

    Args:
        classified_events (List[Dict[str, Any]]): A list of dictionaries, where each dict
                                                  represents a classified drum event with at least
                                                  {'time': float, 'drum_type': str}. 'time' is in seconds.
        tempo (int): The tempo in BPM for converting seconds to beats.
        subdivision (int): The rhythmic subdivision to quantize to (e.g., 16 for sixteenth notes).

    Returns:
        List[Dict[str, Any]]: A list of quantized drum events, with 'time_beats' added.
                              Each event will have 'time' (original seconds),
                              'time_beats_quantized' (quantized time in beats),
                              'drum_type', and other relevant data.
    """
    if not classified_events:
        print("No classified events to quantize.")
        return []

    quantized_events = []
    print(f"Quantizing {len(classified_events)} events to {subdivision}-note subdivision at {tempo} BPM...")

    for event in classified_events:
        onset_time_seconds = event['time']
        drum_type = event['drum_type']

        # Convert onset time from seconds to beats
        onset_time_beats = seconds_to_beats(onset_time_seconds, tempo)

        # Quantize the beat time
        quantized_beat_time = round_to_nearest_subdivision(onset_time_beats, subdivision)

        quantized_event = {
            'time_seconds_original': onset_time_seconds,
            'time_beats_quantized': quantized_beat_time,
            'drum_type': drum_type,
            # Add any other relevant event data here
        }
        quantized_events.append(quantized_event)
    
    print("Quantization complete.")
    return quantized_events

def map_to_drum_parts(quantized_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Maps quantized drum events to specific drum kit parts,
    assigning MIDI pitches and notation properties based on drum type.
    This function might be implicitly handled by `format_event_for_notation_library`
    or explicit here for clearer separation of concerns.

    Args:
        quantized_events (List[Dict[str, Any]]): List of quantized drum events.

    Returns:
        List[Dict[str, Any]]: Events with added notation-specific attributes like MIDI pitch,
                              note head type, and conceptual staff position.
    """
    drum_part_events = []
    print("Mapping drum events to notation parts...")
    for event in quantized_events:
        # Use the helper function to get notation-specific details
        # Note: format_event_for_notation_library expects 'time' in seconds,
        # but here we are using 'time_beats_quantized' for score positioning.
        # We'll pass the drum_type and let it look up MIDI/staff position.
        
        # Create a temporary dict for formatting, as helper expects 'time' and 'drum_type'
        temp_event_for_helper = {
            'time': event['time_seconds_original'], # Pass original time if helper uses it for pitch map
            'drum_type': event['drum_type']
        }
        formatted_notation_data = format_event_for_notation_library(temp_event_for_helper)

        # Combine quantized time with notation data
        drum_part_event = {
            'time_beats': event['time_beats_quantized'],
            'drum_type': event['drum_type'],
            'midi_pitch': formatted_notation_data.get('midi_pitch'),
            'note_head_type': formatted_notation_data.get('note_head_type'),
            'staff_position': formatted_notation_data.get('staff_position'),
            # You might also add duration here if you're inferring it (e.g., all hits are 16th notes)
            'duration_beats': constants.QUARTER_NOTE / (config.QUANTIZATION_SUBDIVISION / 4.0) # E.g., for 16th, this is 0.25 beats
        }
        drum_part_events.append(drum_part_event)
    print("Drum part mapping complete.")
    return drum_part_events

def create_score_data(quantized_and_mapped_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates quantized and mapped drum events into a structured dictionary
    representing the complete score, ready for PDF export.

    Args:
        quantized_and_mapped_events (List[Dict[str, Any]]): A list of events,
                                                          each with quantized beat times
                                                          and notation-specific attributes.

    Returns:
        Dict[str, Any]: A dictionary containing all data necessary to generate the score,
                        e.g., tempo, time signature, and a list of formatted notes.
                        Example structure:
                        {
                            'title': 'Drum Transcription',
                            'tempo': int,
                            'time_signature': '4/4',
                            'parts': {
                                'drums': [
                                    {'beat_time': float, 'midi_pitch': int, 'note_head': str, 'staff_pos': float, 'duration_beats': float},
                                    ...
                                ]
                            }
                        }
    """
    if not quantized_and_mapped_events:
        print("No events to build score data from.")
        return {}

    score_data = {
        'title': 'DrumScript Transcription',
        'composer': 'AI Generated',
        'tempo': config.DEFAULT_TEMPO_BPM, # Use default or derive from input
        'time_signature': f"{constants.TIME_SIGNATURE_NUMERATOR}/{constants.TIME_SIGNATURE_DENOMINATOR}",
        'parts': {
            'drums': []
        }
    }

    current_measure = 0
    current_beat_in_measure = 0.0
    beats_per_measure = constants.TIME_SIGNATURE_NUMERATOR # Assuming quarter note beat

    for event in quantized_and_mapped_events:
        beat_time = event['time_beats']
        
        # Determine measure and beat within measure (simplified)
        # This logic needs to be robust for various time signatures and note durations
        measure_number = math.floor(beat_time / beats_per_measure) + 1
        beat_in_measure = beat_time % beats_per_measure

        # Append the formatted note to the score data
        score_data['parts']['drums'].append({
            'measure': measure_number,
            'beat_in_measure': beat_in_measure,
            'midi_pitch': event['midi_pitch'],
            'drum_type': event['drum_type'],
            'note_head_type': event['note_head_type'],
            'staff_position': event['staff_position'],
            'duration_beats': event['duration_beats'],
            # You might add velocity, ties, dots etc. here
        })
    
    print("Score data creation complete.")
    return score_data