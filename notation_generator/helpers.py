# DrumScript/notation_generator/helpers.py

import math
from . import constants

def round_to_nearest_subdivision(time_in_beats: float, subdivision: int) -> float:
    """
    Rounds a time value (in beats) to the nearest specified rhythmic subdivision.

    Args:
        time_in_beats (float): The time in beats (e.g., 1.25 for a beat and a quarter).
        subdivision (int): The rhythmic subdivision to round to (e.g., 4 for quarter notes, 16 for sixteenth notes).
                           Represents how many subdivisions are in a whole note based on a quarter-note beat.

    Returns:
        float: The time in beats, rounded to the nearest subdivision.
    """
    if subdivision == 0:
        raise ValueError("Subdivision cannot be zero.")

    # Calculate the duration of one subdivision unit in beats.
    # Assuming a quarter note is 1 beat (common time signature context).
    # A whole note is 4 beats. So, a subdivision of 16 means 16 units in a whole note.
    # Thus, one unit = (4 beats / subdivision) e.g., 4/16 = 0.25 beats for a sixteenth note.
    unit_duration_in_beats = 4.0 / subdivision
        
    rounded_time = round(time_in_beats / unit_duration_in_beats) * unit_duration_in_beats
    return rounded_time

def get_note_duration_name(duration_beats: float, tempo_bpm: int) -> str:
    """
    Converts a duration in beats to a standard musical note duration name (e.g., "quarter", "eighth").
    This is a simplification; accurate duration naming (especially for dotted/tied notes) is complex.

    Args:
        duration_beats (float): The duration in beats.
        tempo_bpm (int): The tempo in BPM (used for context, but beat length is key).

    Returns:
        str: The name of the musical note duration.
    """
    # Assuming a quarter note is 1 beat for simplicity (common in 4/4 time)
    # Convert beat duration to a fraction of a whole note (which is 4 beats)
    duration_as_whole_note_fraction = duration_beats / 4.0

    # Find the closest standard duration
    durations = {
        constants.WHOLE_NOTE: "whole",
        constants.HALF_NOTE: "half",
        constants.QUARTER_NOTE: "quarter",
        constants.EIGHTH_NOTE: "eighth",
        constants.SIXTEENTH_NOTE: "sixteenth",
        constants.THIRTYSECOND_NOTE: "thirty-second",
    }

    # Find the duration closest to the standard values
    closest_duration = min(durations.keys(), key=lambda d: abs(d - duration_as_whole_note_fraction))

    # Basic check for common dotted notes (e.g., 1.5 * QUARTER_NOTE = Dotted Quarter Note)
    # This is a very rough approximation and a full music notation library would handle this better.
    if abs(duration_as_whole_note_fraction - (constants.QUARTER_NOTE * 1.5)) < 0.001:
        return "dotted eighth" # Actually dotted eighth of a whole note if QUARTER_NOTE is 0.25 (1 beat)
                               # Should be (1/8)*1.5 of a whole note if you are using MusicXML style duration
    elif abs(duration_as_whole_note_fraction - (constants.HALF_NOTE * 1.5)) < 0.001:
        return "dotted quarter" # Actually dotted quarter of a whole note
    # ... add more dotted note checks if needed

    return durations[closest_duration]


# --- New Helper Functions ---

def seconds_to_beats(time_in_seconds: float, tempo_bpm: int) -> float:
    """
    Converts a time in seconds to beats given a tempo.

    Args:
        time_in_seconds (float): The time in seconds.
        tempo_bpm (int): The tempo in beats per minute (BPM).

    Returns:
        float: The equivalent time in beats.
    """
    if tempo_bpm <= 0:
        raise ValueError("Tempo (BPM) must be a positive value.")
    beats_per_second = tempo_bpm / 60.0
    return time_in_seconds * beats_per_second

def beats_to_seconds(beats: float, tempo_bpm: int) -> float:
    """
    Converts a time in beats to seconds given a tempo.

    Args:
        beats (float): The time in beats.
        tempo_bpm (int): The tempo in beats per minute (BPM).

    Returns:
        float: The equivalent time in seconds.
    """
    if tempo_bpm <= 0:
        raise ValueError("Tempo (BPM) must be a positive value.")
    seconds_per_beat = 60.0 / tempo_bpm
    return beats * seconds_per_beat

def format_event_for_notation_library(event: dict) -> dict:
    """
    Formats a classified drum event dictionary into a structure suitable for a notation library.
    This is a generic placeholder; actual implementation depends heavily on the chosen library
    (e.g., music21, Musescore, LilyPond, VexFlow).

    Args:
        event (dict): A dictionary representing a classified drum event,
                      e.g., {'time': float, 'drum_type': str}.

    Returns:
        dict: A formatted dictionary with notation-specific properties,
              e.g., {'midi_pitch': int, 'note_head': str, 'staff_pos': str}.
    """
    drum_type = event.get('drum_type')
    onset_time_seconds = event.get('time')

    notation_map_entry = constants.DRUM_NOTATION_MAP.get(drum_type)

    midi_pitch = None
    note_head_type = constants.NOTEHEAD_NORMAL # Default
    staff_position_string = None # Will store string like 'F2', 'C3'

    if notation_map_entry:
        staff_position_string = notation_map_entry.get('staff_position')
        note_head_type = notation_map_entry.get('note_head', constants.NOTEHEAD_NORMAL)
    else:
        print(f"Warning: No specific notation map entry defined for drum type '{drum_type}'. Using defaults.")
        # Fallback if drum_type is not in DRUM_NOTATION_MAP
        staff_position_string = 'C4' # A generic central percussion staff position
        # note_head_type remains default

    # MIDI pitch is still derived from constants.py MIDI definitions to align with General MIDI
    if drum_type == 'kick':
        midi_pitch = constants.MIDI_KICK
    elif drum_type == 'snare':
        midi_pitch = constants.MIDI_SNARE_ACOUSTIC
    elif drum_type == 'hi-hat':
        midi_pitch = constants.MIDI_HI_HAT_CLOSED
    elif drum_type == 'crash':
        midi_pitch = constants.MIDI_CRASH_CYMBAL_1
    elif drum_type == 'ride':
        midi_pitch = constants.MIDI_RIDE_CYMBAL_1
    else:
        midi_pitch = constants.MIDI_KICK # Default fallback for MIDI pitch

    formatted_event = {
        'drum_type': drum_type,
        'onset_time_seconds': onset_time_seconds,
        'midi_pitch': midi_pitch, # Keep MIDI pitch based on constants.py
        'note_head_type': note_head_type,
        'staff_position': staff_position_string, # Crucial: pass the string position from DRUM_NOTATION_MAP
    }
    return formatted_event


# Example of a simple mathematical helper (e.g., for musical interval calculations)
# This might be overkill depending on your needs, but shows the concept.
def calculate_cents_difference(freq1: float, freq2: float) -> float:
    """
    Calculates the difference between two frequencies in cents.
    Useful for tuning or pitch analysis.

    Args:
        freq1 (float): First frequency in Hz.
        freq2 (float): Second frequency in Hz.

    Returns:
        float: Difference in cents. Positive if freq2 is higher, negative if lower.
    """
    if freq1 <= 0 or freq2 <= 0:
        raise ValueError("Frequencies must be positive.")
    return 1200 * math.log2(freq2 / freq1)