# DrumScript/notation_generator/constants.py

# Drum part mapping (should align with your model's classification)

DRUM_NOTATION_MAP = {
    'kick': {
        'staff_position': 'F2',  # F2 for kick (bottom space on a 5-line staff)
        'midi_program': 36,      # General MIDI Kick Drum 1 (MIDI 36)
        'note_head': 'normal',   # Standard notehead
        'display_name': 'Bass Drum'
    },
    'snare': {
        'staff_position': 'C3',  # C3 for snare (middle space on a 5-line staff)
        'midi_program': 38,      # General MIDI Acoustic Snare (MIDI 38)
        'note_head': 'normal',   # Standard notehead
        'display_name': 'Snare Drum'
    },
    'hi-hat': { # This 'hi-hat' entry can represent a closed hi-hat
        'staff_position': 'F#3', # F#3 for hi-hat (top line on a 5-line staff) - often shown with 'x' head
        'midi_program': 42,      # General MIDI Closed Hi-Hat (MIDI 42)
        'note_head': 'x',        # 'x' notehead for cymbals/hi-hats
        'display_name': 'Hi-Hat (Closed)'
    },
    'open-hat': { # New entry for open hi-hat
        'staff_position': 'A#3', # A#3 for open hi-hat (above staff)
        'midi_program': 46,      # General MIDI Open Hi-Hat (MIDI 46)
        'note_head': 'x',        # 'x' notehead for cymbals/hi-hats
        'display_name': 'Hi-Hat (Open)'
    },
    'closed-hat': { # New entry for closed hi-hat (can be an alternative to 'hi-hat' above, or specifically for when the model distinguishes)
        'staff_position': 'F#3', # F#3, same as 'hi-hat' for visual consistency
        'midi_program': 42,      # General MIDI Closed Hi-Hat (MIDI 42)
        'note_head': 'x',
        'display_name': 'Hi-Hat (Closed)'
    },
    'crash': {
        'staff_position': 'C4',  # C4 for crash (above staff)
        'midi_program': 49,      # General MIDI Crash Cymbal 1 (MIDI 49)
        'note_head': 'x',
        'display_name': 'Crash Cymbal'
    },
    'ride': {
        'staff_position': 'A3',  # A3 for ride (space above top line)
        'midi_program': 51,      # General MIDI Ride Cymbal 1 (MIDI 51)
        'note_head': 'x',
        'display_name': 'Ride Cymbal'
    },
    'tom': { # This can be a generic tom, or you can add specific ones like 'high_tom', 'mid_tom', 'low_tom'
        'staff_position': 'E3',  # E3 for tom (space below hi-hat)
        'midi_program': 50,      # General MIDI High Tom (MIDI 50) - can be adjusted
        'note_head': 'normal',
        'display_name': 'Tom Drum'
    },
    # Add more drum types as classified by your model
    # Example for other possible drum types (already present in the original file, just for context):
    # 'floor_tom': {'note_head': 'normal', 'staff_position': 'A2'}, # A2 for floor tom
    # 'high_tom': {'note_head': 'normal', 'staff_position': 'D3'}, # D3 for high tom
    # 'mid_tom': {'note_head': 'normal', 'staff_position': 'B2'}, # B2 for mid tom
}

# --- Musical Durations ---
# Common note durations as fractions of a whole note (1.0 means whole note)
DURATION_WHOLE = 4.0
DURATION_HALF = 2.0
DURATION_QUARTER = 1.0
DURATION_EIGHTH = 0.5
DURATION_SIXTEENTH = 0.25
DURATION_THIRTY_SECOND = 0.125

# --- General MIDI Standard Percussion Map ---
# These are common MIDI note numbers for drum sounds.
# Ensure these align with your model's classification output and desired notation.
MIDI_KICK = 36         # C2 on piano
MIDI_SNARE_ACOUSTIC = 38 # D2 on piano
MIDI_SNARE_SIDE_STICK = 37 # C#2
MIDI_HI_HAT_CLOSED = 42 # F#2
MIDI_HI_HAT_PEDAL = 44  # G#2
MIDI_HI_HAT_OPEN = 46   # A#2
MIDI_CRASH_CYMBAL_1 = 49 # C#3
MIDI_RIDE_CYMBAL_1 = 51  # D#3
MIDI_TOM_HIGH = 50      # D3
MIDI_TOM_MID = 47       # A2
MIDI_TOM_LOW = 45       # G2

# --- Notation Symbols / Mapping Hints ---
# These are conceptual for score generation (e.g., using MusicXML or a drawing library)
# Actual rendering depends on the chosen notation library/method.
NOTEHEAD_NORMAL = 'normal'
NOTEHEAD_X = 'x' # For cymbals, hi-hats, often snare side stick

# --- Staff Positions (conceptual, depends on rendering library's coordinate system) ---
# These might relate to MIDI pitch or specific lines/spaces on a percussion staff.
# For a 5-line percussion staff, these are relative positions.
# Actual Y-coordinates would be calculated by pdf_exporter or score_builder.
STAFF_POS_KICK = -2    # Example relative position for kick drum
STAFF_POS_SNARE = 0    # Example relative position for snare drum
STAFF_POS_HI_HAT = 2   # Example relative position for hi-hat
# ... and so on for other drums