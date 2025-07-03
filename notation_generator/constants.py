# DrumScript/notation_generator/constants.py

# Drum part mapping (should align with your model's classification)
DRUM_NOTATION_MAP = { 
    'kick': {'note_head': 'x', 'staff_position': 'F2'},
    'snare': {'note_head': 'normal', 'staff_position': 'C3'},
    'hi-hat': {'note_head': 'x', 'staff_position': 'G3'},
    'crash': {'note_head': 'x', 'staff_position': 'C4'},
    'ride': {'note_head': 'x', 'staff_position': 'A3'},
    # Add more drum types as classified by your model
}

# --- Musical Durations ---
# Common note durations as fractions of a whole note (1.0)
WHOLE_NOTE = 1.0
HALF_NOTE = 0.5
QUARTER_NOTE = 0.25
EIGHTH_NOTE = 0.125
SIXTEENTH_NOTE = 0.0625
THIRTYSECOND_NOTE = 0.03125

# --- MIDI Drum Mappings (General MIDI Standard Percussion Map) ---
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
STAFF_POS_KICK = -2    # Below the lowest staff line
STAFF_POS_SNARE = 0    # On the middle staff line
STAFF_POS_HI_HAT = 2   # Above the highest staff line
STAFF_POS_CRASH = 3    # Above staff
STAFF_POS_RIDE = 2.5   # Between lines/spaces

# --- Other common musical constants ---
TEMPO_DEFAULT_BPM = 120
TIME_SIGNATURE_NUMERATOR = 4
TIME_SIGNATURE_DENOMINATOR = 4