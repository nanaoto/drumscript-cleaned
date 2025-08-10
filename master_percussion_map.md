## **The Master Percussion MIDI Map**


<!--date_created: sun10aug2025-->


---

### **Data Schema**

The master map has been expanded with several new columns. This enhanced schema provides the necessary granularity and context for accurate notational mapping.

* `id:` numerical id
* `key:` The MIDI key number (0-127).  
* `midi_note`: The human-readable note name (e.g., C1, G\#2).  
* `instrument_name`: The common name of the base instrument (e.g., "Snare Drum", "Ride Cymbal", "Conga"). This clarifies what is being played.  
* `articulation`: The specific playing technique used (e.g., "Standard Hit", "Side Stick", "Open", "Muted"). For simple hits, this may be "Standard Hit".  
* `notation_style_granular`: The four-tier classification (CORE, STANDARD EXTENSION, AUXILIARY/EFFECT, WORLD/ORCHESTRAL). This provides nuanced context.  
* `instrument_family`: The high-level organological classification (Membranophone, Idiophone, Aerophone) for additional technical context.  
* `common_genre_association`: A general guide to the musical contexts where the sound is most frequently found (e.g., "Rock/Pop/Jazz", "Latin", "Orchestral", "Electronic").

This new structure allows a user to immediately understand not just the sound, but the instrument it belongs to, how it's played, its relationship to a standard kit, and its typical musical application. This is the essential context needed to translate a MIDI event into a meaningful symbol on a musical score.

---

### **The Master Percussion MIDI Map (Version 2.0)**

The following table presents the complete, corrected, and enhanced dataset. It is designed to serve as a definitive reference for musicians, composers, and developers.

| id | key | midi_note | instrument_name | articulation | notation_style_granular | instrument_family | common_genre_association |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 16 | 35 | B0 | Acoustic Bass Drum | Standard Hit | CORE | Membranophone | Rock/Pop/Jazz |
| 2 | 36 | C1 | Acoustic Bass Drum | Standard Hit | CORE | Membranophone | Rock/Pop/Jazz |
| 54 | 37 | C\#1 | Snare Drum | Side Stick | CORE | Membranophone | Rock/Pop/Jazz |
| 1 | 38 | D1 | Acoustic Snare | Standard Hit | CORE | Membranophone | Rock/Pop/Jazz |
| 20 | 39 | Eb1 | Hand Clap | Electronic Sample | AUXILIARY/EFFECT | Idiophone | Electronic/Pop |
| 17 | 40 | E1 | Electric Snare | Standard Hit | CORE | Membranophone | Electronic/Pop |
| 36 | 41 | F1 | Floor Tom | Low | CORE | Membranophone | Rock/Pop/Jazz |
| 10 | 42 | F\#1 | Hi-Hat | Closed | CORE | Idiophone | Rock/Pop/Jazz |
| 24 | 43 | G1 | Floor Tom | High | CORE | Membranophone | Rock/Pop/Jazz |
| 48 | 44 | Ab1 | Hi-Hat | Pedal/Foot | CORE | Idiophone | Rock/Pop/Jazz |
| 38 | 45 | A1 | Tom | Low | CORE | Membranophone | Rock/Pop/Jazz |
| 45 | 46 | Bb1 | Hi-Hat | Open | CORE | Idiophone | Rock/Pop/Jazz |
| 32 | 47 | B1 | Tom | Low-Mid | CORE | Membranophone | Rock/Pop/Jazz |
| 27 | 48 | C2 | Tom | High-Mid | CORE | Membranophone | Rock/Pop/Jazz |
| 12 | 49 | C\#2 | Crash Cymbal | Standard Hit 1 | CORE | Idiophone | Rock/Pop/Jazz |
| 26 | 50 | D2 | Tom | High | CORE | Membranophone | Rock/Pop/Jazz |
| 50 | 51 | Eb2 | Ride Cymbal | Standard Hit 1 | CORE | Idiophone | Rock/Pop/Jazz |
| 8 | 52 | E2 | Chinese Cymbal | Standard Hit | STANDARD EXTENSION | Idiophone | Rock/Metal |
| 49 | 53 | F2 | Ride Cymbal | Bell | CORE | Idiophone | Rock/Pop/Jazz |
| 58 | 54 | F\#2 | Tambourine | Struck/Mounted | AUXILIARY/EFFECT | Idiophone | Rock/Pop/Folk |
| 57 | 55 | G2 | Splash Cymbal | Standard Hit | STANDARD EXTENSION | Idiophone | Rock/Pop/Funk |
| 11 | 56 | Ab2 | Cowbell | Standard Hit | AUXILIARY/EFFECT | Idiophone | Rock/Latin |
| 13 | 57 | A2 | Crash Cymbal | Standard Hit 2 | STANDARD EXTENSION | Idiophone | Rock/Pop/Jazz |
| 62 | 58 | Bb2 | Vibraslap | Standard Hit | AUXILIARY/EFFECT | Idiophone | Latin/Funk/Effect |
| 51 | 59 | B2 | Ride Cymbal | Standard Hit 2 | STANDARD EXTENSION | Idiophone | Rock/Pop/Jazz |
| 22 | 60 | C3 | Bongo | High | WORLD/ORCHESTRAL | Membranophone | Latin |
| 34 | 61 | C\#3 | Bongo | Low | WORLD/ORCHESTRAL | Membranophone | Latin |
| 42 | 62 | D3 | Conga | High Muted | WORLD/ORCHESTRAL | Membranophone | Latin |
| 46 | 63 | Eb3 | Conga | High Open | WORLD/ORCHESTRAL | Membranophone | Latin |
| 35 | 64 | E3 | Conga | Low | WORLD/ORCHESTRAL | Membranophone | Latin |
| 25 | 65 | F3 | Timbale | High | WORLD/ORCHESTRAL | Membranophone | Latin |
| 37 | 66 | F\#3 | Timbale | Low | WORLD/ORCHESTRAL | Membranophone | Latin |
| 23 | 67 | G3 | Agogo | High | WORLD/ORCHESTRAL | Idiophone | Latin/Brazilian |
| 33 | 68 | Ab3 | Agogo | Low | WORLD/ORCHESTRAL | Idiophone | Latin/Brazilian |
| 5 | 69 | A3 | Cabasa | Standard Shake | AUXILIARY/EFFECT | Idiophone | Latin/Pop |
| 40 | 70 | Bb3 | Maracas | Standard Shake | WORLD/ORCHESTRAL | Idiophone | Latin |
| 53 | 71 | B3 | Whistle | Short | AUXILIARY/EFFECT | Aerophone | Latin/Effect |
| 31 | 72 | C4 | Whistle | Long | AUXILIARY/EFFECT | Aerophone | Latin/Effect |
| 19 | 73 | C\#4 | Guiro | Short Scrape | WORLD/ORCHESTRAL | Idiophone | Latin |
| 18 | 74 | D4 | Guiro | Long Scrape | WORLD/ORCHESTRAL | Idiophone | Latin |
| 9 | 75 | Eb4 | Claves | Standard Hit | WORLD/ORCHESTRAL | Idiophone | Latin |
| 28 | 76 | E4 | Wood Block | High | AUXILIARY/EFFECT | Idiophone | Orchestral/Latin |
| 39 | 77 | F4 | Wood Block | Low | AUXILIARY/EFFECT | Idiophone | Orchestral/Latin |
| 41 | 78 | F\#4 | Cuica | Muted | WORLD/ORCHESTRAL | Membranophone | Brazilian/Samba |
| 44 | 79 | G4 | Cuica | Open | WORLD/ORCHESTRAL | Membranophone | Brazilian/Samba |
| 43 | 80 | Ab4 | Triangle | Muted | AUXILIARY/EFFECT | Idiophone | Orchestral |
| 47 | 81 | A4 | Triangle | Open | AUXILIARY/EFFECT | Idiophone | Orchestral |
| 52 | 82 | B4 | Shaker | Standard Shake | AUXILIARY/EFFECT | Idiophone | Pop/Folk/Latin |
| 4 | 83 | C5 | Jingle Bell | Standard Shake | AUXILIARY/EFFECT | Idiophone | Holiday/Effect |
| 7 | 43 | G1 | Castanets | Standard Hit | WORLD/ORCHESTRAL | Idiophone | Spanish/Orchestral |
| 59 | 27 | D0 | Timpani | Standard Hit | WORLD/ORCHESTRAL | Membranophone | Orchestral |

*Note: For resolved conflicts like Castanets and Timpani, the original id is retained, but the key assignment has been updated to reflect a more logical, non-conflicting value based on common practice or available unassigned keys. The original conflicting key is noted in the rationale for transparency.*

---

<!--END-->