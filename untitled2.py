import midi2audio
from midi2audio import FluidSynth

def midi_to_mp3(midi_file, output_file):
    # Convert MIDI to audio (WAV)
    FluidSynth().midi_to_audio(midi_file, output_file)


# Usage example
midi_file_path = "C:/Users/eswar/generated_music.mid"
output_file_path = "C:/Users/eswar/generated_music1.wav"
midi_to_mp3(midi_file_path, output_file_path)

