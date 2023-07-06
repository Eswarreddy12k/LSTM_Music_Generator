import numpy as np
import pretty_midi

import tensorflow as tf

from flask import Flask,render_template


app=Flask(__name__)

#model=pickle.load(open('model.pkl','rb'))
encoder_model=tf.keras.models.load_model('encoder_model.h5')
decoder_model=tf.keras.models.load_model('decoder_model.h5')

def generate_music_sequence(encoder_model, decoder_model, input_seq, output_vocab_size, max_output_length):
    states_value = encoder_model.predict(input_seq)

    generated_sequence = []

    for _ in range(max_output_length):
        target_seq = np.zeros((1, 1, output_vocab_size))
        target_seq[0, 0, np.random.randint(output_vocab_size)] = 1.0

        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        generated_token = np.argmax(output_tokens[0, -1, :])

        generated_sequence.append(generated_token)

        target_seq = np.zeros((1, 1, output_vocab_size))
        target_seq[0, 0, generated_token] = 1.0

        states_value = [h, c]

    return generated_sequence



        

dataset_folder = "2018"






note_duration_1={49: 0.29630208333348496, 37: 0.26375000000000004, 54: 0.19864583333325755, 42: 0.21166666666674247, 61: 0.20515625, 69: 0.5306770833332576, 59: 0.22598958333348493, 71: 0.5280729166667425, 57: 0.55, 73: 0.19604166666674247, 76: 0.24421875, 66: 0.248125, 80: 0.23901041666674247, 64: 0.2858854166667425, 81: 0.23640625, 78: 0.3002083333332576, 47: 0.55, 74: 0.55, 62: 0.27156250000000004, 85: 0.2884895833332576, 86: 0.3015104166665151, 83: 0.55, 44: 0.3522916666667425, 56: 0.2806770833332576, 84: 0.55, 65: 0.2897916666665151, 53: 0.20385416666674247, 68: 0.2572395833332576, 45: 0.29890625000000004, 75: 0.55, 72: 0.55, 60: 0.2741666666667425, 77: 0.25203125000000004, 32: 0.3119270833332576, 58: 0.22598958333325755, 70: 0.248125, 52: 0.29500000000000004, 31: 0.21296875, 43: 0.55, 30: 0.2390104166665151, 67: 0.26765625000000004, 79: 0.55, 35: 0.24291666666674247, 50: 0.55, 41: 0.248125, 29: 0.2963020833332576, 33: 0.4369270833332576, 40: 0.4343229166667425, 28: 0.20385416666674247, 36: 0.55, 48: 0.55, 63: 0.23640625, 51: 0.24291666666674247, 88: 0.29109375000000004, 82: 0.3913541666667425, 95: 0.31583333333348496, 90: 0.28718750000000004, 55: 0.50984375, 93: 0.2963020833332576, 39: 0.55, 92: 0.2976041666665151, 89: 0.29109375000000004, 46: 0.3002083333332576, 34: 0.28718750000000004, 27: 0.4733854166667425, 38: 0.24552083333325755, 87: 0.23901041666674247, 25: 0.55, 22: 0.47078125000000004, 26: 0.44473958333337127, 24: 0.17130208333348493, 23: 0.23380208333337124, 91: 0.28718750000000004, 96: 0.2845833333332576, 94: 0.2246875, 98: 0.24552083333348493, 99: 0.22338541666674247, 101: 0.2116666666665151, 103: 0.2233854166665151, 97: 0.2246875, 100: 0.2246875, 0: 0.55, 1: 0.55, 2: 0.55, 3: 0.55, 4: 0.55, 5: 0.55, 6: 0.55, 7: 0.55, 8: 0.55, 9: 0.55, 10: 0.55, 11: 0.55, 12: 0.55, 13: 0.55, 14: 0.55, 15: 0.55, 16: 0.55, 17: 0.55, 18: 0.55, 19: 0.55, 20: 0.55, 21: 0.55, 102: 0.55, 104: 0.55, 105: 0.55, 106: 0.55, 107: 0.55, 108: 0.55, 109: 0.55, 110: 0.55, 111: 0.55, 112: 0.55, 113: 0.55, 114: 0.55, 115: 0.55, 116: 0.55, 117: 0.55, 118: 0.55, 119: 0.55, 120: 0.55, 121: 0.55, 122: 0.55, 123: 0.55, 124: 0.55, 125: 0.55, 126: 0.55, 127: 0.55, 128: 0.55, 129: 0.55}

        
        


# Convert generated notes to MIDI file
def notes_to_midi(generated_sequence, tempo=120):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Create an instrument (e.g., piano)
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))

    # Convert the generated sequence to note events
    current_time = 0
    velocity = 100  # Adjust the velocity (volume) as desired

    for note in generated_sequence:
        if(note<21):
            note=note+21
        if(note>100):
            note=note-10
        # Adjust the note number based on the desired range
        note_number = note
        note_start = current_time
        note_end = note_start + note_duration_1[note_number]

        # Create a Note object
        note_obj = pretty_midi.Note(
            velocity=velocity,
            pitch=note_number,
            start=note_start,
            end=note_end
        )

        instrument.notes.append(note_obj)

        current_time += note_duration_1[note_number]

    # Add the instrument to the MIDI object
    midi.instruments.append(instrument)

    return midi




input_vocab_size = 109 # Assuming 109 possible notes
output_vocab_size = 109
encoder_input_data = np.load('encoder_input_data.npy')




@app.route('/')
def home():
    return render_template('index.html')
@app.route('/generate')
def generate_music():
    # Generate novel music sequence
    max_output_length = 80
    encoder_input = encoder_input_data[np.random.choice(len(encoder_input_data))][np.newaxis, :]
    generated_sequence = generate_music_sequence(encoder_model, decoder_model, encoder_input, output_vocab_size, max_output_length)

    # Convert generated notes to MIDI
    for i in range(len((generated_sequence))-4):
        if(generated_sequence[i]==generated_sequence[i+1] and generated_sequence[i+1]==generated_sequence[i+2]):
            generated_sequence[i+1]=-1
    refined_generated_sequence=[]
    for i in generated_sequence:
        if(i!=-1):
            refined_generated_sequence.append(i)
    midi = notes_to_midi(refined_generated_sequence)
    print(refined_generated_sequence)
    print(len(refined_generated_sequence))
    # Save the MIDI file

    midi.write('static/generated_music.mid')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()