import numpy as np
import pretty_midi
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from flask import Flask,render_template
import pickle

app=Flask(__name__)

#model=pickle.load(open('model.pkl','rb'))
encoder_model=pickle.load(open('encoder_model.pkl','rb'))
decoder_model=pickle.load(open('decoder_model.pkl','rb'))

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



        
def preprocess_dataset(dataset_folder):
    dataset = []
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".midi"):
            midi_path = os.path.join(dataset_folder, filename)
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
            # Extract notes and their durations from MIDI file
            notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    pitch = note.pitch
                    start_time = note.start
                    end_time = note.end
                    duration = end_time - start_time
                    notes.append((pitch, duration))
            
            # Add notes to the dataset
            dataset.append(notes)
    
    return dataset

dataset_folder = "2018"
dataset = preprocess_dataset(dataset_folder)





note_duration_1=dict()
for i in dataset[1:5]:
    for (a,b) in i:
        note_duration_1[a]=(b)+0.17
for i in range(130):
    if(note_duration_1.get(i)==None or note_duration_1.get(i)>0.55):
        note_duration_1[i]=0.55
        
        


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



def prepare_training_data(music_data, input_vocab_size, output_vocab_size):
    encoder_input_data = []
    decoder_input_data = []
    decoder_target_data = []

    for notes in music_data:
        encoder_seq = []
        decoder_seq = []
        target_seq = []

        for i in range(len(notes)-1):
            encoder_seq.append(notes[i][0])
            decoder_seq.append(notes[i][0])
            target_seq.append(notes[i+1][0])

        encoder_input_data.append(encoder_seq)
        decoder_input_data.append(decoder_seq)
        decoder_target_data.append(target_seq)

    encoder_input_data = keras.preprocessing.sequence.pad_sequences(encoder_input_data, padding='post')
    decoder_input_data = keras.preprocessing.sequence.pad_sequences(decoder_input_data, padding='post')
    decoder_target_data = keras.preprocessing.sequence.pad_sequences(decoder_target_data, padding='post')

    encoder_input_data = to_categorical(encoder_input_data, num_classes=input_vocab_size)
    decoder_input_data = to_categorical(decoder_input_data, num_classes=output_vocab_size)
    decoder_target_data = to_categorical(decoder_target_data, num_classes=output_vocab_size)

    return encoder_input_data, decoder_input_data, decoder_target_data
input_vocab_size = 109 # Assuming 109 possible notes
output_vocab_size = 109
encoder_input_data, decoder_input_data, decoder_target_data = prepare_training_data(dataset[1:5], input_vocab_size, output_vocab_size)




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