import os
import pretty_midi
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Dense
import pickle
# Step 1: Preprocess the dataset

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

dataset_folder = "C:/Users/eswar/Downloads/archive (3)/maestro-v3.0.0/2018"
dataset = preprocess_dataset(dataset_folder)

def define_model(input_vocab_size, output_vocab_size, hidden_units):
    # Encoder model
    encoder_inputs = Input(shape=(None, input_vocab_size))
    encoder_lstm = LSTM(hidden_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder model
    decoder_inputs = Input(shape=(None, output_vocab_size))
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')

    # Define the encoder model separately
    encoder_model = Model(encoder_inputs, encoder_states)

    # Define the decoder model separately
    decoder_state_input_h = Input(shape=(hidden_units,))
    decoder_state_input_c = Input(shape=(hidden_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

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




# Example usage
music_data = dataset[1:5]

# Prepare the training data
input_vocab_size = 109 # Assuming 109 possible notes
output_vocab_size = 109
encoder_input_data, decoder_input_data, decoder_target_data = prepare_training_data(music_data, input_vocab_size, output_vocab_size)

# Define the model
hidden_units = 128
model, encoder_model, decoder_model = define_model(input_vocab_size, output_vocab_size, hidden_units)

# Train the model
epochs = 10
batch_size = 32
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs)

model.save('model.h5')
encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')

pickle.dump(model,open('model.pkl','wb'))
pickle.dump(encoder_model,open('encoder_model.pkl','wb'))
pickle.dump(decoder_model,open('decoder_model.pkl','wb'))