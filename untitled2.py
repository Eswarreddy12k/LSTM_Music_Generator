import numpy as np
import pretty_midi
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from flask import Flask,render_template
import pickle





        







note_duration_1={49: 0.29630208333348496, 37: 0.26375000000000004, 54: 0.19864583333325755, 42: 0.21166666666674247, 61: 0.20515625, 69: 0.5306770833332576, 59: 0.22598958333348493, 71: 0.5280729166667425, 57: 0.55, 73: 0.19604166666674247, 76: 0.24421875, 66: 0.248125, 80: 0.23901041666674247, 64: 0.2858854166667425, 81: 0.23640625, 78: 0.3002083333332576, 47: 0.55, 74: 0.55, 62: 0.27156250000000004, 85: 0.2884895833332576, 86: 0.3015104166665151, 83: 0.55, 44: 0.3522916666667425, 56: 0.2806770833332576, 84: 0.55, 65: 0.2897916666665151, 53: 0.20385416666674247, 68: 0.2572395833332576, 45: 0.29890625000000004, 75: 0.55, 72: 0.55, 60: 0.2741666666667425, 77: 0.25203125000000004, 32: 0.3119270833332576, 58: 0.22598958333325755, 70: 0.248125, 52: 0.29500000000000004, 31: 0.21296875, 43: 0.55, 30: 0.2390104166665151, 67: 0.26765625000000004, 79: 0.55, 35: 0.24291666666674247, 50: 0.55, 41: 0.248125, 29: 0.2963020833332576, 33: 0.4369270833332576, 40: 0.4343229166667425, 28: 0.20385416666674247, 36: 0.55, 48: 0.55, 63: 0.23640625, 51: 0.24291666666674247, 88: 0.29109375000000004, 82: 0.3913541666667425, 95: 0.31583333333348496, 90: 0.28718750000000004, 55: 0.50984375, 93: 0.2963020833332576, 39: 0.55, 92: 0.2976041666665151, 89: 0.29109375000000004, 46: 0.3002083333332576, 34: 0.28718750000000004, 27: 0.4733854166667425, 38: 0.24552083333325755, 87: 0.23901041666674247, 25: 0.55, 22: 0.47078125000000004, 26: 0.44473958333337127, 24: 0.17130208333348493, 23: 0.23380208333337124, 91: 0.28718750000000004, 96: 0.2845833333332576, 94: 0.2246875, 98: 0.24552083333348493, 99: 0.22338541666674247, 101: 0.2116666666665151, 103: 0.2233854166665151, 97: 0.2246875, 100: 0.2246875, 0: 0.55, 1: 0.55, 2: 0.55, 3: 0.55, 4: 0.55, 5: 0.55, 6: 0.55, 7: 0.55, 8: 0.55, 9: 0.55, 10: 0.55, 11: 0.55, 12: 0.55, 13: 0.55, 14: 0.55, 15: 0.55, 16: 0.55, 17: 0.55, 18: 0.55, 19: 0.55, 20: 0.55, 21: 0.55, 102: 0.55, 104: 0.55, 105: 0.55, 106: 0.55, 107: 0.55, 108: 0.55, 109: 0.55, 110: 0.55, 111: 0.55, 112: 0.55, 113: 0.55, 114: 0.55, 115: 0.55, 116: 0.55, 117: 0.55, 118: 0.55, 119: 0.55, 120: 0.55, 121: 0.55, 122: 0.55, 123: 0.55, 124: 0.55, 125: 0.55, 126: 0.55, 127: 0.55, 128: 0.55, 129: 0.55}

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


print((encoder_input_data))

print(len(encoder_input_data[1][1]))



np.save('encoder_input_data.npy', encoder_input_data)
