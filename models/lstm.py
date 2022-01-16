from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras import regularizers

def build_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150, return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(total_words, activation='softmax'))
    return model
