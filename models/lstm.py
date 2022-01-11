import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding

def lstm_data_prepare(dataX, dataY, seq_len, n_patterns, vocab_size):
    X = np.reshape(dataX, (n_patterns, seq_len, 1))
    y = np_utils.to_categorical(dataY)

    return X, y

class RNNLSTM:

    def __init__(self, X, y, vocab_size):
        self.X = X
        self.y = y
        self.vocab_size = vocab_size

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=200))
        model.add(LSTM(256, input_shape=(None, self.X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256))
        model.add(Dropout(0.2))
        model.add(Dense(self.y.shape[1], activation='softmax'))
        return model
