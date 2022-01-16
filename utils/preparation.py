from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku 
import numpy as np

class DataLoader:

    def __init__(self, file) :
        self.file = file

        self.corpus = None
        self.tokenizer = None
        self.input_sequences = None
        self.total_words = None

    def tokenize(self) :
        tokenizer = Tokenizer()
        data = open(self.file).read()
        self.corpus = data.lower().split("\n")
        tokenizer.fit_on_texts(self.corpus)
        self.total_words = len(tokenizer.word_index) + 1
        self.tokenizer = tokenizer
        return tokenizer

    def txt2seq(self, padding=True):
        input_sequences = []
        for line in self.corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        if padding:
            max_sequence_len = max([len(x) for x in input_sequences])
            input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        self.input_sequences = input_sequences
        return input_sequences

    def get_input_data(self) :
        predictors, label = self.input_sequences[:,:-1],self.input_sequences[:,-1]
        label = ku.to_categorical(label, num_classes=self.total_words)

        return predictors, label
