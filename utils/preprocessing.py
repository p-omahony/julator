import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import layers

def read_data(file):
    file = open(file, "r", encoding = "utf8")
    text = file.read()
    #text = text.replace("\n\n", "\n").replace('"', '').lower()

    return text

def clean_lyrics(text) :
    letters = ('!', '[', ']', 'ó', '%', '&', 'å','ø', 'æ', 'é', 'à', 'è', 'ù', 'â', 'ê', 'î', 'ô', 'û', 'ç', '?', '(', ')', ',', 'ë', '-', '2', 'œ', "'", 'ï', '9', '.', '"')
    replacements = ('', '', '', 'o', '', '','a','o','ae', 'e', 'a', 'e', 'u', 'a', 'e', 'i', 'o', 'u', 'c', ' ', ' ', ' ', '', 'e', ' ', ' ','oe', ' ', 'i', '', '', '', '')
    translationTable = str.maketrans(dict(zip(letters, replacements)))
    cleaned_text = text.translate(translationTable)
    lyrics = cleaned_text.lower().split("\n")
    cleaned_lyrics = np.unique(lyrics)[1:].tolist()

    return cleaned_lyrics

def write_lyrics(file, lyrics) :
    for l in lyrics :
        with open(file, 'a') as f :
            f.write(l.strip()+'\n')

def build_input_ds(batch_size, vocab_size, maxlen, filenames) :
    random.shuffle(filenames)
    text_ds = tf.data.TextLineDataset(filenames)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)

    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size - 1,
        output_mode="int",
        output_sequence_length=maxlen + 1,
    )
    vectorize_layer.adapt(text_ds)
    vocab = vectorize_layer.get_vocabulary()

    def prepare_lm_inputs_labels(text):
        """
        Shift word sequences by 1 position so that the target for position (i) is
        word at position (i+1). The model will use all words up till position (i)
        to predict the next word.
        """
        text = tf.expand_dims(text, -1)
        tokenized_sentences = vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y


    text_ds = text_ds.map(prepare_lm_inputs_labels)
    text_ds = text_ds.prefetch(tf.data.AUTOTUNE)

    return text_ds
