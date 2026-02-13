import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Configuration
MAX_VOCAB_SIZE = 5000
MAX_SEQ_LENGTH = 20
EMBEDDING_DIM = 50

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
        self.max_length = MAX_SEQ_LENGTH
        self.fitted = False

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)
        self.fitted = True

    def transform(self, texts):
        if not self.fitted:
            raise ValueError("Tokenizer not fitted.")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return padded

    def save(self, path='models/tokenizer.pickle'):
        with open(path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path='models/tokenizer.pickle'):
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            self.fitted = True
        else:
            print(f"Warning: {path} not found.")

# Singleton instance to be used across modules
preprocessor = TextPreprocessor()
