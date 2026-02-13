import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding

class ContextRCN:
    def __init__(self, vocab_size, embedding_dim, max_length, context_vector_dim=16):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.context_vector_dim = context_vector_dim
        self.model = self._build_model()

    def _build_model(self):
        # Input: Sequence of recent conversation tokens (concatenated or processed)
        # Output: A vector representing the "context" state
        input_layer = Input(shape=(self.max_length,))
        x = Embedding(self.vocab_size, self.embedding_dim)(input_layer)
        # Return sequences=True to stack LSTMs or just return the final state
        x = LSTM(32, return_sequences=False)(x)
        output_layer = Dense(self.context_vector_dim, activation='tanh')(x) # Context vector
        
        model = Model(inputs=input_layer, outputs=output_layer, name="Context_RCN")
        model.compile(loss='mse', optimizer='adam') # Unsupervised/Self-supervised conceptually, simplified here
        return model

    def predict(self, processed_text):
        return self.model.predict(processed_text)
    
    def save(self, path='models/context_rcn.h5'):
        self.model.save(path)
    
    def load(self, path='models/context_rcn.h5'):
        self.model = tf.keras.models.load_model(path, compile=False)
