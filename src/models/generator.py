import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, RepeatVector

class ResponseGenerator:
    def __init__(self, vocab_size, embedding_dim, max_length, tone_dim, context_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = self._build_model(tone_dim, context_dim)

    def _build_model(self, tone_dim, context_dim):
        # Condition Input: Tone + Context
        state_input = Input(shape=(tone_dim + context_dim,))
        
        # Decoder
        # We repeat the state vector to match the sequence length we want to generate
        x = RepeatVector(self.max_length)(state_input)
        
        # LSTM
        # simplified generator: conditioning provided as input at each step
        x = LSTM(64, return_sequences=True)(x)
        output = Dense(self.vocab_size, activation='softmax')(x)

        model = Model(inputs=state_input, outputs=output, name="Response_Generator")
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        return model

    def generate(self, tone_vector, context_vector):
        # Concatenate tone and context
        state = tf.concat([tone_vector, context_vector], axis=1)
        # Predict the sequence of tokens
        probs = self.model.predict(state)
        # Greedy search for simplicity (argmax)
        token_ids = tf.argmax(probs, axis=-1).numpy()[0]
        return token_ids
    
    def save(self, path='models/generator.h5'):
        self.model.save(path)
    
    def load(self, path='models/generator.h5'):
        self.model = tf.keras.models.load_model(path, compile=False)
