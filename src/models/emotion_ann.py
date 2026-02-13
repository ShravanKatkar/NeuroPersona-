import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, Dropout

class EmotionANN:
    def __init__(self, vocab_size, embedding_dim, max_length, num_classes):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(self.max_length,))
        x = Embedding(self.vocab_size, self.embedding_dim)(input_layer)
        x = GlobalAveragePooling1D()(x)
        x = Dense(24, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(16, activation='relu')(x)
        output_layer = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer, name="Emotion_ANN")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, processed_text):
        return self.model.predict(processed_text)
    
    def save(self, path='models/emotion_ann.h5'):
        self.model.save(path)
    
    def load(self, path='models/emotion_ann.h5'):
        self.model = tf.keras.models.load_model(path, compile=False)
