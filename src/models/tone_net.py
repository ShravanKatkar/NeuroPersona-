import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

class ToneNet:
    def __init__(self, emotion_dim, intent_dim, context_dim, personality_dim, num_tones):
        self.model = self._build_model(emotion_dim, intent_dim, context_dim, personality_dim, num_tones)

    def _build_model(self, emotion_dim, intent_dim, context_dim, personality_dim, num_tones):
        # Inputs
        in_emotion = Input(shape=(emotion_dim,), name='emotion_input')
        in_intent = Input(shape=(intent_dim,), name='intent_input')
        in_context = Input(shape=(context_dim,), name='context_input')
        in_personality = Input(shape=(personality_dim,), name='personality_input')

        # Merge
        merged = Concatenate()([in_emotion, in_intent, in_context, in_personality])
        
        # Dense layers
        x = Dense(32, activation='relu')(merged)
        x = Dense(16, activation='relu')(x)
        output = Dense(num_tones, activation='softmax')(x)

        model = Model(inputs=[in_emotion, in_intent, in_context, in_personality], outputs=output, name="Tone_Net")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def predict(self, emotion, intent, context, personality):
        return self.model.predict({
            'emotion_input': emotion,
            'intent_input': intent,
            'context_input': context,
            'personality_input': personality
        })
    
    def save(self, path='models/tone_net.h5'):
        self.model.save(path)
    
    def load(self, path='models/tone_net.h5'):
        self.model = tf.keras.models.load_model(path, compile=False)
