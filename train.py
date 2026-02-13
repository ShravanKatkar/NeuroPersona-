import numpy as np
import tensorflow as tf
from src.preprocessing import preprocessor
from src.models.emotion_ann import EmotionANN
from src.models.intent_ann import IntentANN
from src.models.context_rcn import ContextRCN
from src.models.tone_net import ToneNet
from src.models.generator import ResponseGenerator
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

# --- Synthetic Data Generation ---
print("Generating Synthetic Data...")

# Emotion Data (Text -> Emotion Class Index)
emotion_data = [
    ("I am so happy today", 0), ("This is amazing", 0), ("I feel great", 0),
    ("I am very sad", 1), ("This is terrible", 1), ("I feel buttoned down", 1),
    ("I am angry at you", 2), ("This is frustrating", 2), ("I hate this", 2),
    ("This is okay", 3), ("I don't know", 3), ("Maybe", 3),
]
# Repeat to make dataset larger
emotion_texts, emotion_labels = zip(* (emotion_data * 50))
emotion_labels = tf.keras.utils.to_categorical(emotion_labels, num_classes=4)

# Intent Data (Text -> Intent Class Index)
intent_data = [
    ("How does this work?", 0), ("What is your name?", 0), ("Can you help?", 0), # Question
    ("Hello there", 1), ("Good morning", 1), ("Hi", 1), # Greeting
    ("I want to delete this", 2), ("Stop it", 2), ("Cancel subscription", 2), # Request/Command
]
intent_texts, intent_labels = zip(* (intent_data * 50))
intent_labels = tf.keras.utils.to_categorical(intent_labels, num_classes=3)

# Fit Tokenizer
all_texts = list(emotion_texts) + list(intent_texts)
preprocessor.fit(all_texts)
preprocessor.save()

# Transform Data
X_emotion = preprocessor.transform(emotion_texts)
X_intent = preprocessor.transform(intent_texts)

# --- Training Models ---

# 1. Emotion ANN
print("Training Emotion ANN...")
emotion_model = EmotionANN(vocab_size=5000, embedding_dim=50, max_length=20, num_classes=4)
emotion_model.train(np.array(X_emotion), np.array(emotion_labels), epochs=10)
emotion_model.save()

# 2. Intent ANN
print("Training Intent ANN...")
intent_model = IntentANN(vocab_size=5000, embedding_dim=50, max_length=20, num_classes=3)
intent_model.train(np.array(X_intent), np.array(intent_labels), epochs=10)
intent_model.save()

# 3. Context RCN (Dummy Training for initialization)
print("Initializing Context RCN...")
context_model = ContextRCN(vocab_size=5000, embedding_dim=50, max_length=20)
# Feed random data just to build the graph and save
dummy_input = np.random.randint(0, 5000, (1, 20))
context_model.predict(dummy_input)
context_model.save()

# 4. Tone Net (Dummy Training)
print("Initializing Tone Net...")
tone_model = ToneNet(emotion_dim=4, intent_dim=3, context_dim=16, personality_dim=5, num_tones=3)
# Dummy forward pass
tone_model.predict(
    np.random.rand(1, 4), np.random.rand(1, 3), np.random.rand(1, 16), np.random.rand(1, 5)
)
tone_model.save()

# 5. Response Generator (Dummy Training)
print("Initializing Response Generator...")
generator_model = ResponseGenerator(vocab_size=5000, embedding_dim=50, max_length=10, tone_dim=3, context_dim=16)
# Dummy forward pass (generating 1 sequence)
generator_model.generate( 
    tf.constant(np.random.rand(1, 3), dtype=tf.float32), 
    tf.constant(np.random.rand(1, 16), dtype=tf.float32)
)
generator_model.save()

print("All models trained and saved successfully.")
