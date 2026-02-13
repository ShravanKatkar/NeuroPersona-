import numpy as np
import tensorflow as tf
from src.preprocessing import preprocessor
from src.models.emotion_ann import EmotionANN
from src.models.intent_ann import IntentANN
from src.models.context_rcn import ContextRCN
from src.models.tone_net import ToneNet
from src.models.generator import ResponseGenerator
from src.personality import PersonalityEngine
from src.analytics import AnalyticsEngine
import os

def main():
    print("Loading NeuroPersona System...")
    
    # Load Preprocessor
    preprocessor.load()
    
    # Load Models
    # Initialize classes first to build structure, then load weights
    # Note: For Keras load_model, we often just need the file if using functional API save,
    # but our wrapper classes expect specific init.
    # To simplify, we will just use the wrappers to load.
    
    
    # Load Models with Debugging
    # Load Models
    try:
        emotion_model = EmotionANN(5000, 50, 20, 4)
        emotion_model.load()
    except Exception as e:
        print(f"Error loading Emotion ANN: {e}")

    try:
        intent_model = IntentANN(5000, 50, 20, 3)
        intent_model.load()
    except Exception as e:
        print(f"Error loading Intent ANN: {e}")

    try:
        context_model = ContextRCN(5000, 50, 20)
        context_model.load()
    except Exception as e:
        print(f"Error loading Context RCN: {e}")

    try:
        tone_model = ToneNet(4, 3, 16, 5, 3)
        tone_model.load()
    except Exception as e:
        print(f"Error loading Tone Net: {e}")

    try:
        generator = ResponseGenerator(5000, 50, 10, 3, 16)
        generator.load()
    except Exception as e:
        print(f"Error loading Response Generator: {e}")
    
    # Engines
    personality_engine = PersonalityEngine()
    analytics = AnalyticsEngine()
    
    user_id = "user_01"
    
    # Session State
    context_state = np.zeros((1, 16)) # Initial context
    emotion_history = []
    personality_history = []
    tone_counts = {}
    
    print("\nNeuroPersona Initialized.")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        # 1. NLP Preprocessing
        processed_input = preprocessor.transform([user_input])
        
        # 2. Emotion & Intent
        emotion_probs = emotion_model.predict(processed_input)
        emotion_idx = np.argmax(emotion_probs)
        emotion_history.append(emotion_idx)
        
        intent_probs = intent_model.predict(processed_input)
        
        # 3. Context Update
        # Ideally this takes sequence of inputs, here we just pass current input for simplicity of demo
        context_state = context_model.predict(processed_input)
        
        # 4. Personality Update
        # Heuristic: Emotion intensity affects personality
        # We simulate an 'interaction signal' based on emotion/intent
        interaction_signal = np.random.rand(5) # Simplified signal extraction
        # e.g., if Angry, lower optimism
        if emotion_idx == 2: # Angry
            interaction_signal[0] = 0.0 # Low optimism signal
            
        current_personality = personality_engine.update_personality(user_id, interaction_signal)
        personality_history.append(current_personality)
        
        # 5. Tone Decision
        tone_probs = tone_model.predict(emotion_probs, intent_probs, context_state, current_personality.reshape(1, -1))
        tone_idx = np.argmax(tone_probs)
        tone_names = ['Formal', 'Casual', 'Empathetic']
        selected_tone = tone_names[tone_idx]
        tone_counts[selected_tone] = tone_counts.get(selected_tone, 0) + 1
        
        # 6. Response Generation
        # Since generator is untrained, we will use a template fallback for the demo
        # but still run the generator to show it works
        _ = generator.generate(tone_probs, context_state)
        
        print(f"Analysis -> Emotion: {['Happy', 'Sad', 'Angry', 'Neutral'][emotion_idx]}, Intent: {['Question', 'Greeting', 'Command'][np.argmax(intent_probs)]}")
        print(f"Personality -> Optimism: {current_personality[0]:.2f}")
        print(f"System Tone -> {selected_tone}")
        print(f"Bot: [Generated Response] I understand you are feeling {['Happy', 'Sad', 'Angry', 'Neutral'][emotion_idx]}. (Tone: {selected_tone})")
        
        # 7. Analytics
        if len(emotion_history) % 3 == 0:
            print("[System] Generating Real-time Analytics...")
            analytics.plot_emotion_trend(emotion_history)
            analytics.plot_personality_evolution(personality_history)
            analytics.plot_tone_distribution(tone_counts)
            print("[System] Graphs updated.")

if __name__ == "__main__":
    main()
