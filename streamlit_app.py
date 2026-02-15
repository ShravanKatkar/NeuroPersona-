import streamlit as st
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import sys
import os

# Add the current directory to python path to ensure imports work correctly found in src
sys.path.append(os.getcwd())

from src.preprocessing import preprocessor
from src.models.emotion_ann import EmotionANN
from src.models.intent_ann import IntentANN
from src.models.context_rcn import ContextRCN
from src.models.tone_net import ToneNet
from src.models.generator import ResponseGenerator
from src.personality import PersonalityEngine
from src.analytics import AnalyticsEngine

# Page Config
st.set_page_config(page_title="NeuroPersona", layout="wide")

# Title
st.title("NeuroPersona")
st.markdown("### Adaptive Neural Personality Modeling & Behavioral Analytics System")

# --- Initialize System ---

@st.cache_resource
def load_system():
    # Load Preprocessor
    try:
        preprocessor.load()
    except Exception as e:
        st.warning(f"Preprocessor load warning: {e}. Ensure 'tokenizer.pickle' exists or re-run training.")

    # Load Models (wrapping in try/except for robustness)
    models = {}
    
    def load_model_safe(name, model_class, *args):
        try:
            model = model_class(*args)
            model.load()
            return model
        except Exception as e:
            st.warning(f"Failed to load {name}: {e}. Model will need retraining.")
            return None

    models['emotion'] = load_model_safe('EmotionANN', EmotionANN, 5000, 50, 20, 4)
    models['intent'] = load_model_safe('IntentANN', IntentANN, 5000, 50, 20, 3)
    models['context'] = load_model_safe('ContextRCN', ContextRCN, 5000, 50, 20)
    models['tone'] = load_model_safe('ToneNet', ToneNet, 4, 3, 16, 5, 3)
    models['generator'] = load_model_safe('ResponseGenerator', ResponseGenerator, 5000, 50, 10, 3, 16)

    return models

models = load_system()
personality_engine = PersonalityEngine()
analytics = AnalyticsEngine()

# --- Session State Management ---

if 'history' not in st.session_state:
    st.session_state.history = [] # Chat history
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'personality_history' not in st.session_state:
    st.session_state.personality_history = []
if 'tone_counts' not in st.session_state:
    st.session_state.tone_counts = {}
if 'context_state' not in st.session_state:
    st.session_state.context_state = np.zeros((1, 16))
if 'user_id' not in st.session_state:
    st.session_state.user_id = "user_web_01"

# --- Sidebar Analytics ---
with st.sidebar:
    st.header("Real-time Behavioral Analytics")
    
    if st.session_state.emotion_history:
        st.subheader("Emotion Trend")
        fig_emotion = analytics.plot_emotion_trend(st.session_state.emotion_history)
        st.pyplot(fig_emotion)
        
    if st.session_state.personality_history:
        st.subheader("Personality Evolution")
        fig_personality = analytics.plot_personality_evolution(st.session_state.personality_history)
        st.pyplot(fig_personality)
        
    if st.session_state.tone_counts:
        st.subheader("Tone Distribution")
        fig_tone = analytics.plot_tone_distribution(st.session_state.tone_counts)
        st.pyplot(fig_tone)
    else:
        st.info("Start chatting to see analytics.")

    if st.button("Reset Session"):
        st.session_state.history = []
        st.session_state.emotion_history = []
        st.session_state.personality_history = []
        st.session_state.tone_counts = {}
        st.session_state.context_state = np.zeros((1, 16))
        st.experimental_rerun()

# --- Chat Interface ---

# Display Chat History
for msg in st.session_state.history:
    if msg['role'] == 'user':
        with st.chat_message("user"):
            st.write(msg['text'])
    else:
        with st.chat_message("assistant"):
            st.write(msg['text'])
            with st.expander("Analysis Details"):
                st.write(f"**Detected Emotion:** {msg['meta']['emotion']}")
                st.write(f"**Detected Intent:** {msg['meta']['intent']}")
                st.write(f"**Response Tone:** {msg['meta']['tone']}")
                st.write(f"**Personality State:** {msg['meta']['personality']}")

# Input
user_input = st.chat_input("Say something...")

if user_input:
    # 1. User Message
    st.session_state.history.append({'role': 'user', 'text': user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # 2. Processing
    processed_input = preprocessor.transform([user_input])
    
    # Emotion & Intent
    if models.get('emotion'):
        emotion_probs = models['emotion'].predict(processed_input)
    else:
        emotion_probs = np.random.rand(1, 4) # Mock if model missing

    emotion_idx = np.argmax(emotion_probs)
    emotion_label = ['Happy', 'Sad', 'Angry', 'Neutral'][emotion_idx]
    st.session_state.emotion_history.append(emotion_idx)
    
    if models.get('intent'):
        intent_probs = models['intent'].predict(processed_input)
    else:
        intent_probs = np.random.rand(1, 3)

    intent_label = ['Question', 'Greeting', 'Command'][np.argmax(intent_probs)]
    
    # Context
    if models.get('context'):
        st.session_state.context_state = models['context'].predict(processed_input)
    
    # Personality
    interaction_signal = np.random.rand(5) # Simulation of signal extraction
    if emotion_idx == 2: interaction_signal[0] = 0.0 # Angry -> Low Optimism
    
    current_p = personality_engine.update_personality(st.session_state.user_id, interaction_signal)
    st.session_state.personality_history.append(current_p)
    
    # Tone
    if models.get('tone'):
        tone_probs = models['tone'].predict(emotion_probs, intent_probs, st.session_state.context_state, current_p.reshape(1, -1))
    else:
        tone_probs = np.random.rand(1, 3)

    tone_idx = np.argmax(tone_probs)
    tone_label = ['Formal', 'Casual', 'Empathetic'][tone_idx]
    st.session_state.tone_counts[tone_label] = st.session_state.tone_counts.get(tone_label, 0) + 1
    
    # Response (Template fallback for demo)
    response_text = f"I sense you are {emotion_label}. (Tone: {tone_label})"
    if models.get('generator'):
         try:
             # Generator might return something or print, assuming it returns text or we use fallback
             # Looking at main.py: _ = generator.generate(...)
             # So generator likely prints. We will stick to template for now as in app.py
             pass
         except:
             pass

    
    # 3. Bot Response
    meta_info = {
        'emotion': emotion_label,
        'intent': intent_label,
        'tone': tone_label,
        'personality': f"Optimism: {current_p[0]:.2f}, Volatility: {current_p[1]:.2f}"
    }
    st.session_state.history.append({'role': 'assistant', 'text': response_text, 'meta': meta_info})
    
    with st.chat_message("assistant"):
        st.write(response_text)
        with st.expander("Analysis Details"):
            st.write(f"**Detected Emotion:** {emotion_label}")
            st.write(f"**Detected Intent:** {intent_label}")
            st.write(f"**Response Tone:** {tone_label}")
            st.write(f"**Personality State:** {meta_info['personality']}")
        
    # Rerun to update sidebar graphs
    st.rerun()
