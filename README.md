# NeuroPersona
## Adaptive Neural Personality Modeling & Behavioral Analytics System

(ANN + RNN + NLP + TensorFlow + Matplotlib)

### 1️⃣ Project Overview
NeuroPersona is an adaptive Generative AI system that models user personality, emotional patterns, and conversational behavior over time using deep neural networks. Unlike traditional chatbots, it builds a dynamic psychological profile of the user and adjusts its tone and strategy.

### 2️⃣ Architecture
- **NLP Processing**: Text to Vector (Tokenization, Padding, GloVe/Embedding).
- **Emotion ANN**: Detects user emotion (Happy, Sad, Angry, etc.).
- **Intent ANN**: Classifies user intent (Question, Casual, Complaint).
- **Context Memory (LSTM)**: Tracks conversation history.
- **Personality Engine**: Updates a persistent personality vector based on interactions.
- **Tone Network**: Decides response style (Formal, Friendly, Direct).
- **Response Generator**: LSTM-based text generation.
- **Analytics**: Matplotlib visualizations of behavioral trends.

### 3️⃣ Setup & Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the models (using synthetic data):
   ```bash
   python train.py
   ```
3. Run the interactive simulation:
   ```bash
   python main.py
   ```

### 4️⃣ Visualizations
The system provides real-time analytics:
- Emotion Trend Over Time
- Personality Evolution
- Tone Usage Distribution

### 5️⃣ Technologies
- Python, TensorFlow 2.x, Keras
- NumPy, Pandas, NLTK
- Matplotlib, SQLite
