import matplotlib.pyplot as plt
import numpy as np

class AnalyticsEngine:
    def __init__(self):
        pass

    def plot_emotion_trend(self, emotion_history):
        """
        emotion_history: list of emotion indices
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(emotion_history, marker='o', linestyle='-', color='b')
        ax.set_title('Emotion Trend Over Time')
        ax.set_xlabel('Interaction Step')
        ax.set_ylabel('Emotion Class')
        ax.grid(True)
        return fig

    def plot_personality_evolution(self, personality_history):
        """
        personality_history: list of vectors (arrays)
        """
        history = np.array(personality_history) # (Steps, 5)
        traits = ['Optimism', 'Volatility', 'Formality', 'Technical', 'Motivation']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        if len(history) > 0:
            for i in range(5):
                ax.plot(history[:, i], label=traits[i])
            
        ax.set_title('Personality Evolution')
        ax.set_xlabel('Interaction Step')
        ax.set_ylabel('Trait Score')
        ax.legend()
        ax.grid(True)
        return fig

    def plot_tone_distribution(self, tone_counts):
        """
        tone_counts: dict {tone_name: count}
        """
        labels = list(tone_counts.keys())
        sizes = list(tone_counts.values())
        
        fig, ax = plt.subplots(figsize=(6, 6))
        if sizes:
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        else:
            ax.text(0.5, 0.5, "No Data Yet", ha='center')
            
        ax.set_title('Tone Usage Distribution')
        return fig
