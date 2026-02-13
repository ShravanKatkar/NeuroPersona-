import sqlite3
import numpy as np
import os

class PersonalityEngine:
    def __init__(self, db_path='data/personality.db'):
        self.db_path = db_path
        self._init_db()
        # Default personality vector: [Optimism, Volatility, Formality, Technical, Motivation]
        self.default_personality = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (user_id TEXT PRIMARY KEY, 
                      p1 REAL, p2 REAL, p3 REAL, p4 REAL, p5 REAL,
                      last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

    def get_personality(self, user_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT p1, p2, p3, p4, p5 FROM users WHERE user_id=?", (user_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return np.array(row)
        else:
            self._save_personality(user_id, self.default_personality)
            return self.default_personality.copy()

    def _save_personality(self, user_id, vector):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO users (user_id, p1, p2, p3, p4, p5) VALUES (?, ?, ?, ?, ?, ?)",
                  (user_id, *vector))
        conn.commit()
        conn.close()

    def update_personality(self, user_id, interaction_signal, decay=0.9, learning_rate=0.1):
        """
        New P = decay * Old P + learning_rate * interaction_signal
        interaction_signal: 5D vector estimated from current conversation
        """
        current_p = self.get_personality(user_id)
        new_p = (decay * current_p) + (learning_rate * interaction_signal)
        # Clip to 0-1 range
        new_p = np.clip(new_p, 0.0, 1.0)
        self._save_personality(user_id, new_p)
        return new_p
