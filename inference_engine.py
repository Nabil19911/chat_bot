import sqlite3
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# Thread-local storage for database connections
local_storage = threading.local()

def get_db_connection():
    """Get a thread-local database connection"""
    if not hasattr(local_storage, 'connection'):
        local_storage.connection = sqlite3.connect('banking_bot.db', check_same_thread=False)
        local_storage.connection.row_factory = sqlite3.Row
    return local_storage.connection

class InferenceEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load or train the TF-IDF model"""
        try:
            if os.path.exists('tfidf_model.joblib'):
                self.vectorizer = joblib.load('tfidf_model.joblib')
                print("Loaded existing TF-IDF model")
            else:
                self.train_model()
        except Exception as e:
            print(f"Model loading/training error: {e}")
            raise

    def train_model(self):
        """Train the TF-IDF model"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT user_input FROM static_responses UNION SELECT user_input FROM learned_responses")
            texts = [row[0] for row in cursor.fetchall()]
            
            if texts:
                self.vectorizer.fit(texts)
                joblib.dump(self.vectorizer, 'tfidf_model.joblib')
                print("TF-IDF model trained and saved")
        except Exception as e:
            print(f"Training error: {e}")

    def get_response(self, user_input):
        """Main method to get response for user input"""
        try:
            if not user_input or not isinstance(user_input, str):
                return "Please provide a valid question."
            
            # Check static responses first
            static_response = self.check_static_responses(user_input)
            if static_response:
                return static_response
            
            # Check learned responses
            learned_response = self.check_learned_responses(user_input)
            if learned_response:
                return learned_response
            
            # Handle banking queries
            banking_response = self.handle_banking_queries(user_input)
            if banking_response:
                self.learn_response(user_input, banking_response, "banking")
                return banking_response
            
            return self.handle_unknown_query(user_input)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error processing your request."

    def check_static_responses(self, user_input):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT bot_response FROM static_responses WHERE LOWER(user_input) = LOWER(?)",
            (user_input,)
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def check_learned_responses(self, user_input):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_input FROM learned_responses")
        all_learned = [row[0] for row in cursor.fetchall()]
        
        if not all_learned:
            return None
            
        # Vectorize input and find most similar
        input_vec = self.vectorizer.transform([user_input])
        learned_vecs = self.vectorizer.transform(all_learned)
        
        similarities = cosine_similarity(input_vec, learned_vecs)
        max_index = np.argmax(similarities)
        max_similarity = similarities[0, max_index]
        
        if max_similarity > 0.6:  # Similarity threshold
            cursor.execute(
                "SELECT bot_response FROM learned_responses WHERE user_input = ?",
                (all_learned[max_index],)
            )
            result = cursor.fetchone()
            if result:
                # Update confidence and timestamp
                cursor.execute(
                    "UPDATE learned_responses SET confidence = confidence + 1, last_used = ? WHERE user_input = ?",
                    (datetime.now(), all_learned[max_index]))
                conn.commit()
                return result[0]
        return None

    def handle_banking_queries(self, user_input):
        conn = get_db_connection()
        # Account types query
        if re.search(r'account types|kinds of accounts|what accounts', user_input, re.I):
            cursor = conn.cursor()
            cursor.execute("SELECT account_name, interest_rate, min_balance, description FROM account_types")
            accounts = cursor.fetchall()
            
            if not accounts:
                return "We currently don't have information about account types."
                
            response = "We offer the following account types:\n"
            for acc in accounts:
                response += f"\n- {acc[0]}: {acc[3]} (Interest: {acc[1]}%, Min Balance: ${acc[2]})"
            return response
        
        # Account opening
        elif re.search(r'open an account|create account|new account', user_input, re.I):
            return "To open a new account, please visit our website or nearest branch with your ID and proof of address."
        
        # Balance inquiry
        elif re.search(r'check balance|my balance|account balance', user_input, re.I):
            return "You can check your balance through our mobile app, online banking, or by visiting an ATM."
        
        return None

    def learn_response(self, user_input, bot_response, context):
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO learned_responses (user_input, bot_response, context, last_used) VALUES (?, ?, ?, ?)",
                (user_input, bot_response, context, datetime.now())
            )
            conn.commit()
            self.train_model()  # Retrain with new data
        except sqlite3.IntegrityError:
            # Already exists, just update
            cursor.execute(
                "UPDATE learned_responses SET confidence = confidence + 1, last_used = ? WHERE user_input = ?",
                (datetime.now(), user_input))
            conn.commit()

    def handle_unknown_query(self, user_input):
        return "I'm still learning about banking. Could you rephrase your question or tell me what would be a good response?"

@app.route('/chat', methods=['POST'])
def chat():
    """Flask endpoint for chat requests"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        user_input = data.get('query', '').strip()
        
        if not user_input:
            return jsonify({"error": "Empty query"}), 400
            
        engine = InferenceEngine()
        response = engine.get_response(user_input)
        
        return jsonify({"response": response})
        
    except Exception as e:
        print(f"Endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.teardown_appcontext
def close_connection(exception):
    """Close database connection at the end of each request"""
    if hasattr(local_storage, 'connection'):
        local_storage.connection.close()
        del local_storage.connection

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)