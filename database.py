import sqlite3
from datetime import datetime

def init_database():
    conn = sqlite3.connect('banking_bot.db')
    cursor = conn.cursor()
    
    # Static knowledge (phrases, greetings)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS static_responses (
        id INTEGER PRIMARY KEY,
        user_input TEXT UNIQUE,
        bot_response TEXT,
        context TEXT
    )
    ''')
    
    # Dynamic banking knowledge
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS account_types (
        id INTEGER PRIMARY KEY,
        account_name TEXT,
        interest_rate REAL,
        min_balance REAL,
        description TEXT
    )
    ''')
    
    # User interactions for learning
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS learned_responses (
        id INTEGER PRIMARY KEY,
        user_input TEXT,
        bot_response TEXT,
        context TEXT,
        confidence INTEGER DEFAULT 1,
        last_used TIMESTAMP
    )
    ''')
    
    # Insert sample static data
    static_data = [
        ("thank you", "You're welcome!", "greeting"),
        ("good morning", "Good morning! How can I help you today?", "greeting"),
        ("hi", "Hello! Welcome to Peoples Bank. How may I assist you?", "greeting")
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO static_responses (user_input, bot_response, context) VALUES (?, ?, ?)",
        static_data
    )
    
    # Insert sample account types
    account_data = [
        ("Savings Account", 3.5, 500.0, "Standard savings account with interest"),
        ("Current Account", 0.0, 1000.0, "Business account with no interest"),
        ("Fixed Deposit", 5.2, 10000.0, "High-interest fixed term deposit")
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO account_types (account_name, interest_rate, min_balance, description) VALUES (?, ?, ?, ?)",
        account_data
    )
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_database()