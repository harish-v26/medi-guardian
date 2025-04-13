import sqlite3
import os
import hashlib
import uuid
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    """Create a connection to the SQLite database"""
    db_dir = 'database'
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    conn = sqlite3.connect(os.path.join(db_dir, 'mediguardian.db'))
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password, salt=None):
    """Hash a password with SHA-256 and a random salt"""
    if salt is None:
        salt = uuid.uuid4().hex
    
    hashed_password = hashlib.sha256(salt.encode() + password.encode()).hexdigest()
    return salt, hashed_password

def initialize_database():
    """Create database tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        date_of_birth TEXT,
        phone_number TEXT,
        address TEXT,
        created_at TEXT NOT NULL,
        last_login TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS emergency_contacts (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        relationship TEXT NOT NULL,
        phone_number TEXT NOT NULL,
        email TEXT,
        is_primary BOOLEAN NOT NULL DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS medical_history (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        condition TEXT,
        diagnosis_date TEXT,
        notes TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_results (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        test_type TEXT NOT NULL,
        test_date TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        features TEXT NOT NULL,
        audio_file_path TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS doctor_referrals (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        doctor_name TEXT NOT NULL,
        specialty TEXT NOT NULL,
        hospital TEXT,
        address TEXT,
        phone_number TEXT,
        email TEXT,
        notes TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    logging.info("Database initialized successfully")

def register_user(username, password, email, first_name, last_name, dob=None, phone=None, address=None):
    """Register a new user in the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return False, "Username already exists"
        
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            return False, "Email already registered"
        
        user_id = str(uuid.uuid4())
        salt, hashed_password = hash_password(password)
        created_at = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO users (id, username, password_hash, salt, email, first_name, last_name, 
                          date_of_birth, phone_number, address, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, username, hashed_password, salt, email, first_name, last_name, 
              dob, phone, address, created_at))
        
        conn.commit()
        conn.close()
        return True, user_id
    except Exception as e:
        logging.error(f"Error registering user: {e}")
        return False, str(e)

def authenticate_user(username, password):
    """Authenticate a user with username and password"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "Invalid username or password"
        
        salt = user['salt']
        stored_password = user['password_hash']
        
        _, hashed_password = hash_password(password, salt)
        
        if hashed_password == stored_password:
            cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                          (datetime.now().isoformat(), user['id']))
            conn.commit()
            conn.close()
            return True, dict(user)
        else:
            conn.close()
            return False, "Invalid username or password"
    except Exception as e:
        logging.error(f"Error authenticating user: {e}")
        return False, str(e)

def add_emergency_contact(user_id, name, relationship, phone, email=None, is_primary=False):
    """Add an emergency contact for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        contact_id = str(uuid.uuid4())
        
        if is_primary:
            cursor.execute("UPDATE emergency_contacts SET is_primary = 0 WHERE user_id = ?", (user_id,))
        
        cursor.execute('''
        INSERT INTO emergency_contacts (id, user_id, name, relationship, phone_number, email, is_primary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (contact_id, user_id, name, relationship, phone, email, is_primary))
        
        conn.commit()
        conn.close()
        return True, contact_id
    except Exception as e:
        logging.error(f"Error adding emergency contact: {e}")
        return False, str(e)

def save_test_result(user_id, test_type, prediction, confidence, features, audio_path=None):
    """Save a test result to the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        result_id = str(uuid.uuid4())
        test_date = datetime.now().isoformat()
        
        features_str = str(features)
        
        cursor.execute('''
        INSERT INTO test_results (id, user_id, test_type, test_date, prediction, confidence, features, audio_file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (result_id, user_id, test_type, test_date, prediction, confidence, features_str, audio_path))
        
        conn.commit()
        conn.close()
        return True, result_id
    except Exception as e:
        logging.error(f"Error saving test result: {e}")
        return False, str(e)

def get_user_test_history(user_id):
    """Get all test results for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM test_results WHERE user_id = ? ORDER BY test_date DESC
        ''', (user_id,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return True, results
    except Exception as e:
        logging.error(f"Error retrieving test history: {e}")
        return False, str(e)

def get_user_emergency_contacts(user_id):
    """Get all emergency contacts for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM emergency_contacts WHERE user_id = ? ORDER BY is_primary DESC
        ''', (user_id,))
        
        contacts = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return True, contacts
    except Exception as e:
        logging.error(f"Error retrieving emergency contacts: {e}")
        return False, str(e)
