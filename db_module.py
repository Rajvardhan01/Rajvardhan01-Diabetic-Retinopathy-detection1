import sqlite3
import os
import hashlib
from datetime import datetime

class Database:
    def __init__(self, db_path="diabetic_retinopathy.db"):
        self.db_path = db_path
        self._create_tables()
    
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _create_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TEXT
            )
        ''')
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_path TEXT,
                prediction_class TEXT,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username, email, password, full_name=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Using SHA-256 for password hashing instead of bcrypt
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            created_at = datetime.utcnow().isoformat()
            
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, email, password_hash, full_name, created_at)
            )
            
            conn.commit()
            
            # Get the created user
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user_data = cursor.fetchone()
            
            return dict(user_data) if user_data else None
            
        except sqlite3.IntegrityError as e:
            conn.rollback()
            if "UNIQUE constraint failed: users.username" in str(e):
                raise Exception("Username already exists")
            elif "UNIQUE constraint failed: users.email" in str(e):
                raise Exception("Email already exists")
            else:
                raise e
        finally:
            conn.close()
    
    def authenticate_user(self, username, password):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        cursor.execute(
            "SELECT * FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )
        
        user_data = cursor.fetchone()
        conn.close()
        
        return dict(user_data) if user_data else None
    
    def save_prediction(self, user_id, image_path, prediction_class, confidence):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            created_at = datetime.utcnow().isoformat()
            
            cursor.execute(
                "INSERT INTO predictions (user_id, image_path, prediction_class, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, image_path, prediction_class, confidence, created_at)
            )
            
            conn.commit()
            
            # Get the created prediction
            prediction_id = cursor.lastrowid
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            prediction_data = cursor.fetchone()
            
            return dict(prediction_data) if prediction_data else None
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_user_predictions(self, user_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        
        predictions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return predictions
    
    def get_user_by_id(self, user_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        
        user_data = cursor.fetchone()
        conn.close()
        
        return dict(user_data) if user_data else None
    
    def update_user_profile(self, user_id, full_name=None, email=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            updates = []
            params = []
            
            if full_name is not None:
                updates.append("full_name = ?")
                params.append(full_name)
            
            if email is not None:
                updates.append("email = ?")
                params.append(email)
            
            if not updates:
                return self.get_user_by_id(user_id)
            
            params.append(user_id)
            
            cursor.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
                tuple(params)
            )
            
            conn.commit()
            
            return self.get_user_by_id(user_id)
            
        except sqlite3.IntegrityError as e:
            conn.rollback()
            if "UNIQUE constraint failed: users.email" in str(e):
                raise Exception("Email already exists")
            else:
                raise e
        finally:
            conn.close()