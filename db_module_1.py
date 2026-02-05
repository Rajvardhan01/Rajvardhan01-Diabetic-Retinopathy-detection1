import sqlite3
import os
import hashlib
import time
from datetime import datetime

class Database:
    def __init__(self):
        # Create database directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        self.db_path = 'data/dr_detection.db'
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(f"Database connection error: {str(e)}")
    
    def _create_tables(self):
        """Create tables if they don't exist."""
        try:
            # Users table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TEXT NOT NULL,
                last_login TEXT
            )
            ''')
            
            # Predictions table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            self.conn.commit()
            
            # Create a default admin user if no users exist
            self.cursor.execute("SELECT COUNT(*) FROM users")
            if self.cursor.fetchone()[0] == 0:
                self.create_user("admin", "admin@example.com", "admin123", "Administrator")
                
        except sqlite3.Error as e:
            print(f"Table creation error: {str(e)}")
    
    def _hash_password(self, password):
        """Hash a password with SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, email, password, full_name=None):
        """Create a new user."""
        try:
            password_hash = self._hash_password(password)
            created_at = datetime.now().isoformat()
            
            self.cursor.execute(
                "INSERT INTO users (username, email, password_hash, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, email, password_hash, full_name, created_at)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            raise Exception("Username or email already exists")
        except sqlite3.Error as e:
            print(f"User creation error: {str(e)}")
            return False
    
    def authenticate_user(self, username, password):
        """Authenticate a user."""
        try:
            password_hash = self._hash_password(password)
            
            self.cursor.execute(
                "SELECT * FROM users WHERE username = ? AND password_hash = ?",
                (username, password_hash)
            )
            user = self.cursor.fetchone()
            
            if user:
                # Update last login time
                self.cursor.execute(
                    "UPDATE users SET last_login = ? WHERE id = ?",
                    (datetime.now().isoformat(), user['id'])
                )
                self.conn.commit()
                
                # Convert SQLite Row to dict
                return dict(user)
            
            return None
        except sqlite3.Error as e:
            print(f"Authentication error: {str(e)}")
            return None
    
    def save_prediction(self, user_id, image_path, predicted_class, confidence):
        """Save a prediction result."""
        try:
            timestamp = datetime.now().isoformat()
            
            self.cursor.execute(
                "INSERT INTO predictions (user_id, image_path, predicted_class, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
                (user_id, image_path, predicted_class, confidence, timestamp)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Save prediction error: {str(e)}")
            return False
    
    def get_user_predictions(self, user_id):
        """Get all predictions for a user."""
        try:
            self.cursor.execute(
                "SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC",
                (user_id,)
            )
            predictions = self.cursor.fetchall()
            
            # Convert SQLite Rows to dicts
            return [dict(pred) for pred in predictions]
        except sqlite3.Error as e:
            print(f"Get predictions error: {str(e)}")
            return []
    
    def delete_prediction(self, prediction_id):
        """Delete a prediction."""
        try:
            self.cursor.execute(
                "DELETE FROM predictions WHERE id = ?",
                (prediction_id,)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Delete prediction error: {str(e)}")
            return False
    
    def update_user_profile(self, user_id, full_name, email):
        """Update user profile."""
        try:
            self.cursor.execute(
                "UPDATE users SET full_name = ?, email = ? WHERE id = ?",
                (full_name, email, user_id)
            )
            self.conn.commit()
            
            # Return updated user
            self.cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            return dict(self.cursor.fetchone())
        except sqlite3.Error as e:
            print(f"Update profile error: {str(e)}")
            return None
    
    def update_user_password(self, user_id, current_password, new_password):
        """Update user password."""
        try:
            current_hash = self._hash_password(current_password)
            
            # Verify current password
            self.cursor.execute(
                "SELECT id FROM users WHERE id = ? AND password_hash = ?",
                (user_id, current_hash)
            )
            
            if not self.cursor.fetchone():
                return False  # Current password is incorrect
            
            # Update password
            new_hash = self._hash_password(new_password)
            self.cursor.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_hash, user_id)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Password update error: {str(e)}")
            return False
    
    def delete_user(self, user_id):
        """Delete a user and all associated predictions."""
        try:
            # Delete predictions first (foreign key constraint)
            self.cursor.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
            
            # Delete user
            self.cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Delete user error: {str(e)}")
            return False
    
    def __del__(self):
        """Close database connection when object is destroyed."""
        if self.conn:
            self.conn.close()