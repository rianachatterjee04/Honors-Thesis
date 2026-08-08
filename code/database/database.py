import sqlite3
import os
import uuid
from datetime import datetime

class VideoDatabaseHelper:
    """Database helper for managing video and photo processing."""
    
    # Database configuration
    DB_NAME = "media_processing.db"
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.connect_to_database()
    
    def connect_to_database(self):
        """Connect to SQLite database and create tables if they don't exist."""
        try:
            print("Connecting to database...")
            self.connection = sqlite3.connect(self.DB_NAME)
            self.cursor = self.connection.cursor()
            
            # Create tables if they don't exist
            self.create_tables()
            print("Database connection established.")
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
    
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        # Media files table
        media_table = """
        CREATE TABLE IF NOT EXISTS media_files (
            id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            date_added TIMESTAMP,
            is_processed BOOLEAN DEFAULT FALSE,
            output_path TEXT,
            analysis_path TEXT
        )
        """
        self.cursor.execute(media_table)
        
        # Analysis results table
        analysis_table = """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id TEXT PRIMARY KEY,
            media_id TEXT NOT NULL,
            description TEXT,
            hazards TEXT,
            recommendations TEXT,
            date_analyzed TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES media_files(id)
        )
        """
        self.cursor.execute(analysis_table)
        
        # Detected objects table
        objects_table = """
        CREATE TABLE IF NOT EXISTS detected_objects (
            id TEXT PRIMARY KEY,
            media_id TEXT NOT NULL,
            object_type TEXT,
            confidence REAL,
            frame_number INTEGER,
            position_x REAL,
            position_y REAL,
            width REAL,
            height REAL,
            distance REAL,
            FOREIGN KEY (media_id) REFERENCES media_files(id)
        )
        """
        self.cursor.execute(objects_table)
        
        self.connection.commit()
    
    def scan_media_folders(self, video_folder='videos', photo_folder='photos'):
        """Scan folders for videos and photos and add them to database if not already present."""
        # Ensure folders exist
        for folder in [video_folder, photo_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # Get existing file paths from database
        self.cursor.execute("SELECT file_path FROM media_files")
        existing_paths = [row[0] for row in self.cursor.fetchall()]
        
        # Scan for videos
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        new_videos = self._scan_folder_for_media(video_folder, video_extensions, 'video', existing_paths)
        
        # Scan for photos
        photo_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        new_photos = self._scan_folder_for_media(photo_folder, photo_extensions, 'photo', existing_paths)
        
        return len(new_videos) + len(new_photos)
    
    def _scan_folder_for_media(self, folder_path, extensions, media_type, existing_paths):
        """Scan a folder for files with specified extensions and add to database."""
        new_files = []
        for file_name in os.listdir(folder_path):
            if any(file_name.lower().endswith(ext) for ext in extensions):
                file_path = os.path.abspath(os.path.join(folder_path, file_name))
                
                if file_path not in existing_paths:
                    media_id = str(uuid.uuid4())
                    now = datetime.now()
                    
                    # Insert into database
                    self.cursor.execute(
                        "INSERT INTO media_files (id, file_name, file_path, media_type, date_added) VALUES (?, ?, ?, ?, ?)",
                        (media_id, file_name, file_path, media_type, now)
                    )
                    self.connection.commit()
                    new_files.append((media_id, file_path))
                    print(f"Added {media_type}: {file_name}")
        
        return new_files
    
    def get_unprocessed_media(self, media_type=None, limit=10):
        """Get unprocessed media files from the database."""
        query = "SELECT id, file_name, file_path, media_type FROM media_files WHERE is_processed = 0"
        params = []
        
        if media_type:
            query += " AND media_type = ?"
            params.append(media_type)
        
        query += " ORDER BY date_added LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(query, params)
        return self.cursor.fetchall()
    
    def update_media_processed(self, media_id, output_path=None, analysis_path=None):
        """Mark a media file as processed in the database."""
        query = "UPDATE media_files SET is_processed = 1"
        params = []
        
        if output_path:
            query += ", output_path = ?"
            params.append(output_path)
        
        if analysis_path:
            query += ", analysis_path = ?"
            params.append(analysis_path)
        
        query += " WHERE id = ?"
        params.append(media_id)
        
        self.cursor.execute(query, params)
        self.connection.commit()
        return True
    
    def save_analysis(self, media_id, description, hazards, recommendations):
        """Save analysis results for a media file."""
        analysis_id = str(uuid.uuid4())
        now = datetime.now()
        
        self.cursor.execute(
            "INSERT INTO analysis_results (id, media_id, description, hazards, recommendations, date_analyzed) VALUES (?, ?, ?, ?, ?, ?)",
            (analysis_id, media_id, description, hazards, recommendations, now)
        )
        
        self.connection.commit()
        return analysis_id
    
    def save_detected_objects(self, media_id, objects_data):
        """Save detected objects for a media file."""
        if not objects_data:
            return 0
        
        count = 0
        for obj in objects_data:
            obj_id = str(uuid.uuid4())
            
            self.cursor.execute(
                """INSERT INTO detected_objects 
                   (id, media_id, object_type, confidence, frame_number, 
                    position_x, position_y, width, height, distance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (obj_id, media_id, 
                 obj.get('object_type'), 
                 obj.get('confidence'), 
                 obj.get('frame_number'),
                 obj.get('position_x'), 
                 obj.get('position_y'), 
                 obj.get('width'), 
                 obj.get('height'),
                 obj.get('distance'))
            )
            count += 1
        
        self.connection.commit()
        return count
    
    def close_connection(self):
        """Close the database connection."""
        if self.connection:
            self.cursor.close()
            self.connection.close()
            print("Database connection closed.")

# Example usage
if __name__ == "__main__":
    db = VideoDatabaseHelper()
    
    # Scan for new media
    new_files = db.scan_media_folders()
    print(f"Found {new_files} new media files")
    
    # Get unprocessed videos
    videos = db.get_unprocessed_media(media_type='video')
    print(f"Found {len(videos)} unprocessed videos")
    
    # Close connection
    db.close_connection()