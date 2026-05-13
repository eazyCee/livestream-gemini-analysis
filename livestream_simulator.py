import os
import time
import json
import cv2
import numpy as np
from datetime import datetime
import queue
import threading
from google.cloud import storage

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "livestream_data")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
SOURCE_VIDEO_PATH = os.path.join(DATA_DIR, "source_video.mp4")

# Ensure directories exist
os.makedirs(CHUNKS_DIR, exist_ok=True)

def load_config():
    """Load configuration, fallback to defaults if missing."""
    default_config = {
        "chunk_duration": 10,  # seconds
        "fps": 10,
        "running": True,
        "prompt": "Describe what happened in this video clip. List any events, objects, or movements.",
        "source_type": "generated",  # "generated" or "video"
        "gcs_enabled": False,
        "gcs_chunks_bucket": "",
        "gcs_analysis_bucket": "",
        "gcp_project_id": ""
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**default_config, **json.load(f)}
        except Exception:
            pass
    return default_config

class GCSUploader:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.storage_client = None

    def upload_chunk(self, local_path, gcs_blob_name, bucket_name):
        self.queue.put((local_path, gcs_blob_name, bucket_name))

    def _worker(self):
        while True:
            local_path, gcs_blob_name, bucket_name = self.queue.get()
            print(f"[GCS Uploader] Found chunk in queue: {local_path} -> gs://{bucket_name}/{gcs_blob_name}")
            
            uploaded = False
            while not uploaded:
                if self.storage_client is None:
                    try:
                        self.storage_client = storage.Client()
                    except Exception as e:
                        print(f"[GCS Uploader] Error initializing storage client: {e}. Will retry in 5s...")
                        time.sleep(5)
                        continue
                
                try:
                    bucket = self.storage_client.bucket(bucket_name)
                    blob = bucket.blob(gcs_blob_name)
                    print(f"[GCS Uploader] Uploading {local_path} to GCS bucket {bucket_name}...")
                    blob.upload_from_filename(local_path)
                    print(f"[GCS Uploader] Successfully uploaded {gcs_blob_name} to GCS!")
                    uploaded = True
                except Exception as e:
                    print(f"[GCS Uploader] Upload failed: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                    
            self.queue.task_done()

def main():
    print("Starting livestream simulator...")
    
    # Initialize config
    config = load_config()
    uploader = GCSUploader()
    
    # Canvas size
    width, height = 640, 480
    fps = config["fps"]
    frame_delay = 1.0 / fps
    
    # Programmatic simulation state (used when source_type == "generated")
    ball_x, ball_y = 100, 100
    ball_dx, ball_dy = 6, 4
    ball_radius = 15
    person_x = -100
    person_speed = 3
    
    frame_buffer = []
    cap = None
    current_source = None
    
    # Main loop
    while True:
        # Load config dynamically
        config = load_config()
        if not config.get("running", True):
            # Make sure to release cap if we pause to release file handles
            if cap is not None:
                cap.release()
                cap = None
                current_source = None
            print("Simulator paused. Waiting...")
            time.sleep(1)
            continue
            
        fps = config.get("fps", 10)
        frame_delay = 1.0 / fps
        chunk_duration = config.get("chunk_duration", 10)
        max_frames_per_chunk = chunk_duration * fps
        
        source_type = config.get("source_type", "generated")
        
        # Handle source transition
        if source_type != current_source:
            if cap is not None:
                cap.release()
                cap = None
            current_source = source_type
            print(f"Source changed to: {source_type}")
            
        frame = None
        
        # --- SOURCE TYPE 1: USE CUSTOM VIDEO FILE ---
        if source_type == "video":
            if not os.path.exists(SOURCE_VIDEO_PATH):
                # Fallback to alert frame if file is missing
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame, "NO SOURCE VIDEO UPLOADED", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Please upload a video in the Streamlit sidebar.", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Lazy initialize VideoCapture
                if cap is None:
                    print(f"Loading source video: {SOURCE_VIDEO_PATH}")
                    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
                    
                ret, raw_frame = cap.read()
                
                # Loop video if it reaches the end
                if not ret or raw_frame is None:
                    print("Source video ended, looping back...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, raw_frame = cap.read()
                    
                if ret and raw_frame is not None:
                    # Resize to uniform resolution for camera simulation
                    frame = cv2.resize(raw_frame, (width, height))
                else:
                    # Fallback in case reading still fails
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(frame, "ERROR READING VIDEO SOURCE", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    
        # --- SOURCE TYPE 2: PROGRAMMATIC GENERATION (DEFAULT) ---
        else:
            # Create base dark canvas (dark grey field)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (0, 240), (640, 480), (20, 50, 20), -1)
            cv2.line(frame, (0, 240), (640, 240), (80, 80, 80), 2)
            cv2.line(frame, (320, 240), (320, 480), (80, 80, 80), 1)
            
            # Simulate bouncing ball
            ball_x += ball_dx
            ball_y += ball_dy
            if ball_x - ball_radius < 0 or ball_x + ball_radius > width:
                ball_dx = -ball_dx
            if ball_y - ball_radius < 0 or ball_y + ball_radius > height:
                ball_dy = -ball_dy
            cv2.circle(frame, (ball_x, ball_y), ball_radius, (50, 205, 50), -1)
            cv2.circle(frame, (ball_x, ball_y), ball_radius, (255, 255, 255), 1)
            
            # Simulate walking person shadow
            person_x += person_speed
            if person_x > width + 50:
                person_x = -100
            if person_x > -50 and person_x < width + 50:
                cv2.rectangle(frame, (person_x, 180), (person_x + 40, 320), (40, 40, 40), -1)
                cv2.circle(frame, (person_x + 20, 160), 20, (40, 40, 40), -1)
                
        # --- SECURITY OVERLAY (Applied to ALL source frames) ---
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        ms_str = f"{int(time.time() * 1000) % 1000:03d}"
        
        # Blinking REC dot
        if int(time.time()) % 2 == 0:
            cv2.circle(frame, (30, 30), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (45, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Live Real-time clock
        cv2.putText(frame, f"{timestamp_str}.{ms_str}", (400, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # CAM info
        cam_label = "CAM-01 MAIN GATE" if source_type == "generated" else "CAM-01 BROADCAST FEED"
        cv2.putText(frame, cam_label, (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Save frame to disk for live Streamlit UI view
        current_frame_path = os.path.join(DATA_DIR, "current_frame.jpg")
        temp_frame_path = os.path.join(DATA_DIR, "current_frame_temp.jpg")
        cv2.imwrite(temp_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        try:
            os.replace(temp_frame_path, current_frame_path)
        except OSError:
            pass
            
        # Buffer frames for segment compilation
        frame_buffer.append(frame)
        
        # Check if chunk is complete and write MP4 clip
        if len(frame_buffer) >= max_frames_per_chunk:
            timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = f"chunk_{timestamp_filename}.mp4"
            video_path = os.path.join(CHUNKS_DIR, video_name)
            
            # Use AVC1 (H.264) codec for standard native HTML5 browser video decoding
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for f in frame_buffer:
                out.write(f)
            out.release()
            
            # Upload to GCS if enabled
            if config.get("gcs_enabled", False) and config.get("gcs_chunks_bucket"):
                uploader.upload_chunk(video_path, video_name, config["gcs_chunks_bucket"])
            
            frame_buffer = []
            print(f"Chunk {video_name} compiled successfully.")
            
        time.sleep(frame_delay)

if __name__ == "__main__":
    main()
