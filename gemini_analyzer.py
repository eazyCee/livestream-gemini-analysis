import os
import time
import json
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

class MatchAnalysisSchema(BaseModel):
    analysis: str = Field(description="Detailed description of what happened in this sports match / livestream video clip.")
    is_highlight: bool = Field(description="True if this clip contains a key moment, scoring attempt, point scored, goal, player celebration, exciting rally, or other significant event.")
    highlight_reason: str = Field(description="A brief one-sentence explanation of why this clip is or is not a highlight.")

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "livestream_data")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")

# Ensure directories exist
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def load_config():
    default_config = {
        "chunk_duration": 10,
        "fps": 10,
        "running": True,
        "prompt": "Analyze the action in this sports match / livestream video clip. Describe the play in detail. If there is a key moment, scoring attempt, point scored, goal, player celebration, exciting rally, or other significant highlight event, flag it as a highlight.",
        "model": "gemini-2.5-flash",  # standard fast multimodal model
        "use_vertex": False,
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

def get_gemini_client(config):
    """Initialize the Gemini Client based on environment and config."""
    # Try environment variable for Vertex AI first, then config file
    api_key = (
        os.environ.get("GOOGLE_CLOUD_API_KEY") 
        or os.environ.get("GEMINI_API_KEY") 
        or config.get("api_key")
    )
    use_vertex = config.get("use_vertex", False) or (os.environ.get("USE_VERTEX_AI", "false").lower() == "true")
    
    if api_key and use_vertex:
        print("Initializing GenAI client using Vertex AI...")
        return genai.Client(vertexai=True, api_key=api_key)
    elif api_key:
        print("Initializing GenAI client using provided API Key...")
        return genai.Client(api_key=api_key)
    else:
        print("Initializing standard GenAI client (using default credentials/ADC)...")
        return genai.Client()

def analyze_video(client, video_path, config):
    model_name = config.get("model", "gemini-2.5-flash")
    prompt = config.get("prompt", "Describe what happened in this video clip.")
    
    print(f"Uploading {video_path} to Gemini...")
    # 1. Upload the video file using GenAI client
    uploaded_file = client.files.upload(file=video_path)
    print(f"Uploaded successfully. File Name on server: {uploaded_file.name}")
    
    # 2. Poll for processing status
    print("Waiting for video processing on Gemini servers...")
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
        print(f"Current state: {uploaded_file.state.name}")
        
    if uploaded_file.state.name != "ACTIVE":
        raise Exception(f"Video processing failed or became inactive: {uploaded_file.state.name}")
        
    print("Video is ACTIVE. Sending analysis request to model...")
    
    # 3. Generate content config enforcing Structured JSON Outputs
    generate_content_config = types.GenerateContentConfig(
        temperature=1.0,
        top_p=0.95,
        response_mime_type="application/json",
        response_schema=MatchAnalysisSchema,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]
    )
    
    # 4. Request analysis from Gemini
    response = client.models.generate_content(
        model=model_name,
        contents=[
            uploaded_file,
            types.Part.from_text(text=prompt)
        ],
        config=generate_content_config
    )
    
    # 5. Clean up uploaded file on Gemini server
    print(f"Cleaning up uploaded file {uploaded_file.name} from server...")
    try:
        client.files.delete(name=uploaded_file.name)
        print("Cleaned up successfully.")
    except Exception as e:
        print(f"Warning: Failed to delete file from Gemini servers: {e}")
        
    return response.text

def cleanup_old_files(max_retained_videos=10, max_retained_analyses=30):
    """Clean up old MP4 video segments and JSON analysis files to protect disk space."""
    try:
        # 1. Clean up old video chunks
        if os.path.exists(CHUNKS_DIR):
            mp4_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.mp4')])
            
            # Find which files are safe to delete (must be analyzed already)
            analyzed_mp4_files = []
            for f in mp4_files:
                base_name = os.path.splitext(f)[0]
                analysis_path = os.path.join(ANALYSIS_DIR, f"{base_name}.json")
                if os.path.exists(analysis_path):
                    try:
                        with open(analysis_path, 'r') as af:
                            status = json.load(af).get("status", "")
                        if status in ["Completed", "Failed"]:
                            analyzed_mp4_files.append(f)
                    except Exception:
                        pass
            
            # Only clean up if analyzed files count exceeds retention limit
            if len(analyzed_mp4_files) > max_retained_videos:
                files_to_delete = analyzed_mp4_files[:-max_retained_videos]
                for f in files_to_delete:
                    file_path = os.path.join(CHUNKS_DIR, f)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"[Cleanup] Deleted old video chunk: {f}")
                        
        # 2. Clean up old JSON analyses
        if os.path.exists(ANALYSIS_DIR):
            json_files = sorted([f for f in os.listdir(ANALYSIS_DIR) if f.endswith('.json')])
            if len(json_files) > max_retained_analyses:
                files_to_delete = json_files[:-max_retained_analyses]
                for f in files_to_delete:
                    file_path = os.path.join(ANALYSIS_DIR, f)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"[Cleanup] Deleted old analysis report: {f}")
    except Exception as e:
        print(f"[Cleanup] Error during sliding-window cleanup: {e}")

def main():
    print("Starting Gemini livestream video analyzer...")
    
    client = None
    print("Watcher started. Monitoring chunks directory...")
    
    while True:
        config = load_config()
        if not config.get("running", True):
            time.sleep(2)
            continue
            
        try:
            # Process the newest chunks first to minimize live-viewing latency and prioritize active match feed segments!
            chunk_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.mp4')], reverse=True)
        except Exception as e:
            print(f"Error reading chunks directory: {e}")
            time.sleep(5)
            continue
            
        for chunk_file in chunk_files:
            base_name = os.path.splitext(chunk_file)[0]
            analysis_file = f"{base_name}.json"
            analysis_path = os.path.join(ANALYSIS_DIR, analysis_file)
            video_path = os.path.join(CHUNKS_DIR, chunk_file)
            
            # If already analyzed (completed or failed or in progress), check status
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, 'r') as f:
                        current_status = json.load(f).get("status", "")
                    # If it's completed or failed, we skip. 
                    # If it's "Missing Key", we retry since the key might be set now.
                    if current_status in ["Completed", "Failed", "Analyzing"]:
                        continue
                except Exception:
                    pass
                
            print(f"\n--- New chunk detected: {chunk_file} ---")
            
            # Lazy initialize/re-initialize Gemini Client if not loaded yet
            if client is None:
                try:
                    client = get_gemini_client(config)
                except Exception as e:
                    print(f"Gemini Client initialization failed: {e}")
                    # Save a "Missing API Key" state
                    missing_key_data = {
                        "video_file": chunk_file,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "Missing API Key",
                        "analysis": "Gemini API Key is not configured yet. Please enter it in the Streamlit sidebar."
                    }
                    with open(analysis_path, 'w') as f:
                        json.dump(missing_key_data, f, indent=2)
                    continue
            
            # Check if video file exists on disk
            if not os.path.exists(video_path):
                print(f"Warning: Video file {chunk_file} missing from disk, skipping analysis.")
                error_data = {
                    "video_file": chunk_file,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Cleaned Up",
                    "analysis": "The source video chunk was removed from disk before analysis could complete."
                }
                with open(analysis_path, 'w') as f:
                    json.dump(error_data, f, indent=2)
                continue
            
            # Create a placeholder/temp analysis file to mark as "Analyzing"
            placeholder_data = {
                "video_file": chunk_file,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Analyzing",
                "analysis": "Analyzing in progress..."
            }
            with open(analysis_path, 'w') as f:
                json.dump(placeholder_data, f, indent=2)
                
            try:
                # Perform Gemini analysis
                start_time = time.time()
                analysis_text = analyze_video(client, video_path, config)
                duration = time.time() - start_time
                
                # Parse Gemini structured output
                try:
                    parsed_analysis = json.loads(analysis_text)
                except Exception:
                    parsed_analysis = {
                        "analysis": analysis_text,
                        "is_highlight": False,
                        "highlight_reason": "Failed to parse structured JSON response."
                    }
                
                # Save success result
                result_data = {
                    "video_file": chunk_file,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Completed",
                    "analysis": parsed_analysis.get("analysis", ""),
                    "is_highlight": parsed_analysis.get("is_highlight", False),
                    "highlight_reason": parsed_analysis.get("highlight_reason", ""),
                    "analysis_duration_seconds": round(duration, 2)
                }
                with open(analysis_path, 'w') as f:
                    json.dump(result_data, f, indent=2)
                print(f"Successfully analyzed {chunk_file}.")
                
            except Exception as e:
                print(f"Error analyzing {chunk_file}: {e}")
                # Check if the error was due to credentials/auth (e.g. API key became invalid)
                if "API_KEY" in str(e) or "API key" in str(e) or "credentials" in str(e) or "key inputs" in str(e):
                    # Reset client so we re-initialize next time
                    client = None
                    
                # Save failure result
                error_data = {
                    "video_file": chunk_file,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Failed",
                    "analysis": f"Error during Gemini analysis: {str(e)}"
                }
                with open(analysis_path, 'w') as f:
                    json.dump(error_data, f, indent=2)
                    
        # Disable automatic cleanup to preserve all video chunks for highlight reel stitching
        # cleanup_old_files(max_retained_videos=10, max_retained_analyses=30)
        
        time.sleep(2)

if __name__ == "__main__":
    main()
