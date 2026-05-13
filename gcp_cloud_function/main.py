import functions_framework
from cloudevents.http import CloudEvent
from google import genai
from google.genai import types
from google.cloud import storage
import json
import os
import time
from datetime import datetime
from pydantic import BaseModel, Field

class MatchAnalysisSchema(BaseModel):
    analysis: str = Field(description="Detailed description of what happened in this sports match / livestream video clip.")
    is_highlight: bool = Field(description="True if this clip contains a key moment, scoring attempt, point scored, goal, player celebration, exciting rally, or other significant event.")
    highlight_reason: str = Field(description="A brief one-sentence explanation of why this clip is or is not a highlight.")

# Initialize clients outside the handler for container reuse / warm starts
storage_client = storage.Client()

@functions_framework.cloud_event
def analyze_gcs_chunk(cloud_event: CloudEvent):
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    
    print(f"Processing file: {file_name} from bucket: {bucket_name}")
    
    # Only process MP4 files
    if not file_name.endswith(".mp4"):
        print("Not an MP4 file, skipping.")
        return
        
    # 1. Initialize Gemini client
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = os.environ.get(
        "GEMINI_PROMPT", 
        "Analyze the action in this sports match / livestream video clip. Describe the play in detail. If there is a key moment, scoring attempt, point scored, goal, player celebration, exciting rally, or other significant highlight event, flag it as a highlight."
    )
    
    # Initialize GenAI client
    # The google-genai SDK automatically resolves standard environment variables:
    # - API keys: GEMINI_API_KEY or GOOGLE_API_KEY
    # - Vertex AI: GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT, and GOOGLE_CLOUD_LOCATION
    # - ADC / Service Account credentials automatically on GCP environments (Cloud Functions)
    print("Initializing GenAI client...")
    client = genai.Client()
        
    # 2. Setup GCS references
    analysis_bucket_name = os.environ.get("GCS_ANALYSIS_BUCKET")
    if not analysis_bucket_name:
        print("Error: GCS_ANALYSIS_BUCKET environment variable is not set.")
        return
        
    base_name = os.path.splitext(file_name)[0]
    analysis_file_name = f"{base_name}.json"
    
    analysis_bucket = storage_client.bucket(analysis_bucket_name)
    analysis_blob = analysis_bucket.blob(analysis_file_name)
    
    # 3. Write a placeholder "Analyzing" file to GCS to update the frontend timeline in real-time
    placeholder_data = {
        "video_file": file_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Analyzing",
        "analysis": "Analyzing in progress..."
    }
    try:
        analysis_blob.upload_from_string(
            json.dumps(placeholder_data, indent=2), 
            content_type="application/json"
        )
    except Exception as e:
        print(f"Warning: Failed to upload 'Analyzing' placeholder to GCS: {e}")
    
    # 4. Download the chunk from GCS to container local /tmp directory
    temp_local_path = os.path.join("/tmp", file_name)
    try:
        chunks_bucket = storage_client.bucket(bucket_name)
        chunk_blob = chunks_bucket.blob(file_name)
        print(f"Downloading chunk to container memory: {temp_local_path}")
        chunk_blob.download_to_filename(temp_local_path)
    except Exception as e:
        print(f"Error downloading chunk from GCS: {e}")
        # Write failed status to GCS
        error_data = {
            "video_file": file_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Failed",
            "analysis": f"Failed to download chunk from GCS: {str(e)}"
        }
        analysis_blob.upload_from_string(json.dumps(error_data, indent=2), content_type="application/json")
        return
        
    # 5. Analyze chunk using Gemini API
    try:
        start_time = time.time()
        print(f"Uploading {file_name} to Gemini server...")
        uploaded_file = client.files.upload(file=temp_local_path)
        print(f"Uploaded successfully. Name on server: {uploaded_file.name}")
        
        # Poll for processing status
        print("Waiting for video processing on Gemini servers...")
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)
            print(f"Current state: {uploaded_file.state.name}")
            
        if uploaded_file.state.name != "ACTIVE":
            raise Exception(f"Video processing failed or became inactive: {uploaded_file.state.name}")
            
        print("Video is ACTIVE. Generating analysis content...")
        
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
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                uploaded_file,
                types.Part.from_text(text=prompt)
            ],
            config=generate_content_config
        )
        duration = time.time() - start_time
        
        # Clean up uploaded file on Gemini server
        print(f"Cleaning up uploaded file {uploaded_file.name} from server...")
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception as e:
            print(f"Warning: Failed to delete file from Gemini servers: {e}")
            
        # Parse Gemini structured output
        try:
            parsed_analysis = json.loads(response.text)
        except Exception:
            parsed_analysis = {
                "analysis": response.text,
                "is_highlight": False,
                "highlight_reason": "Failed to parse structured JSON response."
            }
            
        # 6. Save success result to GCS analysis bucket
        result_data = {
            "video_file": file_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Completed",
            "analysis": parsed_analysis.get("analysis", ""),
            "is_highlight": parsed_analysis.get("is_highlight", False),
            "highlight_reason": parsed_analysis.get("highlight_reason", ""),
            "analysis_duration_seconds": round(duration, 2)
        }
        analysis_blob.upload_from_string(
            json.dumps(result_data, indent=2), 
            content_type="application/json"
        )
        print(f"Successfully completed analysis for {file_name} and saved results to GCS!")
        
    except Exception as e:
        print(f"Error during Gemini analysis for {file_name}: {e}")
        # Save failure result to GCS analysis bucket
        error_data = {
            "video_file": file_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Failed",
            "analysis": f"Error during Gemini analysis: {str(e)}"
        }
        analysis_blob.upload_from_string(
            json.dumps(error_data, indent=2), 
            content_type="application/json"
        )
    finally:
        # Clean up /tmp local container file to protect function instance storage limits
        if os.path.exists(temp_local_path):
            try:
                os.remove(temp_local_path)
                print("Cleaned up temporary container file successfully.")
            except Exception as e:
                print(f"Warning: Failed to remove temp file {temp_local_path}: {e}")
