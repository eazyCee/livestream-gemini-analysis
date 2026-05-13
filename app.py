import os
import json
import time
from datetime import datetime, timedelta
import streamlit as st
from google.cloud import storage

# Page Config
st.set_page_config(
    page_title="Gemini Livestream Security Analyzer",
    page_icon="📹",
    layout="wide"
)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "livestream_data")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")

# Ensure directories exist
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

HIGHLIGHTS_DIR = os.path.join(DATA_DIR, "highlights")
REELS_DIR = os.path.join(DATA_DIR, "reels")
os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)
os.makedirs(REELS_DIR, exist_ok=True)

import cv2
import numpy as np

def stitch_videos(video_paths, output_path):
    """Stitch multiple video chunks together sequentially into a single MP4 file."""
    if not video_paths:
        return False
        
    # Find first valid video to inspect parameters
    valid_video = None
    for vp in video_paths:
        if os.path.exists(vp):
            valid_video = vp
            break
            
    if not valid_video:
        return False
        
    cap = cv2.VideoCapture(valid_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 10.0  # fallback
    cap.release()
    
    try:
        # Use AVC1 (H.264) codec for standard native HTML5 browser video decoding
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Fallback for Linux environments without hardware H.264 devices enabled
        if not out.isOpened():
            print("[Warning] 'avc1' codec failed in stitcher. Trying 'mp4v' fallback...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        if not out.isOpened():
            print("[Warning] 'mp4v' codec failed in stitcher. Trying 'XVID' fallback...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for vp in video_paths:
            if not os.path.exists(vp):
                continue
                
            cap = cv2.VideoCapture(vp)
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                # Resize to guarantee dimension match for stitching
                resized_frame = cv2.resize(frame, (width, height))
                out.write(resized_frame)
            cap.release()
            
        out.release()
        return True
    except Exception as e:
        print(f"Error stitching videos: {e}")
        return False

def get_segment_display_time(video_file):
    """Extract the segment clock timestamp from the filename itself for clear UX chronological order."""
    try:
        parts = video_file.replace("chunk_", "").replace(".mp4", "").split("_")
        date_str = parts[0]
        time_str = parts[1]
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

def load_config():
    default_config = {
        "chunk_duration": 10,
        "fps": 10,
        "running": True,
        "prompt": "Analyze the action in this sports match / livestream video clip. Describe the play in detail. If there is a key moment, scoring attempt, point scored, goal, player celebration, exciting rally, or other significant highlight event, flag it as a highlight.",
        "model": "gemini-2.5-flash",
        "use_vertex": False,
        "api_key": "",
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

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        st.error(f"Error saving configuration: {e}")

def get_storage_client():
    """Initialize and return GCS storage client, caching it in Streamlit session state for performance."""
    if "gcs_client" not in st.session_state:
        try:
            st.session_state.gcs_client = storage.Client()
        except Exception:
            return None
    return st.session_state.gcs_client

def ensure_local_chunk(video_file, config):
    """Ensure that a video chunk is available locally in Chunks directory, downloading from GCS if needed."""
    local_path = os.path.join(CHUNKS_DIR, video_file)
    if os.path.exists(local_path):
        return local_path
        
    if config.get("gcs_enabled", False) and config.get("gcs_chunks_bucket"):
        storage_client = get_storage_client()
        if storage_client:
            try:
                bucket = storage_client.bucket(config["gcs_chunks_bucket"])
                blob = bucket.blob(video_file)
                if blob.exists():
                    os.makedirs(CHUNKS_DIR, exist_ok=True)
                    with st.spinner(f"Downloading {video_file} from Cloud Storage for stitching..."):
                        blob.download_to_filename(local_path)
                    return local_path
            except Exception as e:
                st.error(f"Error downloading {video_file} from GCS: {e}")
    return None

def get_all_chunk_names(config):
    """Get names of all mp4 chunks chronologically from GCS if enabled, falling back to local disk."""
    if config.get("gcs_enabled", False) and config.get("gcs_chunks_bucket"):
        storage_client = get_storage_client()
        if storage_client:
            try:
                bucket = storage_client.bucket(config["gcs_chunks_bucket"])
                blobs = bucket.list_blobs(prefix="")
                return sorted([b.name for b in blobs if b.name.endswith('.mp4')])
            except Exception as e:
                st.error(f"Error listing GCS chunks: {e}")
                return []
    # Local fallback
    try:
        return sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.mp4')])
    except Exception:
        return []

# 1. Initialize Session State & Load Config
config = load_config()

# Title
st.title("📹 Gemini Livestream Security Analyzer")
st.markdown(
    "This application simulates a security camera feed, automatically splits the livestream "
    "into short video clips, and sends them to **Gemini 2.5** for real-time event analysis."
)

# 2. Sidebar Configuration Panel
st.sidebar.header("🔑 Credentials & Setup")

# API Key
api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=config.get("api_key", ""),
    type="password",
    help="Enter your Gemini API key to enable Gemini analysis. Key will be saved locally in livestream_data/config.json."
)

# Update API Key if changed
if api_key != config.get("api_key", ""):
    config["api_key"] = api_key
    save_config(config)
    st.sidebar.success("API Key saved successfully!")

st.sidebar.markdown("---")
st.sidebar.header("☁️ Google Cloud Storage (GCS)")

gcs_enabled = st.sidebar.checkbox(
    "Enable GCS Integration",
    value=config.get("gcs_enabled", False),
    help="Toggle this on to ingest compiled video segments and load Gemini analysis directly from GCS."
)
if gcs_enabled != config.get("gcs_enabled", False):
    config["gcs_enabled"] = gcs_enabled
    save_config(config)
    st.rerun()

gcs_chunks_bucket = st.sidebar.text_input(
    "GCS Chunks Bucket",
    value=config.get("gcs_chunks_bucket", ""),
    help="GCS bucket where video segment clips are uploaded (e.g. livestream-chunks-bucket)."
)
if gcs_chunks_bucket != config.get("gcs_chunks_bucket", ""):
    config["gcs_chunks_bucket"] = gcs_chunks_bucket
    save_config(config)

gcs_analysis_bucket = st.sidebar.text_input(
    "GCS Analysis Bucket",
    value=config.get("gcs_analysis_bucket", ""),
    help="GCS bucket where Cloud Function writes the analysis JSON reports."
)
if gcs_analysis_bucket != config.get("gcs_analysis_bucket", ""):
    config["gcs_analysis_bucket"] = gcs_analysis_bucket
    save_config(config)

gcp_project_id = st.sidebar.text_input(
    "GCP Project ID",
    value=config.get("gcp_project_id", ""),
    help="Optional. GCP Project ID for initializing storage clients if required."
)
if gcp_project_id != config.get("gcp_project_id", ""):
    config["gcp_project_id"] = gcp_project_id
    save_config(config)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Livestream Simulator Settings")

# Running toggle
simulator_running = st.sidebar.checkbox(
    "Simulator Active (Running)",
    value=config.get("running", True),
    help="Uncheck this to temporarily pause the livestream simulation."
)
if simulator_running != config.get("running", True):
    config["running"] = simulator_running
    save_config(config)

# Feed Source Selection
source_type = st.sidebar.radio(
    "Livestream Source Type",
    options=["generated", "video"],
    format_func=lambda x: "Generated Security Camera" if x == "generated" else "Custom Video File Upload",
    index=0 if config.get("source_type", "generated") == "generated" else 1,
    help="Choose whether to stream programmatically generated events or upload your own video to be simulated as a security livestream feed."
)
if source_type != config.get("source_type", "generated"):
    config["source_type"] = source_type
    save_config(config)

# Video Uploader if source type is 'video'
if source_type == "video":
    uploaded_video = st.sidebar.file_uploader(
        "Upload Livestream Source Video",
        type=["mp4", "mov", "avi"],
        help="Upload a pre-recorded video file. It will be streamed frame-by-frame with live security camera overlays."
    )
    if uploaded_video is not None:
        source_video_path = os.path.join(DATA_DIR, "source_video.mp4")
        try:
            with open(source_video_path, "wb") as f:
                f.write(uploaded_video.read())
            st.sidebar.success("Source video uploaded successfully! The feed simulator will loop this video.")
        except Exception as e:
            st.sidebar.error(f"Error saving uploaded video: {e}")

# Chunk Duration
chunk_duration = st.sidebar.slider(
    "Chunk Segment Duration (sec)",
    min_value=5,
    max_value=30,
    value=config.get("chunk_duration", 10),
    step=5,
    help="Select how often the video feed should be grouped up and sent to Gemini."
)
if chunk_duration != config.get("chunk_duration", 10):
    config["chunk_duration"] = chunk_duration
    save_config(config)

# Model Selection
model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"]
selected_model = st.sidebar.selectbox(
    "Gemini Model",
    options=model_options,
    index=model_options.index(config.get("model", "gemini-2.5-flash"))
)
if selected_model != config.get("model", "gemini-2.5-flash"):
    config["model"] = selected_model
    save_config(config)

# Analysis Prompt
prompt_text = st.sidebar.text_area(
    "Analysis Prompt",
    value=config.get("prompt", "Describe what happened in this video clip in detail. List any movements, objects, or interesting events."),
    height=120
)
if prompt_text != config.get("prompt", ""):
    config["prompt"] = prompt_text
    save_config(config)

# Fetch system counts
if config.get("gcs_enabled", False) and config.get("gcs_chunks_bucket") and config.get("gcs_analysis_bucket"):
    storage_client = get_storage_client()
    if storage_client:
        try:
            chunks_bucket = storage_client.bucket(config["gcs_chunks_bucket"])
            analysis_bucket = storage_client.bucket(config["gcs_analysis_bucket"])
            chunks_count = len(list(chunks_bucket.list_blobs(prefix="")))
            analysis_count = len(list(analysis_bucket.list_blobs(prefix="")))
        except Exception:
            chunks_count, analysis_count = 0, 0
    else:
        chunks_count, analysis_count = 0, 0
else:
    try:
        chunks_count = len([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.mp4')])
        analysis_count = len([f for f in os.listdir(ANALYSIS_DIR) if f.endswith('.json')])
    except Exception:
        chunks_count, analysis_count = 0, 0

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**System Status:**\n"
    f"- Chunks generated: `{chunks_count}`\n"
    f"- Chunks analyzed: `{analysis_count}`"
)

# 3. Main Layout Components
@st.fragment(run_every=0.5)
def render_live_feed(play):
    current_frame_path = os.path.join(DATA_DIR, "current_frame.jpg")
    if os.path.exists(current_frame_path):
        try:
            if play:
                st.image(current_frame_path, width="stretch", caption="CAM-01 Live View (2 FPS)")
            else:
                st.image(current_frame_path, width="stretch", caption="CAM-01 Paused")
        except Exception:
            pass
    else:
        st.info("Simulator is not running or hasn't generated any frames yet.")

col_feed, col_timeline = st.columns([1, 1])

with col_feed:
    st.subheader("🔴 Live Security Feed (CAM-01)")
    
    # Control to enable auto-refresh/playback of the stream
    play_feed = st.checkbox("Play Live Stream (Interactive Feed)", value=True)
    
    # Render the live feed inside the high-performance fragment container
    render_live_feed(play_feed)

with col_timeline:
    st.subheader("⏳ Timeline & Highlights Manager")
    
    # Load all analyses
    if config.get("gcs_enabled", False) and config.get("gcs_analysis_bucket"):
        storage_client = get_storage_client()
        if storage_client:
            try:
                bucket = storage_client.bucket(config["gcs_analysis_bucket"])
                blobs = bucket.list_blobs(prefix="")
                analysis_files = sorted([b.name for b in blobs if b.name.endswith('.json')], reverse=True)
            except Exception as e:
                st.error(f"Error listing GCS analysis bucket: {e}")
                analysis_files = []
        else:
            analysis_files = []
    else:
        try:
            analysis_files = sorted([f for f in os.listdir(ANALYSIS_DIR) if f.endswith('.json')], reverse=True)
        except Exception:
            analysis_files = []
        
    if not analysis_files:
        st.info("Waiting for video segments to be generated and analyzed...")
    else:
        # Create a beautiful 3-Tab Dashboard
        tab_timeline, tab_highlights, tab_compiler = st.tabs([
            "⏳ Full Timeline", 
            "⚡ Exciting Highlights", 
            "🎬 Highlight Reel Compiler"
        ])
        
        # --- TAB 1: FULL TIMELINE ---
        with tab_timeline:
            st.markdown("Review all segments recorded from the security livestream.")
            
            # Render a list of segments
            for idx, file_name in enumerate(analysis_files[:999]): # Limit to last 15 for readability
                data = None
                if config.get("gcs_enabled", False) and config.get("gcs_analysis_bucket"):
                    storage_client = get_storage_client()
                    if storage_client:
                        try:
                            bucket = storage_client.bucket(config["gcs_analysis_bucket"])
                            blob = bucket.blob(file_name)
                            content = blob.download_as_text()
                            data = json.loads(content)
                        except Exception:
                            pass
                else:
                    analysis_path = os.path.join(ANALYSIS_DIR, file_name)
                    try:
                        with open(analysis_path, 'r') as f:
                            data = json.load(f)
                    except Exception:
                        pass
                        
                if not data:
                    continue
                    
                video_file = data.get("video_file", "")
                status = data.get("status", "Unknown")
                timestamp = data.get("timestamp", "")
                analysis_text = data.get("analysis", "")
                is_hl = data.get("is_highlight", False)
                hl_reason = data.get("highlight_reason", "")
                
                # Prepare status labels
                if status == "Completed":
                    status_emoji = "⚡ ✅" if is_hl else "✅"
                elif status == "Analyzing":
                    status_emoji = "⏳"
                elif status == "Missing API Key":
                    status_emoji = "⚠️"
                else:
                    status_emoji = "❌"
                    
                display_time = get_segment_display_time(video_file)
                expander_header = f"{status_emoji} Segment {video_file.replace('chunk_', '').replace('.mp4', '')} ({display_time})"
                if is_hl:
                    expander_header += f" - [HIGHLIGHT: {hl_reason}]"
                
                with st.expander(expander_header, expanded=(idx == 0)):
                    st.markdown(f"**Status:** `{status}`")
                    if is_hl:
                        st.success(f"💡 **AI Highlight Reason:** {hl_reason}")
                    
                    if config.get("gcs_enabled", False) and config.get("gcs_chunks_bucket"):
                        storage_client = get_storage_client()
                        if storage_client:
                            try:
                                bucket = storage_client.bucket(config["gcs_chunks_bucket"])
                                blob = bucket.blob(video_file)
                                if blob.exists():
                                    # Generate v4 signed URL with 1 hour expiration
                                    signed_url = blob.generate_signed_url(
                                        version="v4",
                                        expiration=timedelta(hours=1),
                                        method="GET"
                                    )
                                    st.video(signed_url)
                                else:
                                    st.warning("Video chunk file not found in Cloud Storage.")
                            except Exception as e:
                                st.error(f"Could not generate signed URL for GCS video: {e}")
                        else:
                            st.error("GCS client initialization failed.")
                    else:
                        video_path = os.path.join(CHUNKS_DIR, video_file)
                        if os.path.exists(video_path):
                            try:
                                with open(video_path, 'rb') as vf:
                                    st.video(vf.read())
                            except Exception as ve:
                                st.error(f"Could not play video file: {ve}")
                        else:
                            st.warning("Video chunk file not found (might have been cleaned up).")
                    
                    st.markdown("### Gemini Security Analysis:")
                    if status == "Completed":
                        st.markdown(analysis_text)
                    elif status == "Missing API Key":
                        st.warning(analysis_text)
                    elif status == "Analyzing":
                        st.info("⚡ Gemini is currently processing this video clip. Please wait...")
                    else:
                        st.error(analysis_text)
            
            if not play_feed:
                st.button("🔄 Refresh Timeline", key="btn_refresh_timeline")
                
        # --- TAB 2: EXCITING HIGHLIGHTS (WITH 10S BUILD-UP) ---
        with tab_highlights:
            st.markdown("### AI Flagged Match Highlights (Preceding 10s Build-up + Moment)")
            
            # Filter highlights
            highlight_records = []
            for file_name in sorted(analysis_files): # chronological scan
                data = None
                if config.get("gcs_enabled", False) and config.get("gcs_analysis_bucket"):
                    storage_client = get_storage_client()
                    if storage_client:
                        try:
                            bucket = storage_client.bucket(config["gcs_analysis_bucket"])
                            blob = bucket.blob(file_name)
                            data = json.loads(blob.download_as_text())
                        except Exception:
                            pass
                else:
                    analysis_path = os.path.join(ANALYSIS_DIR, file_name)
                    try:
                        with open(analysis_path, 'r') as f:
                            data = json.load(f)
                    except Exception:
                        pass
                
                if data and data.get("status") == "Completed" and data.get("is_highlight", False):
                    highlight_records.append(data)
            
            if not highlight_records:
                st.info("No highlights flagged by Gemini yet. Adjust your source settings or prompts!")
            else:
                # Gather all mp4 chunks in chronological order to find preceding segments
                all_mp4_chunks = get_all_chunk_names(config)
                
                for idx, hl_data in enumerate(reversed(highlight_records)):
                    hl_video = hl_data.get("video_file", "")
                    hl_reason = hl_data.get("highlight_reason", "")
                    hl_timestamp = hl_data.get("timestamp", "")
                    
                    st.markdown(f"#### ⚡ Highlight Segment: {hl_video.replace('chunk_', '').replace('.mp4', '')}")
                    st.success(f"**AI Highlight Flag:** {hl_reason} ({hl_timestamp})")
                    
                    # Find the preceding video chunk (representing the preceding 10s build-up!)
                    preceding_video = None
                    if hl_video in all_mp4_chunks:
                        hl_index = all_mp4_chunks.index(hl_video)
                        if hl_index > 0:
                            preceding_video = all_mp4_chunks[hl_index - 1]
                            
                    hl_video_path = ensure_local_chunk(hl_video, config)
                    
                    if hl_video_path:
                        # Play Stitched Highlight (Build-up + Moment)
                        if preceding_video:
                            prec_video_path = ensure_local_chunk(preceding_video, config)
                            
                            if prec_video_path:
                                stitched_hl_filename = f"stitched_{preceding_video.replace('.mp4', '')}_{hl_video}"
                                stitched_hl_path = os.path.join(HIGHLIGHTS_DIR, stitched_hl_filename)
                                
                                # Stitch if not already cached on disk
                                if not os.path.exists(stitched_hl_path):
                                    stitch_videos([prec_video_path, hl_video_path], stitched_hl_path)
                                    
                                if os.path.exists(stitched_hl_path):
                                    try:
                                        with open(stitched_hl_path, 'rb') as vf:
                                            st.video(vf.read())
                                    except Exception as ve:
                                        st.error(f"Could not play stitched highlight: {ve}")
                                    st.caption("📹 Plays preceding 10 seconds of build-up footage leading directly into the highlight!")
                                else:
                                    try:
                                        with open(hl_video_path, 'rb') as vf:
                                            st.video(vf.read())
                                    except Exception as ve:
                                        st.error(f"Could not play video: {ve}")
                            else:
                                try:
                                    with open(hl_video_path, 'rb') as vf:
                                        st.video(vf.read())
                                except Exception as ve:
                                    st.error(f"Could not play video: {ve}")
                                st.caption("📹 Plays highlight moment (Preceding build-up chunk not available).")
                        else:
                            try:
                                with open(hl_video_path, 'rb') as vf:
                                    st.video(vf.read())
                            except Exception as ve:
                                st.error(f"Could not play video: {ve}")
                            st.caption("📹 Plays highlight moment (No preceding footage available).")
                    else:
                        st.warning("Highlight clip file not found (failed to download from GCS or missing locally).")
                    
                    st.markdown(f"**Analysis:** {hl_data.get('analysis', '')}")
                    st.markdown("---")
                    
        # --- TAB 3: HIGHLIGHT REEL COMPILER ---
        with tab_compiler:
            st.markdown("### 🎬 Highlight Reel Compiler")
            st.markdown("Select your favorite highlights below to compile them into a single custom video reel!")
            
            # Re-collect highlights for checklist
            compilable_highlights = []
            for file_name in sorted(analysis_files): # chronological order for sewing
                data = None
                if config.get("gcs_enabled", False) and config.get("gcs_analysis_bucket"):
                    storage_client = get_storage_client()
                    if storage_client:
                        try:
                            bucket = storage_client.bucket(config["gcs_analysis_bucket"])
                            blob = bucket.blob(file_name)
                            data = json.loads(blob.download_as_text())
                        except Exception:
                            pass
                else:
                    analysis_path = os.path.join(ANALYSIS_DIR, file_name)
                    try:
                        with open(analysis_path, 'r') as f:
                            data = json.load(f)
                    except Exception:
                        pass
                
                if data and data.get("status") == "Completed" and data.get("is_highlight", False):
                    compilable_highlights.append(data)
            
            if not compilable_highlights:
                st.info("No highlights available to compile yet.")
            else:
                # Checklist of highlights
                selected_highlights = []
                
                st.markdown("**Select Highlights to Include in Reel:**")
                for hl in compilable_highlights:
                    hl_file = hl.get("video_file", "")
                    hl_reason = hl.get("highlight_reason", "")
                    hl_time = hl.get("timestamp", "")
                    
                    # Checkbox state
                    cb_label = f"📹 {hl_file.replace('chunk_', '').replace('.mp4', '')} - {hl_reason} ({hl_time})"
                    is_checked = st.checkbox(cb_label, value=True, key=f"compiler_cb_{hl_file}")
                    if is_checked:
                        selected_highlights.append(hl)
                
                st.markdown("---")
                
                if not selected_highlights:
                    st.warning("Please select at least one highlight segment to compile.")
                else:
                    # Render "Stitch & Compile" button
                    st.markdown(f"Selected segments for Reel: `{len(selected_highlights)}` highlights (plus preceding build-ups).")
                    
                    if st.button("🎬 Compile Highlights into Master Reel", use_container_width=True):
                        with st.spinner("Processing and stitching video frames... Please wait..."):
                            # Build master sequential video list (preceding + highlight for each selection)
                            all_mp4_chunks = get_all_chunk_names(config)
                            
                            master_stitch_list = []
                            download_failed = False
                            
                            for hl in selected_highlights:
                                hl_file = hl.get("video_file", "")
                                hl_local_path = ensure_local_chunk(hl_file, config)
                                if not hl_local_path:
                                    st.error(f"Could not retrieve video clip {hl_file} from Cloud Storage.")
                                    download_failed = True
                                    break
                                    
                                # Preceding build-up
                                if hl_file in all_mp4_chunks:
                                    hl_idx = all_mp4_chunks.index(hl_file)
                                    if hl_idx > 0:
                                        prec_file = all_mp4_chunks[hl_idx - 1]
                                        prec_local_path = ensure_local_chunk(prec_file, config)
                                        if prec_local_path:
                                            master_stitch_list.append(prec_local_path)
                                        else:
                                            st.warning(f"Preceding build-up clip {prec_file} could not be retrieved; compiling highlight without it.")
                                            
                                master_stitch_list.append(hl_local_path)
                                
                            if download_failed:
                                success = False
                            else:
                                # Compile output file
                                timestamp_reel = datetime.now().strftime("%Y%m%d_%H%M%S")
                                reel_filename = f"highlight_reel_{timestamp_reel}.mp4"
                                output_reel_path = os.path.join(REELS_DIR, reel_filename)
                                
                                success = stitch_videos(master_stitch_list, output_reel_path)
                            
                            if success and os.path.exists(output_reel_path):
                                st.balloons()
                                st.success(f"🎉 Master Highlight Reel successfully compiled! Saved as {reel_filename}")
                                
                                # Play compiled reel
                                st.subheader("🍿 Watch Your Compiled Highlight Reel:")
                                try:
                                    with open(output_reel_path, 'rb') as vf:
                                        st.video(vf.read())
                                except Exception as ve:
                                    st.error(f"Could not play compiled reel: {ve}")
                                
                                # Provide local download button
                                with open(output_reel_path, "rb") as f:
                                    st.download_button(
                                        label="💾 Download Compiled Highlight Reel",
                                        data=f.read(),
                                        file_name=reel_filename,
                                        mime="video/mp4",
                                        use_container_width=True
                                    )
                            else:
                                st.error("Error: Video stitching failed. Verify that the segment video source files are still present in the chunks directory.")
