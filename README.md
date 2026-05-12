# 📹 Gemini Match Highlight Extractor & Video Compiler

A modular, robust Python application that simulates a real-time livestream or processes custom sports match uploads, automatically groups frames into 10-second segments, uses **Gemini 2.5 (Structured JSON Mode)** to identify highlights, and lets you compile your selected clips into a custom video highlight reel!

The frontend is built using **Streamlit**, presenting a live security feed/match stream alongside a tabbed highlights manager and a sequential video stitcher.

---

## 🏗️ Decoupled Architecture

To ensure robust performance and prevent Streamlit rerun mechanisms from interrupting video encoding, stitching, or API traffic, the app is built around a **directory-driven decoupled architecture**:

```
[ Livestream Simulator (Process 1) ]
             │
             ├─► Write current frame to livestream_data/current_frame.jpg
             └─► Write MP4 segments to livestream_data/chunks/chunk_<timestamp>.mp4
 
                                   ▲
                                   │ Read MP4 chunks
                                   │
[ Gemini Analyzer (Structured Output) ] ──► Analyze & Flag Highlights ("is_highlight": true)
                                           │ Write result to livestream_data/analysis/*.json
 
                                   ▲
                                   │ Read current_frame.jpg & completed analyses
                                   │
[ Streamlit Frontend (Process 3) ] ─► Stitch 10s build-up on-the-fly
                                   └─► Compile selected clips into a final highlight reel
```

- **`livestream_data/`**: Single source of truth for inter-process communication.
  - `current_frame.jpg`: Active rolling frame displayed on the frontend.
  - `chunks/`: Compiled H.264 `.mp4` video segments.
  - `analysis/`: Output structured JSON reports containing `is_highlight` and `highlight_reason` keys.
  - `highlights/`: Stitched video segments containing preceding build-up footage.
  - `reels/`: Final compiled master highlight reel video files.
  - `config.json`: Shared configuration parameters.

---

## 🚀 Quick Setup Guide

### Prerequisites
Make sure you have **Python 3.9+** installed on your system.

### 1. Install Dependencies
Install Streamlit, OpenCV, and the official Google GenAI SDK:
```bash
pip3 install streamlit opencv-python google-genai
```

### 2. Configure your Gemini API Key
You can launch the app directly and paste your API Key into the Streamlit sidebar. It will be safely saved inside the local `livestream_data/config.json` and automatically read by the analyzer.

Alternatively, you can set the standard environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

---

## 🏃 How to Run the Application

To run the full system, start the two background workers followed by the Streamlit web server:

```bash
# 1. Start the livestream simulator
python3 livestream_simulator.py &

# 2. Start the Gemini analyzer watcher
python3 gemini_analyzer.py &

# 3. Start the Streamlit web dashboard
python3 -m streamlit run app.py --server.port 8599
```

👉 Open your web browser and navigate to: **[http://localhost:8599](http://localhost:8599)**

---

## ⚙️ Key Features & Usage

### 🔴 Live Match Stream
- Renders a simulated 10 FPS match stream with overlays, or streams your own custom uploaded sports recording.
- Offers a **Play Live Stream** checkbox for continuous frame updates.

### 📁 Custom Video Match Uploads
Under **Livestream Source Type** in the sidebar, you can toggle to **Custom Video File Upload** to upload your own pre-recorded match (`.mp4`). The simulator loops your video and slices it into 10-second chunks for Gemini to analyze.

### ⚡ Structured AI Highlight Flags
Gemini is configured to return structured JSON outputs conforming to a strict Pydantic schema. It flags any scoring play, attempt, celebration, or exciting rally as `is_highlight = true` and provides a brief `highlight_reason` tagline.

### ⏳ Interactive Timeline & Highlights Manager (3 Tabs)
1. **⏳ Full Timeline**: View a chronological feed of all analyzed segments. Exciting highlight rows show a distinct `⚡` badge with their highlight reason.
2. **⚡ Exciting Highlights**: Lists only segments flagged as highlights. For each, the app locates the preceding 10-second chunk (build-up), stitches them sequentially on-the-fly, and lets you play the combined clip so you see the build-up play leading directly into the goal/point scored!
3. **🎬 Highlight Reel Compiler**: Displays a checklist of all detected highlights. Check the clips you want to keep, and click **Compile Selected Highlights into Master Reel** to stitch them sequentially (along with their build-ups) into a single compiled video file `highlight_reel_*.mp4` ready for download!
