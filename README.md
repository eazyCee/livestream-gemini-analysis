# 📹 Gemini Match Highlight Extractor & Video Compiler

A modular, robust Python application that processes live match video feeds, compiles them into short segments, triggers event-driven **Gemini 2.5 (Structured Output)** video analysis in the cloud, and compiles selected clips into custom stitched video highlight reels.

The application supports both a **Hybrid Event-Driven Cloud Mode** (backed by Google Cloud Storage and Cloud Functions) and a **100% Offline Local Fallback Mode** for rapid testing and local development.

---

## 🏗️ Hybrid Event-Driven Architecture

This system can be run in two architectures, switched instantly by a checkbox in the Streamlit sidebar:

### Architecture A: Hybrid Cloud Mode (Recommended)

Highly scalable, production-ready event-driven flow. Heavy-lifting video processing is delegated to cloud-native event triggers.

```
[ Local Machine ]
  ├── [ Livestream Simulator ] ──► Write current_frame.jpg (Local view)
  │         └── Compile MP4 ──► [ Local disk cache ]
  │                                   │
  │                         (Background Thread Queue)
  │                                   ▼
  │                          [ GCS Chunks Bucket ]
  │                                   │
                               (GCS Finalize Event)
                                      ▼
[ Google Cloud Platform ]    [ GCP Cloud Function (Gen 2) ]
                                      │
                             (Standard GenAI SDK)
                                      ▼
                               [ Gemini API ] ──► Flag highlights & describe play
                                      │
                               (JSON Response)
                                      ▼
                             [ GCS Analysis Bucket ]
                                      ▲
                                      │
[ Local/Cloud UI ]           [ Streamlit Dashboard ]
                                      ├── Reads JSON analysis from GCS bucket
                                      ├── Streams video chunks via GCS Signed URLs (v4)
                                      └── Downloads chunks on-demand to cache for local stitching
```

### Architecture B: Local Fallback Mode (Offline)

Fully local, directory-driven multi-process architecture. Perfect for running completely offline with standard Gemini API keys.

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

---

## 🚀 Setup Guide

### 1. Install Dependencies
Install Streamlit, OpenCV, GCS client, and the official Google GenAI SDK:
```bash
pip3 install streamlit opencv-python google-genai google-cloud-storage
```

### 2. Retrieve Project Number & Create GCS Buckets
If using the Cloud Mode, fetch your active GCP project number and create the GCS buckets using `gcloud storage`:
```bash
# Fetch your active GCP project number
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")

# Create GCS buckets using project number as suffix
gcloud storage buckets create gs://livestream-chunks-${PROJECT_NUMBER} --location=us-central1
gcloud storage buckets create gs://livestream-analysis-${PROJECT_NUMBER} --location=us-central1
```
*(You can change `--location` to a region close to you, e.g., `us-central1`.)*

### 3. Configure Local Credentials
Configure your local environment to use your GCP credentials:
```bash
gcloud auth application-default login
```

---

## ☁️ Deploying the Cloud Function

To handle video chunk analysis in the cloud, deploy the event-driven Gen 2 Cloud Function located in the `gcp_cloud_function/` folder:

```bash
# Ensure PROJECT_NUMBER variable is set
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")

# Deploy the event-driven Gen 2 Cloud Function
gcloud functions deploy analyze_gcs_chunk \
    --gen2 \
    --runtime=python310 \
    --region=us-central1 \
    --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
    --trigger-event-filters="bucket=livestream-chunks-${PROJECT_NUMBER}" \
    --entry-point=analyze_gcs_chunk \
    --source=./gcp_cloud_function \
    --set-env-vars="GEMINI_API_KEY=YOUR_GEMINI_API_KEY,GCS_ANALYSIS_BUCKET=livestream-analysis-${PROJECT_NUMBER},GEMINI_MODEL=gemini-2.5-flash" \
    --memory=512Mi
```

---

## 🏃 How to Run the Application

### Running Mode A: Hybrid Cloud Mode (Recommended)

In Cloud Mode, the local simulator compiles video chunks, safely queues them in a thread-safe background uploader, and uploads them to GCS. The Cloud Function analyzes them in GCP, and the Streamlit dashboard displays and streams them directly from GCS via secure signed URLs.

1. Start the livestream simulator locally:
   ```bash
   python3 livestream_simulator.py &
   ```
2. Start the Streamlit dashboard:
   ```bash
   python3 -m streamlit run app.py --server.port 8599
   ```
3. Open **[http://localhost:8599](http://localhost:8599)**, expand the **Google Cloud Storage (GCS)** section in the sidebar, tick **Enable GCS Integration**, and input your GCS bucket details.

---

### Running Mode B: Local Fallback Mode (Offline)

1. Start the livestream simulator:
   ```bash
   python3 livestream_simulator.py &
   ```
2. Start the local Gemini analyzer watcher:
   ```bash
   python3 gemini_analyzer.py &
   ```
3. Start the Streamlit web dashboard:
   ```bash
   python3 -m streamlit run app.py --server.port 8599
   ```
4. Open **[http://localhost:8599](http://localhost:8599)** and ensure **Enable GCS Integration** is unchecked in the sidebar.

---

## ⚙️ Key Features

- **Zero Stream Loss**: The simulator uploads segments using a background thread-safe daemon queue. If the network drops, chunks are safely cached locally and retried indefinitely in chronological order once the connection is restored.
- **Signed URL Direct Video Streaming**: Videos in the Streamlit timeline stream directly from GCS using secure v4 signed URLs, eliminating heavy file-download workloads on the Streamlit application server.
- **On-Demand Stitching Cache**: Video chunks are downloaded on-the-fly and cached in a local folder only when highlight stitching or compilation is requested.
