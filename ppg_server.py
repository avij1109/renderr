"""
PPG Health Monitoring Server - Simplified without MongoDB
Real-time heart rate and blood pressure analysis via WebSocket
"""
import asyncio
import json
import logging
import numpy as np
import cv2
import base64
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List
import uvicorn
from bp_analyzer import BPAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="PPG Health Monitor", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize BP Analyzer
bp_analyzer = BPAnalyzer()

# Global variables for active sessions
active_connections: Dict[WebSocket, dict] = {}

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 PPG Health Server starting up...")
    logger.info("✅ Server ready - MongoDB removed for simplicity")

@app.get("/")
async def root():
    return {"message": "PPG Health Monitor API", "status": "running", "version": "2.0-simple"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

class PPGProcessor:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.ppg_values = []
        self.heart_rates = []
        self.frame_count = 0
        self.bp_collection_active = False
        self.bp_collection_start_time = None
        self.bp_frames = []
        
    def process_frame(self, frame_data):
        try:
            # Decode base64 frame
            frame_bytes = base64.b64decode(frame_data)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None
                
            # Extract PPG signal (average red channel intensity)
            red_channel = frame[:, :, 2]  # Red channel for PPG
            ppg_value = float(np.mean(red_channel))
            
            self.ppg_values.append(ppg_value)
            self.frame_count += 1
            
            # Auto-start BP collection on first frame
            if self.frame_count == 1 and not self.bp_collection_active:
                self.bp_collection_active = True
                self.bp_collection_start_time = datetime.now()
                self.bp_frames = []
                logger.info("🔴 Auto-started BP analysis with first PPG frame")
            
            # Collect frames for BP analysis
            if self.bp_collection_active:
                self.bp_frames.append(ppg_value)
                elapsed = (datetime.now() - self.bp_collection_start_time).total_seconds()
                
                # Log progress every 5 seconds
                if int(elapsed) % 5 == 0 and len(self.bp_frames) % 150 == 0:
                    logger.info(f"🔵 BP Collection Progress: {elapsed:.1f}s, {len(self.bp_frames)} samples")
                
                # Auto-stop after 30 seconds and analyze
                if elapsed >= 30:
                    self.bp_collection_active = False
                    logger.info("🔵 Auto-stopped BP collection after 30 seconds")
                    return self.analyze_bp()
            
            # Calculate heart rate from recent PPG values
            if len(self.ppg_values) >= 100:  # Need sufficient data
                recent_values = self.ppg_values[-300:]  # Last 10 seconds at 30fps
                hr_result = self.calculate_heart_rate(recent_values)
                return hr_result
                
            return None
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def calculate_heart_rate(self, ppg_values):
        try:
            # Apply bandpass filter for heart rate range (0.5-3 Hz)
            from scipy.signal import butter, filtfilt, find_peaks
            
            fs = 30  # Sampling frequency (30 fps)
            lowcut, highcut = 0.5, 3.0
            nyquist = fs / 2
            low, high = lowcut / nyquist, highcut / nyquist
            
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, ppg_values)
            
            # Find peaks
            peaks, _ = find_peaks(filtered_signal, distance=fs//3, prominence=np.std(filtered_signal)*0.3)
            
            if len(peaks) < 5:
                logger.warning(f"Only {len(peaks)} peaks detected, need at least 5 for stable reading")
                return None
            
            # Calculate heart rate
            intervals = np.diff(peaks) / fs
            avg_interval = np.mean(intervals)
            raw_hr = 60 / avg_interval
            
            # Smooth heart rate
            self.heart_rates.append(raw_hr)
            if len(self.heart_rates) > 5:
                self.heart_rates = self.heart_rates[-5:]
            
            smoothed_hr = np.mean(self.heart_rates)
            confidence = min(95, len(peaks) * 15)
            
            # Quality assessment
            quality = "excellent" if confidence >= 85 else "good" if confidence >= 70 else "fair"
            
            logger.info(f"RAW HR: {raw_hr:.1f} BPM → SMOOTHED: {smoothed_hr:.1f} BPM")
            logger.info(f"Detected {len(peaks)} peaks, avg interval: {avg_interval:.3f}s")
            logger.info(f"Confidence: {confidence}%, Quality: {quality}")
            
            return {
                "heart_rate": int(smoothed_hr),
                "confidence": confidence,
                "signal_quality": quality,
                "raw_hr": raw_hr,
                "peak_count": len(peaks),
                "elapsed_time": len(self.ppg_values) / 30
            }
            
        except Exception as e:
            logger.error(f"Error calculating heart rate: {e}")
            return None
    
    def analyze_bp(self):
        try:
            if len(self.bp_frames) < 300:  # Need at least 10 seconds
                logger.warning(f"Insufficient data for BP analysis: {len(self.bp_frames)} frames")
                return None
            
            logger.info(f"🔵 Analyzing BP with {len(self.bp_frames)} frames...")
            
            # Use BP analyzer
            result = bp_analyzer.analyze_bp_from_ppg(self.bp_frames)
            
            if result:
                logger.info(f"🎯 BP Analysis completed: {result['category']} ({result['confidence']}%)")
                return {
                    "bp_analysis_result": result,
                    "analysis_complete": True,
                    "elapsed_time": 40  # Mark as complete
                }
            else:
                logger.warning("⚠️ BP analysis failed")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing BP: {e}")
            return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to WebSocket")
    
    # Initialize PPG processor for this connection
    processor = PPGProcessor()
    active_connections[websocket] = {"processor": processor}
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                logger.info(f"Received message type: {message.get('type', 'unknown')}")
                
                if message.get("type") == "frame":
                    # Check if frame data exists in different possible fields
                    frame_data = message.get("data") or message.get("frame_data") or message.get("image")
                    
                    if frame_data:
                        # Process the frame
                        result = processor.process_frame(frame_data)
                        
                        if result:
                            # Send result back to client
                            await websocket.send_text(json.dumps(result))
                    else:
                        logger.warning("No frame data found in message")
                        
                elif message.get("type") == "reset":
                    # Reset processor for new measurement
                    processor.reset()
                    logger.info("🔄 PPG processor reset for new measurement")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
            except KeyError as e:
                logger.error(f"Missing key in message: {e}")
                logger.error(f"Message keys: {list(message.keys()) if 'message' in locals() else 'No message'}")
                
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket")
        if websocket in active_connections:
            del active_connections[websocket]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            del active_connections[websocket]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
