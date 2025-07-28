from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import base64
import time
import json
import numpy as np
import cv2
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import asyncio
from typing import List, Dict, Any
import logging

# Import BP Analyzer
from bp_analyzer import BPAnalyzer
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPG Signal Processing Server", version="1.0.0")

# Add CORS middleware for mobile app connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for WebSocket connections
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class PPGProcessor:
    def __init__(self):
        self.green_signal = []
        self.red_signal = []
        self.blue_signal = []
        self.timestamps = []
        self.frame_count = 0
        self.start_time = None
        
        # Signal processing parameters (HealthWatcher inspired)
        self.sampling_rate = 30  # Assuming 30 FPS
        self.window_size = 150   # 5 seconds at 30 FPS
        self.hr_freq_range = (0.7, 4.0)  # 42-240 BPM
        self.resp_freq_range = (0.1, 0.5)  # 6-30 breaths per minute
        
        # BP Analysis parameters
        self.bp_collection_duration = 30  # 30 seconds for BP analysis
        self.bp_collection_active = False
        self.bp_green_signal_buffer = []  # Dedicated buffer for BP analysis
        self.bp_start_time = None
        
        # Initialize BP Analyzer
        try:
            self.bp_analyzer = BPAnalyzer()
            logger.info("✅ BP Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BP Analyzer: {str(e)}")
            self.bp_analyzer = None
            
        # SMOOTHING: Moving average for stable heart rate
        from collections import deque
        self.hr_history = deque(maxlen=5)  # Last 5 heart rate readings
        self.last_stable_hr = None
        
    def reset(self):
        """Reset all signals for new measurement"""
        self.green_signal = []
        self.red_signal = []
        self.blue_signal = []
        self.timestamps = []
        self.frame_count = 0
        self.start_time = None
        
        # Reset BP analysis buffers
        self.bp_collection_active = False
        self.bp_green_signal_buffer = []
        self.bp_start_time = None
        
        # Reset smoothing history
        self.hr_history.clear()
        self.last_stable_hr = None
        logger.info("PPG processor reset (including BP analysis)")
    
    def start_bp_collection(self):
        """Start collecting PPG data for BP analysis"""
        self.bp_collection_active = True
        self.bp_green_signal_buffer = []
        self.bp_start_time = time.time()
        logger.info("🔴 Started BP analysis data collection (30 seconds)")
        
    def stop_bp_collection(self):
        """Stop BP data collection and return analysis results"""
        if not self.bp_collection_active:
            return {"error": "BP collection not active"}
            
        self.bp_collection_active = False
        collection_duration = time.time() - self.bp_start_time if self.bp_start_time else 0
        
        logger.info(f"🔵 Stopped BP collection after {collection_duration:.1f} seconds, {len(self.bp_green_signal_buffer)} samples")
        
        # Analyze collected data
        if len(self.bp_green_signal_buffer) >= 60:  # Minimum 2 seconds of data
            if self.bp_analyzer is not None:
                bp_result = self.bp_analyzer.predict_bp_category(self.bp_green_signal_buffer)
                bp_interpretation = self.bp_analyzer.get_bp_interpretation(bp_result.get("bp_category", "unknown"))
                
                result = {
                    "bp_analysis": bp_result,
                    "interpretation": bp_interpretation,
                    "collection_duration": collection_duration,
                    "samples_collected": len(self.bp_green_signal_buffer),
                    "status": "completed"
                }
                
                logger.info(f"🎯 BP Analysis completed: {bp_result.get('bp_category')} ({bp_result.get('confidence')}%)")
                return result
            else:
                return {"error": "BP Analyzer not available", "status": "error"}
        else:
            return {"error": "Insufficient data for BP analysis", "samples_collected": len(self.bp_green_signal_buffer), "status": "error"}
    
    def extract_rgb_from_frame(self, frame_data: bytes) -> Dict[str, float]:
        """Extract average RGB values from frame data"""
        try:
            # Decode base64 image
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"red": 0, "green": 0, "blue": 0, "error": "Failed to decode image"}
            
            # Convert BGR to RGB (OpenCV uses BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Calculate average RGB values for the entire frame
            # In a real implementation, you might want to focus on a specific ROI
            avg_red = np.mean(img_rgb[:, :, 0])
            avg_green = np.mean(img_rgb[:, :, 1])
            avg_blue = np.mean(img_rgb[:, :, 2])
            
            return {
                "red": float(avg_red),
                "green": float(avg_green), 
                "blue": float(avg_blue),
                "width": img.shape[1],
                "height": img.shape[0]
            }
            
        except Exception as e:
            logger.error(f"Error extracting RGB: {str(e)}")
            return {"red": 0, "green": 0, "blue": 0, "error": str(e)}
    
    def apply_bandpass_filter(self, data: List[float], low_freq: float, high_freq: float) -> np.ndarray:
        """Apply bandpass filter to signal"""
        try:
            if len(data) < 10:  # Need minimum data points
                return np.array(data)
            
            nyquist = self.sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Ensure frequencies are valid
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data)
            return filtered
            
        except Exception as e:
            logger.error(f"Error in bandpass filter: {str(e)}")
            return np.array(data)
    
    def calculate_heart_rate(self, green_signal: List[float]) -> Dict[str, Any]:
        """Calculate EXACT heart rate using time-domain peak detection (from real-time analyzer)"""
        try:
            if len(green_signal) < 60:  # Need at least 2 seconds at 30fps
                return {"heart_rate": 0, "confidence": 0, "method": "insufficient_data"}
            
            # Convert to numpy array
            signal_array = np.array(green_signal)
            
            # Remove DC component and normalize
            signal_array = signal_array - np.mean(signal_array)
            if np.std(signal_array) > 0:
                signal_array = signal_array / np.std(signal_array)
            
            # Apply bandpass filter for heart rate (0.5-4.0 Hz = 30-240 BPM)
            filtered_signal = self.apply_bandpass_filter(signal_array.tolist(), 0.5, 4.0)
            
            if len(filtered_signal) < 30:
                return {"heart_rate": 0, "confidence": 0, "method": "filter_failed"}
            
            # TIME DOMAIN APPROACH: Find peaks in the PPG signal
            from scipy.signal import find_peaks
            
            # Calculate adaptive threshold based on signal statistics
            signal_mean = np.mean(filtered_signal)
            signal_std = np.std(filtered_signal)
            threshold = signal_mean + 0.3 * signal_std
            
            # Find peaks with proper constraints
            # distance = minimum time between heartbeats (0.4s = 150 BPM max)
            min_distance = int(self.sampling_rate * 0.4)  
            prominence_threshold = signal_std * 0.2
            
            peaks, properties = find_peaks(filtered_signal, 
                                         height=threshold,
                                         distance=min_distance,
                                         prominence=prominence_threshold)
            
            # REQUIRE MINIMUM 5 PEAKS for stable measurement (ChatGPT suggestion)
            if len(peaks) < 5:
                logger.warning(f"Only {len(peaks)} peaks detected, need at least 5 for stable reading")
                return {"heart_rate": 0, "confidence": 0, "method": "insufficient_peaks_for_stability"}
            
            # Calculate heart rate from time intervals between peaks
            peak_intervals = np.diff(peaks) / self.sampling_rate  # Convert to seconds
            avg_interval = np.mean(peak_intervals)
            heart_rate = 60.0 / avg_interval  # Convert to BPM
            
            # Calculate heart rate variability for confidence
            interval_std = np.std(peak_intervals)
            interval_cv = interval_std / avg_interval  # Coefficient of variation
            
            # Validate heart rate range (realistic physiological limits)
            if heart_rate < 35 or heart_rate > 200:
                logger.warning(f"Heart rate {heart_rate:.1f} outside physiological range")
                return {"heart_rate": 0, "confidence": 0, "method": "invalid_range"}
            
            # SIGNAL QUALITY CHECK: Reject if too much noise
            signal_variance = np.var(filtered_signal)
            if signal_variance < 0.01:  # Too flat = poor signal
                logger.warning(f"Poor signal quality, variance too low: {signal_variance:.4f}")
                return {"heart_rate": 0, "confidence": 0, "method": "poor_signal_quality"}
            
            # SMOOTHING: Add to moving average window
            self.hr_history.append(heart_rate)
            
            if len(self.hr_history) >= 3:  # Need at least 3 readings
                # Calculate smoothed heart rate
                smoothed_hr = sum(self.hr_history) / len(self.hr_history)
                
                # Check if current reading is too different from trend (outlier detection)
                if len(self.hr_history) >= 4:
                    recent_avg = sum(list(self.hr_history)[-3:]) / 3
                    if abs(heart_rate - recent_avg) > 25:  # More than 25 BPM difference
                        logger.warning(f"Outlier detected: {heart_rate:.1f} vs recent avg {recent_avg:.1f}")
                        # Don't return this reading, use smoothed instead
                        final_hr = smoothed_hr
                    else:
                        final_hr = smoothed_hr
                else:
                    final_hr = smoothed_hr
            else:
                # Not enough history yet, use raw reading but be cautious
                final_hr = heart_rate
                smoothed_hr = heart_rate
            
            # Calculate confidence based on signal quality and regularity
            signal_to_noise = np.max(properties['prominences']) / (signal_std + 1e-10)
            regularity_score = max(0, 1 - interval_cv * 5)  # Lower CV = more regular
            
            # Boost confidence if smoothed reading is stable
            stability_bonus = 0
            if len(self.hr_history) >= 4:
                hr_std = np.std(list(self.hr_history))
                if hr_std < 5:  # Very stable readings
                    stability_bonus = 15
                elif hr_std < 10:  # Moderately stable
                    stability_bonus = 10
            
            # Combined confidence score
            confidence = min(95, int((signal_to_noise * 10 + regularity_score * 30 + 25 + stability_bonus)))
            confidence = max(15, confidence)
            
            # Determine signal quality
            if confidence > 80:
                quality = "excellent"
            elif confidence > 60:
                quality = "good"
            elif confidence > 40:
                quality = "fair"
            else:
                quality = "poor"
            
            # Return SMOOTHED heart rate for stability
            exact_heart_rate = round(final_hr, 1)
            self.last_stable_hr = exact_heart_rate
            
            logger.info(f"RAW HR: {heart_rate:.1f} BPM → SMOOTHED: {exact_heart_rate:.1f} BPM")
            logger.info(f"Detected {len(peaks)} peaks, avg interval: {avg_interval:.3f}s")
            logger.info(f"Confidence: {confidence}%, Quality: {quality}")
            logger.info(f"HR History: {[round(x, 1) for x in self.hr_history]}")
            
            return {
                "heart_rate": exact_heart_rate,
                "raw_heart_rate": round(heart_rate, 1),  # Also return raw for comparison
                "confidence": confidence,
                "method": "smoothed_time_domain_peak_detection", 
                "peaks_detected": len(peaks),
                "avg_interval_seconds": float(avg_interval),
                "heart_rate_variability": float(interval_cv),
                "signal_quality": quality,
                "signal_to_noise_ratio": float(signal_to_noise),
                "hr_history_size": len(self.hr_history),
                "signal_variance": float(signal_variance)
            }
            
        except Exception as e:
            logger.error(f"Error calculating heart rate: {str(e)}")
            return {"heart_rate": 0, "confidence": 0, "method": "error", "error": str(e)}
    
    def calculate_respiration_rate(self, red_signal: List[float]) -> Dict[str, Any]:
        """Calculate respiration rate from red channel variations"""
        try:
            if len(red_signal) < 90:  # Need at least 3 seconds
                return {"respiration_rate": 0, "confidence": 0}
            
            # Apply bandpass filter for respiration frequency range
            filtered_signal = self.apply_bandpass_filter(red_signal, 
                                                       self.resp_freq_range[0], 
                                                       self.resp_freq_range[1])
            
            # Perform FFT
            fft_result = fft(filtered_signal)
            freqs = fftfreq(len(filtered_signal), 1/self.sampling_rate)
            
            # Get magnitude and find peak in respiration range
            magnitude = np.abs(fft_result)
            
            # Find frequencies in respiration range
            resp_mask = (freqs >= self.resp_freq_range[0]) & (freqs <= self.resp_freq_range[1])
            if not np.any(resp_mask):
                return {"respiration_rate": 0, "confidence": 0}
            
            resp_freqs = freqs[resp_mask]
            resp_magnitudes = magnitude[resp_mask]
            
            # Find peak frequency
            peak_idx = np.argmax(resp_magnitudes)
            peak_freq = resp_freqs[peak_idx]
            
            # Convert to breaths per minute
            respiration_rate = int(peak_freq * 60)
            
            # Calculate confidence
            mean_magnitude = np.mean(resp_magnitudes)
            peak_magnitude = resp_magnitudes[peak_idx]
            confidence = min(100, int((peak_magnitude / mean_magnitude) * 15))
            
            return {
                "respiration_rate": respiration_rate,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating respiration rate: {str(e)}")
            return {"respiration_rate": 0, "confidence": 0}
    
    def calculate_spo2_estimate(self, red_signal: List[float], infrared_signal: List[float]) -> Dict[str, Any]:
        """Estimate SpO2 using AC/DC ratio method (simplified)"""
        try:
            if len(red_signal) < 30 or len(infrared_signal) < 30:
                return {"spo2": 0, "confidence": 0}
            
            # Calculate AC (standard deviation) and DC (mean) components
            red_ac = np.std(red_signal)
            red_dc = np.mean(red_signal)
            ir_ac = np.std(infrared_signal)
            ir_dc = np.mean(infrared_signal)
            
            if red_dc == 0 or ir_dc == 0:
                return {"spo2": 0, "confidence": 0}
            
            # Calculate ratio of ratios
            red_ratio = red_ac / red_dc
            ir_ratio = ir_ac / ir_dc
            
            if ir_ratio == 0:
                return {"spo2": 0, "confidence": 0}
            
            ratio = red_ratio / ir_ratio
            
            # Simplified SpO2 calculation (calibration needed for accuracy)
            spo2 = int(100 - (ratio * 25))  # Simplified formula
            spo2 = max(70, min(100, spo2))  # Clamp to reasonable range
            
            confidence = 60 if 85 <= spo2 <= 100 else 30
            
            return {
                "spo2": spo2,
                "confidence": confidence,
                "ratio": float(ratio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating SpO2: {str(e)}")
            return {"spo2": 0, "confidence": 0}
    
    def process_frame(self, frame_data: str, timestamp: float) -> Dict[str, Any]:
        """Process a single frame and update signals"""
        try:
            if self.start_time is None:
                self.start_time = timestamp
            
            # Decode base64 frame
            image_data = base64.b64decode(frame_data)
            rgb_values = self.extract_rgb_from_frame(image_data)
            
            if "error" in rgb_values:
                return {"status": "error", "error": rgb_values["error"]}
            
            # Add to signal buffers
            self.green_signal.append(rgb_values["green"])
            self.red_signal.append(rgb_values["red"])
            self.blue_signal.append(rgb_values["blue"])
            self.timestamps.append(timestamp)
            self.frame_count += 1
            
            # Add to BP collection buffer if active
            if self.bp_collection_active:
                self.bp_green_signal_buffer.append(rgb_values["green"])
                bp_elapsed_time = time.time() - self.bp_start_time if self.bp_start_time else 0
                
                # Auto-stop collection after 30 seconds
                if bp_elapsed_time >= self.bp_collection_duration:
                    bp_analysis_result = self.stop_bp_collection()
                    logger.info("🔵 Auto-stopped BP collection after 30 seconds")
                else:
                    bp_analysis_result = None
            else:
                bp_analysis_result = None
                bp_elapsed_time = 0
            
            # Maintain sliding window
            if len(self.green_signal) > self.window_size:
                self.green_signal.pop(0)
                self.red_signal.pop(0)
                self.blue_signal.pop(0)
                self.timestamps.pop(0)
            
            # Calculate vital signs if we have enough data
            results = {
                "status": "processing",
                "frame_count": self.frame_count,
                "elapsed_time": timestamp - self.start_time,
                "rgb_values": rgb_values,
                "buffer_size": len(self.green_signal),
                "green_signal_value": rgb_values["green"],  # Current green value for real-time display
                "bp_collection_active": self.bp_collection_active,
                "bp_collection_progress": {
                    "active": self.bp_collection_active,
                    "elapsed_time": bp_elapsed_time,
                    "samples_collected": len(self.bp_green_signal_buffer),
                    "progress_percentage": min(100, int((bp_elapsed_time / self.bp_collection_duration) * 100)) if self.bp_collection_active else 0
                }
            }
            
            # Add BP analysis result if completed
            if bp_analysis_result:
                results["bp_analysis_result"] = bp_analysis_result
            
            # Calculate heart rate every 15 frames (twice per second) for responsive display
            if len(self.green_signal) >= 60 and self.frame_count % 15 == 0:
                hr_result = self.calculate_heart_rate(self.green_signal)
                
                results.update({
                    "heart_rate": hr_result,
                    "green_signal_history": self.green_signal[-30:] if len(self.green_signal) >= 30 else self.green_signal,  # Last 1 second of data
                    "signal_processing": "active"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {"status": "error", "error": str(e)}

# Global PPG processor instance
ppg_processor = PPGProcessor()

@app.get("/")
async def root():
    return {"message": "PPG Signal Processing Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "server_time": time.time(),
        "processor_status": {
            "frame_count": ppg_processor.frame_count,
            "buffer_size": len(ppg_processor.green_signal),
            "is_active": ppg_processor.start_time is not None,
            "bp_analysis_available": ppg_processor.bp_analyzer is not None,
            "bp_collection_active": ppg_processor.bp_collection_active,
            "bp_samples_collected": len(ppg_processor.bp_green_signal_buffer)
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to WebSocket")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                # Parse incoming message
                message = json.loads(data)
                message_type = message.get("type", "frame")
                
                if message_type == "reset":
                    ppg_processor.reset()
                    response = {
                        "type": "reset_ack",
                        "status": "success",
                        "timestamp": time.time()
                    }
                elif message_type == "start_bp_analysis":
                    ppg_processor.start_bp_collection()
                    response = {
                        "type": "bp_analysis_started",
                        "status": "success",
                        "message": "Started BP analysis data collection for 30 seconds",
                        "duration": ppg_processor.bp_collection_duration,
                        "timestamp": time.time()
                    }
                
                elif message_type == "stop_bp_analysis":
                    bp_result = ppg_processor.stop_bp_collection()
                    response = {
                        "type": "bp_analysis_completed",
                        "status": "success",
                        "data": bp_result,
                        "timestamp": time.time()
                    }
                
                elif message_type == "frame":
                    frame_data = message.get("frame", "")
                    timestamp = message.get("timestamp", time.time())
                    
                    if not frame_data:
                        response = {
                            "type": "error",
                            "error": "No frame data provided",
                            "timestamp": time.time()
                        }
                    else:
                        # Process the frame
                        result = ppg_processor.process_frame(frame_data, timestamp)
                        response = {
                            "type": "result",
                            "data": result,
                            "timestamp": time.time()
                        }
                
                else:
                    response = {
                        "type": "error",
                        "error": f"Unknown message type: {message_type}",
                        "timestamp": time.time()
                    }
                
                await websocket.send_text(json.dumps(response))
                
            except json.JSONDecodeError as e:
                error_response = {
                    "type": "error",
                    "error": f"Invalid JSON: {str(e)}",
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(error_response))
                
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                error_response = {
                    "type": "error",
                    "error": f"Processing error: {str(e)}",
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(error_response))
    
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
