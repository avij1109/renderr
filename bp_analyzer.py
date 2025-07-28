"""
Blood Pressure Analysis Module
Integrates machine learning model for real-time BP classification
"""

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, find_peaks
import joblib
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class BPAnalyzer:
    def __init__(self, model_path: str = "bp_classification_model_enhanced.pkl", 
                 encoder_path: str = "label_encoder_enhanced.pkl"):
        """
        Initialize BP Analyzer with trained model
        
        Args:
            model_path: Path to trained Random Forest model
            encoder_path: Path to label encoder
        """
        self.model = None
        self.label_encoder = None
        self.fs = 30  # Sampling rate (frames per second)
        
        # Load trained model and encoder
        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            logger.info(f"✅ BP model loaded successfully from {model_path}")
            logger.info(f"✅ Label encoder loaded from {encoder_path}")
            logger.info(f"📊 Model classes: {self.label_encoder.classes_}")
        except Exception as e:
            logger.error(f"❌ Error loading BP model: {str(e)}")
            raise

    def remove_spike_outliers(self, signal: np.ndarray, threshold: float = 3) -> np.ndarray:
        """Remove spike outliers from signal using z-score method"""
        try:
            signal = np.array(signal, dtype=float)
            z = np.abs((signal - np.mean(signal)) / np.std(signal))
            signal[z > threshold] = np.median(signal)
            return signal
        except Exception as e:
            logger.error(f"Error in spike outlier removal: {str(e)}")
            return signal

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to 0-1 range"""
        try:
            signal = np.array(signal, dtype=float)
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val - min_val == 0:
                return signal
            return (signal - min_val) / (max_val - min_val)
        except Exception as e:
            logger.error(f"Error in signal normalization: {str(e)}")
            return signal

    def wavelet_denoise(self, signal: np.ndarray, wavelet: str = 'db6', level: int = 3) -> np.ndarray:
        """Apply wavelet denoising to signal"""
        try:
            signal = np.array(signal, dtype=float)
            if len(signal) < 2**level:
                logger.warning("Signal too short for wavelet denoising, returning original")
                return signal
                
            coeff = pywt.wavedec(signal, wavelet, level=level)
            sigma = np.median(np.abs(coeff[-level])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
            coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
            return pywt.waverec(coeff, wavelet)
        except Exception as e:
            logger.error(f"Error in wavelet denoising: {str(e)}")
            return signal

    def bandpass_filter(self, signal: np.ndarray, low: float = 0.5, high: float = 8.0, 
                       fs: int = 30, order: int = 4) -> np.ndarray:
        """Apply bandpass filter to signal"""
        try:
            signal = np.array(signal, dtype=float)
            if len(signal) < 3 * order:
                logger.warning("Signal too short for filtering, returning original")
                return signal
                
            nyq = 0.5 * fs
            low_norm = low / nyq
            high_norm = high / nyq
            
            # Ensure valid frequency ranges
            low_norm = max(0.01, min(low_norm, 0.99))
            high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
            
            b, a = butter(order, [low_norm, high_norm], btype='band')
            return filtfilt(b, a, signal)
        except Exception as e:
            logger.error(f"Error in bandpass filtering: {str(e)}")
            return signal

    def preprocess_ppg_signal(self, ppg_signal: List[float]) -> np.ndarray:
        """
        Full preprocessing pipeline for PPG signal
        
        Args:
            ppg_signal: Raw PPG signal data
            
        Returns:
            Preprocessed PPG signal
        """
        try:
            # Convert to numpy array
            signal = np.array(ppg_signal, dtype=float)
            
            # Step 1: Remove spike outliers
            signal = self.remove_spike_outliers(signal)
            
            # Step 2: Normalize signal
            signal = self.normalize_signal(signal)
            
            # Step 3: Wavelet denoising
            signal = self.wavelet_denoise(signal)
            
            # Step 4: Bandpass filtering
            signal = self.bandpass_filter(signal, fs=self.fs)
            
            logger.info(f"✅ Preprocessed PPG signal: {len(signal)} samples")
            return signal
            
        except Exception as e:
            logger.error(f"Error in PPG preprocessing: {str(e)}")
            return np.array(ppg_signal, dtype=float)

    def extract_features(self, ppg_segment: np.ndarray) -> Dict[str, float]:
        """
        Extract features from PPG segment for BP classification
        
        Args:
            ppg_segment: Preprocessed PPG signal segment
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            # Statistical features
            features['mean'] = np.mean(ppg_segment)
            features['std'] = np.std(ppg_segment)
            features['kurtosis'] = np.mean((ppg_segment - np.mean(ppg_segment))**4) / (np.std(ppg_segment)**4)
            features['skewness'] = np.mean((ppg_segment - np.mean(ppg_segment))**3) / (np.std(ppg_segment)**3)
            
            # Peak detection for physiological features
            peaks, _ = find_peaks(ppg_segment, distance=self.fs*0.6)  # Minimum 0.6s between peaks
            peak_vals = ppg_segment[peaks]
            
            if len(peaks) >= 2:
                # Systolic peak height
                features['systolic_peak_height'] = np.mean(peak_vals)
                
                # Heart rate calculation
                rr_intervals = np.diff(peaks) / self.fs
                hr_values = 60 / rr_intervals
                features['mean_hr'] = np.mean(hr_values)
                
                # Heart rate variability
                features['hrv_sdnn'] = np.std(rr_intervals)
                
                # Notch to peak timing
                notches = []
                for i in range(len(peaks) - 1):
                    start = peaks[i]
                    end = peaks[i+1]
                    notch_index = start + np.argmin(ppg_segment[start:end])
                    time_to_peak = (peaks[i+1] - notch_index) / self.fs
                    notches.append(time_to_peak)
                features['notch_to_peak_time'] = np.mean(notches)
            else:
                # Default values when insufficient peaks
                features['systolic_peak_height'] = 0
                features['mean_hr'] = 0
                features['hrv_sdnn'] = 0
                features['notch_to_peak_time'] = 0
            
            logger.info(f"✅ Extracted {len(features)} features from PPG segment")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            # Return default features on error
            return {
                'mean': 0, 'std': 0, 'kurtosis': 0, 'skewness': 0,
                'systolic_peak_height': 0, 'mean_hr': 0, 'hrv_sdnn': 0, 'notch_to_peak_time': 0
            }

    def predict_bp_category(self, ppg_signal: List[float]) -> Dict[str, Any]:
        """
        Predict blood pressure category from PPG signal
        
        Args:
            ppg_signal: Raw PPG signal (30 seconds worth)
            
        Returns:
            BP prediction results with confidence and details
        """
        try:
            if self.model is None or self.label_encoder is None:
                return {
                    "bp_category": "unknown",
                    "confidence": 0,
                    "error": "Model not loaded",
                    "status": "error"
                }
            
            # Preprocess the signal
            processed_signal = self.preprocess_ppg_signal(ppg_signal)
            
            # Extract features
            features = self.extract_features(processed_signal)
            
            # Prepare feature vector for prediction
            feature_vector = np.array([
                features['mean'], features['std'], features['kurtosis'], features['skewness'],
                features['systolic_peak_height'], features['mean_hr'], 
                features['hrv_sdnn'], features['notch_to_peak_time']
            ]).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(feature_vector)[0]
            prediction_proba = self.model.predict_proba(feature_vector)[0]
            
            # Get category name
            bp_category = self.label_encoder.inverse_transform([prediction])[0]
            
            # Calculate confidence (max probability * 100)
            confidence = int(np.max(prediction_proba) * 100)
            
            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_probabilities[class_name] = float(prediction_proba[i])
            
            result = {
                "bp_category": bp_category,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "features_extracted": features,
                "signal_length": len(ppg_signal),
                "processed_length": len(processed_signal),
                "status": "success"
            }
            
            logger.info(f"🎯 BP Prediction: {bp_category} (confidence: {confidence}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error in BP prediction: {str(e)}")
            return {
                "bp_category": "unknown",
                "confidence": 0,
                "error": str(e),
                "status": "error"
            }

    def get_bp_interpretation(self, bp_category: str) -> Dict[str, str]:
        """
        Get interpretation and recommendations for BP category
        
        Args:
            bp_category: Predicted BP category
            
        Returns:
            Interpretation and recommendations
        """
        interpretations = {
            "normotensive": {
                "interpretation": "Normal Blood Pressure",
                "description": "Your blood pressure is in the normal range",
                "recommendation": "Maintain healthy lifestyle habits",
                "risk_level": "Low"
            },
            "prehypertensive": {
                "interpretation": "Prehypertension", 
                "description": "Your blood pressure is elevated but not yet hypertensive",
                "recommendation": "Consider lifestyle modifications and regular monitoring",
                "risk_level": "Moderate"
            },
            "hypertensive": {
                "interpretation": "Hypertension",
                "description": "Your blood pressure is in the high range",
                "recommendation": "Consult healthcare provider for proper evaluation and treatment",
                "risk_level": "High"
            }
        }
        
        return interpretations.get(bp_category, {
            "interpretation": "Unknown",
            "description": "Unable to determine blood pressure category",
            "recommendation": "Please try again or consult healthcare provider",
            "risk_level": "Unknown"
        })
