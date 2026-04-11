import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import logging
from collections import deque

logger = logging.getLogger(__name__)

class NuclearLSTMPredictor:
    """Real-time nuclear reactor fault detection predictor"""
    
    def __init__(self, model_path='advanced_nuclear_lstm_model.h5', 
                 config_path='model_config.json', 
                 scaler_path='feature_scaler.pkl'):
        """
        Initialize the predictor with trained model components
        """
        self.model = None
        self.scaler = None
        self.config = None
        self.feature_names = None
        self.buffer = None
        self.confidence_thresholds = {
            'high_confidence': 0.9,
            'medium_confidence': 0.7,
            'low_confidence': 0.5
        }
        self.class_names = ['Normal', 'Scram', 'LOFA']
        
        # Load model components
        self.load_model_components(model_path, config_path, scaler_path)
        
    def load_model_components(self, model_path, config_path, scaler_path):
        """Load all model components"""
        try:
            # Load model
            self.model = load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Load configuration
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.config = config_data['config']
                self.feature_names = config_data['feature_names']
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded")
            
            # Initialize buffer for real-time predictions
            self.buffer = deque(maxlen=self.config['window_size'])
            
            logger.info("Predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            raise
    
    def preprocess_single_sample(self, sample_data):
        """
        Preprocess a single sample for prediction
        
        Args:
            sample_data: dict or array-like with feature values
        
        Returns:
            preprocessed sample ready for model input
        """
        if isinstance(sample_data, dict):
            # Convert dict to array in correct order
            sample_array = np.array([sample_data[feature] for feature in self.feature_names])
        else:
            sample_array = np.array(sample_data)
        
        # Handle missing features by filling with zeros or interpolation
        if len(sample_array) < len(self.feature_names):
            padded_sample = np.zeros(len(self.feature_names))
            padded_sample[:len(sample_array)] = sample_array
            sample_array = padded_sample
        
        return sample_array
    
    def add_sample(self, sample_data):
        """
        Add a new sample to the buffer
        
        Args:
            sample_data: New sensor reading
        
        Returns:
            bool: True if buffer is ready for prediction
        """
        processed_sample = self.preprocess_single_sample(sample_data)
        self.buffer.append(processed_sample)
        
        return len(self.buffer) == self.config['window_size']
    
    def predict_current_state(self):
        """
        Predict the current reactor state based on buffer contents
        
        Returns:
            dict with prediction results
        """
        if len(self.buffer) < self.config['window_size']:
            return {
                'status': 'insufficient_data',
                'message': f"Need {self.config['window_size'] - len(self.buffer)} more samples"
            }
        
        try:
            # Convert buffer to sequence
            sequence = np.array(list(self.buffer))
            
            # Scale the sequence
            sequence_scaled = self._scale_sequence(sequence)
            
            # Reshape for model input
            X = sequence_scaled.reshape(1, self.config['window_size'], len(self.feature_names))
            
            # Make prediction
            prediction_prob = self.model.predict(X, verbose=0)[0]
            predicted_class = np.argmax(prediction_prob)
            confidence = np.max(prediction_prob)
            
            # Determine confidence level
            if confidence >= self.confidence_thresholds['high_confidence']:
                confidence_level = 'high'
            elif confidence >= self.confidence_thresholds['medium_confidence']:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
            
            # Risk assessment
            risk_level = self._assess_risk(predicted_class, confidence, prediction_prob)
            
            result = {
                'status': 'prediction_ready',
                'predicted_class': int(predicted_class),
                'predicted_state': self.class_names[predicted_class],
                'confidence': float(confidence),
                'confidence_level': confidence_level,
                'class_probabilities': {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(prediction_prob)
                },
                'risk_level': risk_level,
                'timestamp': pd.Timestamp.now().isoformat(),
                'recommendations': self._generate_recommendations(predicted_class, confidence, risk_level)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _scale_sequence(self, sequence):
        """Scale a sequence using the trained scaler"""
        sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
        sequence_scaled = self.scaler.transform(sequence_reshaped)
        return sequence_scaled.reshape(sequence.shape)
    
    def _assess_risk(self, predicted_class, confidence, probabilities):
        """
        Assess risk level based on prediction and confidence
        
        Returns:
            str: 'low', 'medium', 'high', 'critical'
        """
        # LOFA (Loss of Flow Accident) is most critical
        if predicted_class == 2:  # LOFA
            if confidence >= 0.8:
                return 'critical'
            elif confidence >= 0.6:
                return 'high'
            else:
                return 'medium'
        
        # Scram (Emergency shutdown)
        elif predicted_class == 1:  # Scram
            if confidence >= 0.8:
                return 'high'
            elif confidence >= 0.6:
                return 'medium'
            else:
                return 'low'
        
        # Normal operation
        else:  # Normal
            # Check if there's significant probability of fault states
            fault_probability = probabilities[1] + probabilities[2]  # Scram + LOFA
            
            if fault_probability > 0.3:
                return 'medium'  # Some uncertainty about fault states
            elif fault_probability > 0.1:
                return 'low'
            else:
                return 'low'
    
    def _generate_recommendations(self, predicted_class, confidence, risk_level):
        """Generate operational recommendations"""
        recommendations = []
        
        if predicted_class == 2:  # LOFA
            if confidence >= 0.8:
                recommendations.extend([
                    "IMMEDIATE ACTION REQUIRED: Potential Loss of Flow Accident detected",
                    "Verify coolant pump status and flow rates",
                    "Check for pump failures or blockages",
                    "Consider emergency cooling procedures",
                    "Alert operations team immediately"
                ])
            else:
                recommendations.extend([
                    "Monitor coolant flow rates closely",
                    "Check pump performance indicators",
                    "Verify flow sensor readings",
                    "Increase monitoring frequency"
                ])
        
        elif predicted_class == 1:  # Scram
            if confidence >= 0.8:
                recommendations.extend([
                    "Emergency shutdown sequence detected",
                    "Verify control rod positions",
                    "Monitor reactor power levels",
                    "Check shutdown systems status",
                    "Prepare for controlled shutdown procedures"
                ])
            else:
                recommendations.extend([
                    "Monitor control rod systems",
                    "Check for unusual power fluctuations",
                    "Verify automatic shutdown systems",
                    "Increase operational vigilance"
                ])
        
        else:  # Normal
            if risk_level == 'medium':
                recommendations.extend([
                    "System operating normally but monitor closely",
                    "Some uncertainty detected in fault state probabilities",
                    "Verify all sensor readings",
                    "Continue routine monitoring"
                ])
            else:
                recommendations.append("System operating within normal parameters")
        
        # General recommendations based on confidence
        if confidence < 0.7:
            recommendations.append("Low prediction confidence - verify sensor data quality")
        
        return recommendations
    
    def batch_predict(self, data_sequence):
        """
        Make predictions on a batch of sequential data
        
        Args:
            data_sequence: numpy array or DataFrame with sequential data
            
        Returns:
            list of prediction results
        """
        if isinstance(data_sequence, pd.DataFrame):
            data_sequence = data_sequence[self.feature_names].values
        
        results = []
        
        # Initialize buffer
        temp_buffer = deque(maxlen=self.config['window_size'])
        
        for i, sample in enumerate(data_sequence):
            temp_buffer.append(sample)
            
            if len(temp_buffer) == self.config['window_size']:
                # Make prediction
                sequence = np.array(list(temp_buffer))
                sequence_scaled = self._scale_sequence(sequence)
                X = sequence_scaled.reshape(1, self.config['window_size'], len(self.feature_names))
                
                prediction_prob = self.model.predict(X, verbose=0)[0]
                predicted_class = np.argmax(prediction_prob)
                confidence = np.max(prediction_prob)
                
                results.append({
                    'sample_index': i,
                    'predicted_class': int(predicted_class),
                    'predicted_state': self.class_names[predicted_class],
                    'confidence': float(confidence),
                    'probabilities': prediction_prob.tolist()
                })
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': 'Nuclear Reactor LSTM Fault Detector',
            'window_size': self.config['window_size'],
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'model_parameters': self.model.count_params(),
            'confidence_thresholds': self.confidence_thresholds
        }
    
    def reset_buffer(self):
        """Reset the prediction buffer"""
        self.buffer.clear()
        logger.info("Prediction buffer reset")

# Example usage and testing
def example_usage():
    """Example of how to use the predictor"""
    
    # Initialize predictor
    predictor = NuclearLSTMPredictor()
    
    # Get model info
    info = predictor.get_model_info()
    print("Model Info:", info)
    
    # Simulate real-time data feeding
    print("\nSimulating real-time predictions...")
    
    # Load some test data
    df = pd.read_csv('nuclear_fault_data_enhanced.csv')
    test_episode = df[df['Episode_ID'] == 0].head(50)  # First 50 samples of episode 0
    
    for idx, (_, row) in enumerate(test_episode.iterrows()):
        # Prepare sample data
        sample_data = {feature: row[feature] for feature in predictor.feature_names}
        
        # Add sample to buffer
        buffer_ready = predictor.add_sample(sample_data)
        
        if buffer_ready:
            # Make prediction
            result = predictor.predict_current_state()
            
            if result['status'] == 'prediction_ready':
                print(f"\nSample {idx + 1}:")
                print(f"Predicted State: {result['predicted_state']}")
                print(f"Confidence: {result['confidence']:.3f} ({result['confidence_level']})")
                print(f"Risk Level: {result['risk_level']}")
                
                if result['risk_level'] in ['high', 'critical']:
                    print("⚠️ ALERT: High risk detected!")
                    print("Recommendations:")
                    for rec in result['recommendations'][:3]:  # Show first 3 recommendations
                        print(f"  - {rec}")

if __name__ == "__main__":
    example_usage()