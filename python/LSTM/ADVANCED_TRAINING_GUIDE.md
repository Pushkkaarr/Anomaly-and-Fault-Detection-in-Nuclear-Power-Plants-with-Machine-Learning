# Nuclear LSTM Fault Detection - Model Training Instructions

## Overview
This enhanced LSTM model provides robust fault detection for nuclear reactor safety with advanced features including:
- Bidirectional LSTM with attention mechanism
- Advanced feature engineering
- Robust data preprocessing
- Class imbalance handling
- Comprehensive evaluation metrics
- Real-time prediction capabilities

## Prerequisites

### Required Python Libraries
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn joblib
```

### Hardware Recommendations
- **Minimum**: 8GB RAM, Modern CPU
- **Recommended**: 16GB+ RAM, GPU support for faster training
- **Training Time**: 30-60 minutes on modern hardware

## Quick Start

### 1. Basic Training
Run the advanced model with default settings:
```bash
cd python/LSTM
python advanced_lstm_model.py
```

### 2. Custom Configuration Training
To customize model parameters, edit the config dictionary in `advanced_lstm_model.py`:

```python
config = {
    'window_size': 20,          # Temporal window size (10-30 recommended)
    'batch_size': 64,           # Training batch size (32-128)
    'epochs': 100,              # Maximum training epochs
    'learning_rate': 0.001,     # Learning rate (0.0001-0.01)
    'dropout_rate': 0.3,        # Dropout rate for regularization
    'lstm_units': [128, 64, 32], # LSTM layer sizes
    'use_bidirectional': True,   # Use bidirectional LSTM
    'use_attention': True,       # Use attention mechanism
    'scaler_type': 'robust'      # Scaling method (robust/minmax/standard)
}
```

## Model Architecture Features

### Enhanced Preprocessing
- **Outlier Handling**: IQR-based outlier capping
- **Feature Engineering**: Moving averages, standard deviations, temperature differentials
- **Missing Value Handling**: Forward/backward fill within episodes
- **Robust Scaling**: Less sensitive to outliers than standard scaling

### Advanced Architecture
- **Bidirectional LSTM**: Processes sequences in both directions
- **Attention Mechanism**: Focuses on most relevant time steps
- **Batch Normalization**: Improves training stability
- **Regularization**: L1/L2 regularization + dropout
- **Class Weighting**: Handles imbalanced data

### Training Improvements
- **Stratified Splitting**: Maintains class distribution across splits
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best model during training

## Expected Performance

### Target Metrics
- **Overall Accuracy**: >95%
- **LOFA Detection**: >98% (most critical)
- **Scram Detection**: >90%
- **Normal Operation**: >95%
- **False Positive Rate**: <2%

### Confidence Analysis
The model provides confidence scores for each prediction:
- **High Confidence**: >0.9 (Act on prediction)
- **Medium Confidence**: 0.7-0.9 (Monitor closely)
- **Low Confidence**: <0.7 (Verify sensors)

## Real-Time Usage

### Loading Trained Model
```python
from real_time_predictor import NuclearLSTMPredictor

# Initialize predictor
predictor = NuclearLSTMPredictor(
    model_path='advanced_nuclear_lstm_model.h5',
    config_path='model_config.json', 
    scaler_path='feature_scaler.pkl'
)
```

### Real-Time Prediction
```python
# Add sensor readings one by one
sample_data = {
    'Power': 1.0,
    'Fuel_Temp': 1090.0,
    'Coolant_Temp': 290.0,
    'Pressure': 9.6,
    'Flow': 8000.0,
    'Power_ROC': 0.01,
    'Temp_Fuel_ROC': 0.5,
    'Temp_Coolant_ROC': 0.5,
    'Flow_ROC': 10.0
    # ... additional engineered features
}

# Add to buffer
buffer_ready = predictor.add_sample(sample_data)

if buffer_ready:
    result = predictor.predict_current_state()
    
    if result['predicted_state'] == 'LOFA':
        print("⚠️ CRITICAL: Loss of Flow Accident detected!")
        print(f"Confidence: {result['confidence']:.3f}")
        for rec in result['recommendations']:
            print(f"- {rec}")
```

## File Outputs

After training, the following files are created:

1. **advanced_nuclear_lstm_model.h5** - Complete trained model
2. **best_nuclear_lstm_model.h5** - Best model during training
3. **model_config.json** - Model configuration and metadata
4. **feature_scaler.pkl** - Trained feature scaler
5. **nuclear_lstm_results.png** - Training visualization plots

## Troubleshooting

### Common Issues

#### Low Accuracy (<90%)
- **Cause**: Insufficient training data or poor feature quality
- **Solution**: 
  - Increase `window_size` (20-30)
  - Enable `use_attention=True`
  - Increase `epochs` (100-200)

#### Training Takes Too Long
- **Cause**: Large model or insufficient hardware
- **Solution**:
  - Reduce `lstm_units` sizes
  - Decrease `batch_size`
  - Set `use_bidirectional=False`

#### High False Positives
- **Cause**: Model too sensitive
- **Solution**:
  - Increase `dropout_rate` (0.4-0.5)
  - Add more regularization
  - Tune confidence thresholds

#### Memory Errors
- **Cause**: Insufficient RAM
- **Solution**:
  - Reduce `batch_size`
  - Reduce `window_size`
  - Use gradient checkpointing

### Performance Optimization

#### For Speed
```python
config = {
    'window_size': 10,
    'batch_size': 128,
    'lstm_units': [64, 32],
    'use_bidirectional': False,
    'use_attention': False
}
```

#### For Accuracy
```python
config = {
    'window_size': 25,
    'batch_size': 32,
    'lstm_units': [256, 128, 64],
    'use_bidirectional': True,
    'use_attention': True,
    'epochs': 150
}
```

## Model Validation

### Cross-Validation
The model uses episode-based stratified splitting to ensure:
- No data leakage between train/test sets
- Balanced class distribution
- Realistic performance estimates

### Key Metrics to Monitor
1. **Precision for LOFA**: Should be >95% (minimize false alarms)
2. **Recall for LOFA**: Should be >98% (catch all incidents)
3. **Overall F1-Score**: Should be >90%
4. **Confidence Distribution**: Should show clear separation between classes

## Integration Guidelines

### Safety Systems Integration
- **High Confidence LOFA**: Trigger automatic safety systems
- **Medium Confidence Fault**: Alert operators for manual verification
- **Low Confidence**: Log for analysis but don't trigger alarms

### Monitoring Dashboard
Create real-time displays showing:
- Current prediction with confidence
- Trend of predictions over time
- Sensor data quality indicators
- Model performance metrics

## Maintenance

### Model Retraining
Retrain monthly or when:
- New fault patterns are discovered
- Performance degrades below thresholds
- Plant configuration changes
- New sensor data becomes available

### Performance Monitoring
Track these metrics in production:
- Prediction accuracy vs actual events
- Confidence score distributions
- Response time for predictions
- False positive/negative rates

## Advanced Features

### Uncertainty Quantification
The model provides:
- Prediction confidence scores
- Per-class probability distributions
- Risk level assessments
- Actionable recommendations

### Explainability
- Feature importance analysis
- Attention weight visualization
- Temporal pattern identification
- Anomaly detection capabilities

---

For technical support or questions about the model implementation, refer to the detailed code comments in `advanced_lstm_model.py` and `real_time_predictor.py`.