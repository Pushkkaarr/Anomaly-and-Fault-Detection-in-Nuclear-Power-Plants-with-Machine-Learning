import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Attention, Input, Concatenate, Softmax, Multiply, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import json
import logging
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NuclearLSTMModel:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.model = None
        self.scaler = None
        self.history = None
        self.class_weights = None
        self.feature_names = ['Power', 'Fuel_Temp', 'Coolant_Temp', 'Pressure', 'Flow', 
                             'Power_ROC', 'Temp_Fuel_ROC', 'Temp_Coolant_ROC', 'Flow_ROC']
        
    def _default_config(self):
        return {
            'window_size': 15,  # Increased for better temporal patterns
            'batch_size': 64,
            'epochs': 100,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'l1_reg': 0.01,
            'l2_reg': 0.01,
            'lstm_units': [128, 64, 32],  # Multi-layer LSTM
            'dense_units': [64, 32],
            'patience': 15,
            'validation_split': 0.2,
            'use_bidirectional': True,
            'use_attention': True,
            'scaler_type': 'robust'  # More robust to outliers
        }
    
    def load_and_preprocess_data(self, file_path):
        """Enhanced data loading with robust preprocessing"""
        logger.info("Loading and preprocessing data...")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded: {df.shape}")
            
            # Data validation
            self._validate_data(df)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Feature engineering
            df = self._engineer_features(df)
            
            # Outlier detection and handling
            df = self._handle_outliers(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_data(self, df):
        """Validate data quality"""
        required_cols = self.feature_names + ['Label', 'Episode_ID', 'Time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for valid label range
        unique_labels = df['Label'].unique()
        if not all(label in [0, 1, 2] for label in unique_labels):
            raise ValueError(f"Invalid labels found: {unique_labels}")
        
        logger.info("Data validation passed")
    
    def _handle_missing_values(self, df):
        """Handle missing values with forward fill within episodes"""
        for episode in df['Episode_ID'].unique():
            episode_mask = df['Episode_ID'] == episode
            df.loc[episode_mask, self.feature_names] = df.loc[episode_mask, self.feature_names].fillna(method='ffill')
            df.loc[episode_mask, self.feature_names] = df.loc[episode_mask, self.feature_names].fillna(method='bfill')
        
        return df
    
    def _engineer_features(self, df):
        """Advanced feature engineering"""
        logger.info("Engineering additional features...")
        
        # Calculate moving averages with different windows
        for feature in ['Power', 'Fuel_Temp', 'Coolant_Temp']:
            df[f'{feature}_MA3'] = df.groupby('Episode_ID')[feature].transform(lambda x: x.rolling(3, min_periods=1).mean())
            df[f'{feature}_MA7'] = df.groupby('Episode_ID')[feature].transform(lambda x: x.rolling(7, min_periods=1).mean())
        
        # Calculate standard deviation features
        for feature in ['Power', 'Fuel_Temp', 'Coolant_Temp']:
            df[f'{feature}_STD'] = df.groupby('Episode_ID')[feature].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)
        
        # Temperature differentials
        df['Temp_Differential'] = df['Fuel_Temp'] - df['Coolant_Temp']
        df['Temp_Diff_ROC'] = df.groupby('Episode_ID')['Temp_Differential'].diff().fillna(0)
        
        # Power-to-flow ratio (important for nuclear safety)
        df['Power_Flow_Ratio'] = df['Power'] / (df['Flow'] + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Update feature names
        new_features = [col for col in df.columns if col.endswith(('_MA3', '_MA7', '_STD')) or col in ['Temp_Differential', 'Temp_Diff_ROC', 'Power_Flow_Ratio']]
        self.feature_names.extend(new_features)
        
        return df
    
    def _handle_outliers(self, df):
        """Handle outliers using IQR method"""
        logger.info("Handling outliers...")
        
        for feature in self.feature_names:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them to preserve temporal structure
            df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def prepare_sequences(self, df):
        """Enhanced sequence creation with better handling"""
        logger.info("Creating sequences...")
        
        X_seq = []
        y_seq = []
        episode_seq = []
        
        unique_episodes = sorted(df['Episode_ID'].unique())
        
        for episode in unique_episodes:
            episode_data = df[df['Episode_ID'] == episode].reset_index(drop=True)
            episode_features = episode_data[self.feature_names].values
            episode_labels = episode_data['Label'].values
            
            # Create overlapping sequences for better data utilization
            step_size = max(1, self.config['window_size'] // 3)  # 33% overlap
            
            for i in range(0, len(episode_features) - self.config['window_size'], step_size):
                sequence = episode_features[i:i + self.config['window_size']]
                label = episode_labels[i + self.config['window_size'] - 1]  # Label of last timestep
                
                X_seq.append(sequence)
                y_seq.append(label)
                episode_seq.append(episode)
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        episode_seq = np.array(episode_seq)
        
        logger.info(f"Created {len(X_seq)} sequences with shape {X_seq.shape}")
        return X_seq, y_seq, episode_seq
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale features with selected scaler"""
        logger.info(f"Scaling features using {self.config['scaler_type']} scaler...")
        
        if self.config['scaler_type'] == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.config['scaler_type'] == 'standard':
            self.scaler = StandardScaler()
        else:  # robust
            self.scaler = RobustScaler()
        
        # Reshape for fitting scaler
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        # Transform all sets
        X_train_scaled = self._scale_sequences(X_train)
        X_val_scaled = self._scale_sequences(X_val)
        X_test_scaled = self._scale_sequences(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def _scale_sequences(self, X):
        """Scale sequence data"""
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        return X_scaled.reshape(X.shape)
    
    def split_data(self, X_seq, y_seq, episode_seq):
        """Split data by episodes with stratification"""
        logger.info("Splitting data by episodes...")
        
        unique_episodes = np.unique(episode_seq)
        
        # Calculate label distribution per episode for stratification
        episode_labels = []
        for episode in unique_episodes:
            episode_mask = episode_seq == episode
            episode_label = np.bincount(y_seq[episode_mask]).argmax()  # Dominant label
            episode_labels.append(episode_label)
        
        # Stratified split
        train_episodes, test_episodes, _, _ = train_test_split(
            unique_episodes, episode_labels, 
            test_size=0.2, stratify=episode_labels, random_state=42
        )
        
        # Further split train into train/val
        train_episodes, val_episodes, _, _ = train_test_split(
            train_episodes, [episode_labels[i] for i in range(len(unique_episodes)) if unique_episodes[i] in train_episodes],
            test_size=0.2, stratify=[episode_labels[i] for i in range(len(unique_episodes)) if unique_episodes[i] in train_episodes],
            random_state=42
        )
        
        # Create masks
        train_mask = np.isin(episode_seq, train_episodes)
        val_mask = np.isin(episode_seq, val_episodes)
        test_mask = np.isin(episode_seq, test_episodes)
        
        X_train, y_train = X_seq[train_mask], y_seq[train_mask]
        X_val, y_val = X_seq[val_mask], y_seq[val_mask]
        X_test, y_test = X_seq[test_mask], y_seq[test_mask]
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self, input_shape):
        """Build advanced LSTM model with attention"""
        logger.info("Building advanced LSTM model...")
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        if self.config['use_bidirectional']:
            lstm1 = Bidirectional(LSTM(
                self.config['lstm_units'][0], 
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['dropout_rate'],
                kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
            ))(inputs)
        else:
            lstm1 = LSTM(
                self.config['lstm_units'][0], 
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['dropout_rate'],
                kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
            )(inputs)
        
        lstm1 = BatchNormalization()(lstm1)
        
        # Second LSTM layer
        lstm2 = LSTM(
            self.config['lstm_units'][1], 
            return_sequences=True if self.config['use_attention'] else False,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate'],
            kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg'])
        )(lstm1)
        
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention mechanism (if enabled)
        if self.config['use_attention']:
            # Simple attention mechanism
            attention_scores = Dense(1, activation='tanh')(lstm2)
            attention_weights = Softmax(axis=1)(attention_scores)
            weighted_sequence = Multiply()([lstm2, attention_weights])
            x = Lambda(lambda t: tf.reduce_sum(t, axis=1))(weighted_sequence)
        else:
            x = lstm2
        
        # Dense layers
        for units in self.config['dense_units']:
            x = Dense(units, activation='relu', 
                     kernel_regularizer=l1_l2(self.config['l1_reg'], self.config['l2_reg']))(x)
            x = BatchNormalization()(x)
            x = Dropout(self.config['dropout_rate'])(x)
        
        # Output layer
        outputs = Dense(3, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with custom optimizer
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, checkpoint_path='best_nuclear_lstm_model.h5'):
        """Train the model with advanced callbacks"""
        logger.info("Training model...")
        
        # Calculate class weights for imbalanced data
        self.class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = dict(enumerate(self.class_weights))
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Convert to categorical
        y_train_cat = to_categorical(y_train, num_classes=3)
        y_val_cat = to_categorical(y_val, num_classes=3)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['patience']//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        logger.info("Training completed")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Convert to categorical for loss calculation
        y_test_cat = to_categorical(y_test, num_classes=3)
        
        # Basic metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        # Detailed classification report
        class_names = ['Normal', 'Scram', 'LOFA']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class confidence analysis
        confidence_analysis = self._analyze_confidence(y_pred_prob, y_test)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'classification_report': report,
            'confusion_matrix': cm,
            'confidence_analysis': confidence_analysis,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_prob
        }
        
        # Log results
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        
        return results
    
    def _analyze_confidence(self, y_pred_prob, y_true):
        """Analyze prediction confidence"""
        confidence_scores = np.max(y_pred_prob, axis=1)
        
        analysis = {
            'overall_confidence': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            }
        }
        
        # Per-class confidence
        for class_idx, class_name in enumerate(['Normal', 'Scram', 'LOFA']):
            class_mask = y_true == class_idx
            if np.any(class_mask):
                class_confidence = confidence_scores[class_mask]
                analysis[f'{class_name}_confidence'] = {
                    'mean': np.mean(class_confidence),
                    'std': np.std(class_confidence),
                    'samples': int(np.sum(class_mask))
                }
        
        return analysis
    
    def plot_results(self, results, output_path='nuclear_lstm_results.png'):
        """Plot comprehensive results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training history
        if self.history:
            axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
            axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].legend()
            
            axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
            axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
            axes[0, 1].set_title('Model Loss')
            axes[0, 1].legend()
        
        # Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   xticklabels=['Normal', 'Scram', 'LOFA'],
                   yticklabels=['Normal', 'Scram', 'LOFA'],
                   ax=axes[0, 2])
        axes[0, 2].set_title('Confusion Matrix')
        
        # Confidence distribution
        confidence_scores = np.max(results['prediction_probabilities'], axis=1)
        axes[1, 0].hist(confidence_scores, bins=30, alpha=0.7)
        axes[1, 0].set_title('Prediction Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence Score')
        
        # Per-class performance
        report = results['classification_report']
        classes = ['Normal', 'Scram', 'LOFA']
        metrics = ['precision', 'recall', 'f1-score']

        # classification_report output_dict uses class names as provided in target_names
        bar_positions = np.arange(len(classes))
        bar_width = 0.25
        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in classes]
            axes[1, 1].bar(bar_positions + i * bar_width, values, width=bar_width, alpha=0.8, label=metric)
        axes[1, 1].set_xticks(bar_positions + bar_width)
        axes[1, 1].set_xticklabels(classes)
        axes[1, 1].legend()
        axes[1, 1].set_title('Per-Class Performance')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Learning rate (if available)
        if self.history and 'learning_rate' in self.history.history:
            axes[1, 2].plot(self.history.history['learning_rate'])
            axes[1, 2].set_title('Learning Rate Schedule')
        elif self.history and 'lr' in self.history.history:
            axes[1, 2].plot(self.history.history['lr'])
            axes[1, 2].set_title('Learning Rate Schedule')
        else:
            axes[1, 2].text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                          ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='nuclear_lstm_model.h5'):
        """Save the complete model"""
        output_dir = os.path.dirname(filepath) or '.'
        os.makedirs(output_dir, exist_ok=True)

        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        
        # Save configuration and scaler
        config_data = {
            'config': self.config,
            'feature_names': self.feature_names,
            'scaler_params': {
                'type': type(self.scaler).__name__,
                'scale_': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'center_': self.scaler.center_.tolist() if hasattr(self.scaler, 'center_') else None,
            }
        }
        
        config_path = os.path.join(output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Save scaler separately
        import joblib
        scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model configuration saved to {config_path}")
        logger.info(f"Scaler saved to {scaler_path}")

def main():
    """Main training pipeline"""

    parser = argparse.ArgumentParser(description='Train advanced nuclear LSTM model.')
    parser.add_argument(
        '--data-path',
        default='nuclear_fault_data_enhanced.csv',
        help='Path to training CSV file.'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Directory to write trained model and artifacts.'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration
    config = {
        'window_size': 20,  # Increased for better temporal understanding
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'l1_reg': 0.01,
        'l2_reg': 0.01,
        'lstm_units': [128, 64, 32],
        'dense_units': [64, 32],
        'patience': 20,
        'use_bidirectional': True,
        'use_attention': True,
        'scaler_type': 'robust'
    }
    
    # Initialize model
    model = NuclearLSTMModel(config)
    
    # Load and preprocess data
    df = model.load_and_preprocess_data(args.data_path)
    
    # Prepare sequences
    X_seq, y_seq, episode_seq = model.prepare_sequences(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(X_seq, y_seq, episode_seq)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = model.scale_features(X_train, X_val, X_test)
    
    # Build model
    input_shape = (config['window_size'], len(model.feature_names))
    model.build_model(input_shape)
    
    # Print model summary
    print(model.model.summary())
    
    # Train model
    checkpoint_path = os.path.join(args.output_dir, 'best_nuclear_lstm_model.h5')
    history = model.train(X_train_scaled, y_train, X_val_scaled, y_val, checkpoint_path=checkpoint_path)
    
    # Evaluate model
    results = model.evaluate(X_test_scaled, y_test)
    
    # Print detailed results
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Precision: {results['test_precision']:.4f}")
    print(f"Test Recall: {results['test_recall']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, results['predictions'], 
                              target_names=['Normal', 'Scram', 'LOFA']))
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    print("\nConfidence Analysis:")
    conf_analysis = results['confidence_analysis']
    print(f"Overall Confidence - Mean: {conf_analysis['overall_confidence']['mean']:.3f}")
    for class_name in ['Normal', 'Scram', 'LOFA']:
        if f'{class_name}_confidence' in conf_analysis:
            class_conf = conf_analysis[f'{class_name}_confidence']
            print(f"{class_name} Confidence - Mean: {class_conf['mean']:.3f} (±{class_conf['std']:.3f})")
    
    # Plot results
    results_plot_path = os.path.join(args.output_dir, 'nuclear_lstm_results.png')
    model.plot_results(results, output_path=results_plot_path)
    
    # Save model
    model.save_model(os.path.join(args.output_dir, 'advanced_nuclear_lstm_model.h5'))
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()