import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Step 1: Environment Setup
# Ensure you have TensorFlow, Pandas, NumPy, Scikit-Learn installed
# pip install tensorflow pandas numpy scikit-learn

# Step 2: Data Loading & Scaling
def load_and_scale_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Features as per document
    features = ['Power', 'Fuel_Temp', 'Coolant_Temp', 'Pressure', 'Flow', 'Power_ROC', 'Temp_Fuel_ROC', 'Temp_Coolant_ROC', 'Flow_ROC']
    target = 'Label'
    
    # Separate features and target
    X = df[features]
    y = df[target]
    
    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values, scaler, df['Episode_ID'].values

# Step 3: Sequence Creation (Sliding Windows)
def create_sequences(X, y, episode_ids, window_size=10):
    X_seq = []
    y_seq = []
    ep_seq = []  # to track which episode each sequence belongs to
    
    unique_episodes = np.unique(episode_ids)
    
    for ep in unique_episodes:
        ep_mask = episode_ids == ep
        X_ep = X[ep_mask]
        y_ep = y[ep_mask]
        
        for i in range(len(X_ep) - window_size):
            X_seq.append(X_ep[i:i+window_size])
            y_seq.append(y_ep[i+window_size])  # Label of the last timestep
            ep_seq.append(ep)
    
    return np.array(X_seq), np.array(y_seq), np.array(ep_seq)

# Step 4: Train-Test Split by Episodes
def split_by_episodes(X_seq, y_seq, ep_seq, train_episodes, test_episodes):
    train_mask = np.isin(ep_seq, train_episodes)
    test_mask = np.isin(ep_seq, test_episodes)
    
    X_train = X_seq[train_mask]
    y_train = y_seq[train_mask]
    X_test = X_seq[test_mask]
    y_test = y_seq[test_mask]
    
    return X_train, y_train, X_test, y_test

# Step 5: Model Definition & Compilation
def build_model(input_shape, num_classes=3):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main function
def main():
    file_path = 'nuclear_fault_data_enhanced.csv'
    
    # Load and scale
    X_scaled, y, scaler, episode_ids = load_and_scale_data(file_path)
    
    # Create sequences
    window_size = 10
    X_seq, y_seq, ep_seq = create_sequences(X_scaled, y, episode_ids, window_size)
    
    # One-hot encode labels
    y_seq_cat = to_categorical(y_seq, num_classes=3)
    
    # Split by episodes
    train_episodes = list(range(80))  # 0-79
    test_episodes = list(range(80, 100))  # 80-99
    X_train, y_train, X_test, y_test = split_by_episodes(X_seq, y_seq_cat, ep_seq, train_episodes, test_episodes)
    
    # Build model
    input_shape = (window_size, X_train.shape[2])
    model = build_model(input_shape)
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, classification_report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=['Normal', 'Scram', 'LOFA']))
    
    # Save model
    model.save('lstm_nuclear_fault_detection.h5')
    print("Model saved as lstm_nuclear_fault_detection.h5")
    
    # Plot training history
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()