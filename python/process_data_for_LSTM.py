import pandas as pd
import numpy as np

def enhance_dataset():
    print("Loading nuclear_fault_data_v2.csv...")
    df = pd.read_csv('nuclear_fault_data_v2.csv')

    # --- STEP 1: Sort Data ---
    # Ensure data is in order so 'Previous Value' makes sense
    df = df.sort_values(by=['Episode_ID', 'Time'])

    # --- STEP 2: Group by Episode ---
    # We must group by 'Episode_ID' so that the last row of Episode 0 
    # doesn't get subtracted from the first row of Episode 1.
    grouped = df.groupby('Episode_ID')

    # --- STEP 3: Calculate Time Step (dt) ---
    # How much time passed between rows? (approx 0.1s)
    # We use fillna(0.1) for the first row of each episode to avoid division by zero
    dt = grouped['Time'].diff().fillna(0.1)

    # --- STEP 4: Calculate ROC (Derivatives) ---
    # Formula: (Current - Prev) / dt
    print("Calculating Rate of Change (Physics Derivatives)...")
    df['Power_ROC'] = grouped['Power'].diff().fillna(0) / dt
    df['Temp_Fuel_ROC'] = grouped['Fuel_Temp'].diff().fillna(0) / dt
    df['Temp_Coolant_ROC'] = grouped['Coolant_Temp'].diff().fillna(0) / dt
    df['Flow_ROC'] = grouped['Flow'].diff().fillna(0) / dt

    # --- STEP 5: Calculate Smoothing (Moving Average) ---
    # Takes the average of the last 5 rows to reduce noise
    print("Calculating Moving Averages (Noise Filtering)...")
    window = 5
    df['Power_Smooth'] = grouped['Power'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['Temp_Fuel_Smooth'] = grouped['Fuel_Temp'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['Temp_Coolant_Smooth'] = grouped['Coolant_Temp'].transform(lambda x: x.rolling(window, min_periods=1).mean())

    # --- STEP 6: Reorder & Save ---
    # Put 'Label' at the very end (standard practice)
    columns_order = [
        'Time', 'Episode_ID', 
        'Power', 'Power_ROC', 'Power_Smooth',
        'Fuel_Temp', 'Temp_Fuel_ROC', 'Temp_Fuel_Smooth',
        'Coolant_Temp', 'Temp_Coolant_ROC', 'Temp_Coolant_Smooth',
        'Pressure', 
        'Flow', 'Flow_ROC',
        'Label'
    ]
    df = df[columns_order]
    
    output_file = 'nuclear_fault_data_enhanced.csv'
    df.to_csv(output_file, index=False)
    print(f"SUCCESS: Saved enhanced data to {output_file}")
    print("You can now give this file to the LSTM team.")

if __name__ == "__main__":
    enhance_dataset()