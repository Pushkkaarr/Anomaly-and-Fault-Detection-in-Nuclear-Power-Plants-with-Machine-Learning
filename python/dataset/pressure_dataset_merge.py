import pandas as pd

# Paths to your two CSV datasets
csv1_path = 'csv/pressure_tpfr_dataset.csv'   # previous anomaly-heavy dataset
csv2_path = 'csv/pressure_tpfr_dataset_balanced.csv'  # current mostly-normal dataset

# Load datasets
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Concatenate vertically
merged_df = pd.concat([df1, df2], ignore_index=True)

# Optional: shuffle rows to mix anomalies and normal periods
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Optional: recalculate timestamp if you want a continuous sequence
merged_df['timestamp'] = range(len(merged_df))

# Save merged dataset
merged_csv_path = 'merged_pressure_dataset.csv'
merged_df.to_csv(merged_csv_path, index=False)

print(f"Merged dataset saved as {merged_csv_path}, total rows: {len(merged_df)}")
