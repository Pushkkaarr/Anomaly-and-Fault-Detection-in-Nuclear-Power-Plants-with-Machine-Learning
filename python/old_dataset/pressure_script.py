# Generating a TPFR dataset (Pressure-focused) with physics-informed rules and anomalies
# - Timesteps: 20,000
# - Sampling interval: 1 minute
# - Random seed: 42
# Outputs:
# - CSV saved to /mnt/data/pressure_tpfr_dataset.csv
# - Dataframe displayed for user preview
# - Summary statistics printed for validation checks

import numpy as np
import pandas as pd
from pathlib import Path

# reproducibility
np.random.seed(42)

# Parameters (from specification)
TIMESTEPS = 20000
SAMPLING_INTERVAL_MIN = 1  # minute per timestep
SEED = 42

# Nominal points and ranges
T_mean = 300.0
T_min, T_max = 280.0, 320.0

F_mean = 150.0
F_min, F_max = 100.0, 200.0

R_mean = 750.0
R_min, R_max = 500.0, 1000.0

P_mean = 175.0
P_min, P_max = 120.0, 220.0
P_soft_min, P_soft_max = 150.0, 200.0

# Dynamics coefficients for P
alpha = 0.08   # bar per °C
beta = -0.05   # bar per (m^3/h)
gamma = 0.002  # bar per MW

sigma_proc = 0.5   # process noise for P (bar)
sigma_sensor = 0.3 # sensor noise for P_obs (bar)
max_step_clip = 2.5
max_plausible_step = 6.0

# Anomaly frequency targets (for 20k timesteps)
freq_spike = 0.002    # => ~40 spikes
freq_stuck = 0.001    # => ~20 stuck events
freq_drift = 0.003    # => ~60 drift starts
freq_process_fault = 0.0015  # => ~30 process faults
freq_physics_violation = 0.0005  # => ~10 physics violations
freq_missing = 0.005  # => ~100 missing samples

# Derived counts (rounded)
count_spike = max(1, int(round(freq_spike * TIMESTEPS)))
count_stuck = max(1, int(round(freq_stuck * TIMESTEPS)))
count_drift = max(1, int(round(freq_drift * TIMESTEPS)))
count_process_fault = max(1, int(round(freq_process_fault * TIMESTEPS)))
count_physics_violation = max(1, int(round(freq_physics_violation * TIMESTEPS)))
count_missing = max(1, int(round(freq_missing * TIMESTEPS)))

# Initialize arrays
timestamps = pd.date_range(start='2025-10-06', periods=TIMESTEPS, freq=f'{SAMPLING_INTERVAL_MIN}T')
T_true = np.zeros(TIMESTEPS)
F_true = np.zeros(TIMESTEPS)
R_true = np.zeros(TIMESTEPS)
P_true = np.zeros(TIMESTEPS)

# Simulate baseline for T, F, R using small AR(1)-like dynamics with low-frequency transients
sigma_T = 0.5  # °C per step noise
sigma_F = 1.0  # m3/h per step noise
sigma_R = 2.0  # MW per step noise

T_true[0] = T_mean + np.random.normal(0, 0.5)
F_true[0] = F_mean + np.random.normal(0, 2.0)
R_true[0] = R_mean + np.random.normal(0, 5.0)

for t in range(1, TIMESTEPS):
    # small autoregressive tendencies with slight mean reversion
    T_true[t] = 0.995 * T_true[t-1] + 0.005 * T_mean + np.random.normal(0, sigma_T)
    F_true[t] = 0.995 * F_true[t-1] + 0.005 * F_mean + np.random.normal(0, sigma_F)
    R_true[t] = 0.995 * R_true[t-1] + 0.005 * R_mean + np.random.normal(0, sigma_R)
    # clip to physical plausible ranges
    T_true[t] = np.clip(T_true[t], T_min, T_max)
    F_true[t] = np.clip(F_true[t], F_min, F_max)
    R_true[t] = np.clip(R_true[t], R_min, R_max)

# Pre-generate anomaly schedules (start indices and durations)
rng = np.random.default_rng(SEED)

# Helper to pick non-overlapping start indices for events with durations (best-effort)
def pick_starts(count, min_dur=1, max_dur=1):
    starts = []
    attempts = 0
    while len(starts) < count and attempts < count * 100:
        s = rng.integers(0, TIMESTEPS - max_dur)
        # allow overlap but avoid too many exact collisions
        starts.append(int(s))
        attempts += 1
    return starts

spike_starts = pick_starts(count_spike, 1, 1)
stuck_starts = pick_starts(count_stuck, 50, 500)
drift_starts = pick_starts(count_drift, 200, 2000)
process_fault_starts = pick_starts(count_process_fault, 100, 1000)
physics_violation_starts = pick_starts(count_physics_violation, 1, 5)
missing_starts = pick_starts(count_missing, 1, 10)

# For events needing durations, randomly select durations
stuck_durations = [int(rng.integers(50, 501)) for _ in range(len(stuck_starts))]
drift_durations = [int(rng.integers(200, 2001)) for _ in range(len(drift_starts))]
process_fault_durations = [int(rng.integers(100, 1001)) for _ in range(len(process_fault_starts))]
physics_violation_durations = [int(rng.integers(1, 6)) for _ in range(len(physics_violation_starts))]
missing_durations = [int(rng.integers(1, 11)) for _ in range(len(missing_starts))]

# Prepare arrays for annotations
anomaly_flag = np.zeros(TIMESTEPS, dtype=int)
anomaly_type = np.array(['none'] * TIMESTEPS, dtype=object)
physics_residual = np.zeros(TIMESTEPS, dtype=float)
physics_violation_bool = np.zeros(TIMESTEPS, dtype=bool)

# First, create baseline P_true using the dynamics and process noise
P_true[0] = P_mean + np.random.normal(0, 0.5)
for t in range(1, TIMESTEPS):
    P_pred = P_true[t-1] + alpha * (T_true[t] - T_true[t-1]) + beta * (F_true[t] - F_true[t-1]) + gamma * (R_true[t] - R_true[t-1])
    proc_noise = np.random.normal(0, sigma_proc)
    P_true[t] = P_pred + proc_noise
    # soft clipping for normal transients (enforce inertia)
    step = P_true[t] - P_true[t-1]
    if abs(step) > max_step_clip:
        # limit to max_step_clip but keep sign
        P_true[t] = P_true[t-1] + np.sign(step) * max_step_clip

    # ensure within absolute safe bounds for baseline (we will allow violations only via anomalies)
    P_true[t] = np.clip(P_true[t], P_min, P_max)

# Now inject process_faults by altering F_true during those durations
for idx, dur in zip(process_fault_starts, process_fault_durations):
    if idx >= TIMESTEPS:
        continue
    dur = min(dur, TIMESTEPS - idx - 1)
    # change in flow: decrease by 20-40 m3/h sustained, implemented as immediate drop then slow recovery
    delta_f = -rng.integers(20, 41)
    for k in range(dur):
        t = idx + k
        F_true[t] = np.clip(F_true[t] + delta_f, F_min, F_max)

# Recompute P_true after process_faults to reflect the F changes (recompute forward from start)
P_true[0] = P_mean + np.random.normal(0, 0.5)
for t in range(1, TIMESTEPS):
    P_pred = P_true[t-1] + alpha * (T_true[t] - T_true[t-1]) + beta * (F_true[t] - F_true[t-1]) + gamma * (R_true[t] - R_true[t-1])
    proc_noise = np.random.normal(0, sigma_proc)
    P_true[t] = P_pred + proc_noise
    step = P_true[t] - P_true[t-1]
    if abs(step) > max_step_clip:
        P_true[t] = P_true[t-1] + np.sign(step) * max_step_clip
    P_true[t] = np.clip(P_true[t], P_min, P_max)

# Now create observed P_obs from P_true with sensor noise, and then apply P-specific anomalies
P_obs = P_true.copy() + np.random.normal(0, sigma_sensor, size=TIMESTEPS)

# Apply spike anomalies (single-step large deviations)
for s in spike_starts:
    if s < 0 or s >= TIMESTEPS:
        continue
    magnitude = rng.integers(8, 21) * (1 if rng.random() < 0.5 else -1)  # ±[8,20]
    P_obs[s] = np.clip(P_obs[s] + magnitude, P_min - 50, P_max + 50)  # allow outside safe bounds for spikes
    anomaly_flag[s] = 1
    anomaly_type[s] = 'spike'

# Apply stuck-at anomalies (freeze sensor reading)
for s, dur in zip(stuck_starts, stuck_durations):
    if s < 0 or s >= TIMESTEPS:
        continue
    end = min(TIMESTEPS, s + dur)
    stuck_value = float(P_obs[s])
    for t in range(s, end):
        P_obs[t] = stuck_value
        anomaly_flag[t] = 1
        anomaly_type[t] = 'stuck'

# Apply drift anomalies (slow bias accumulation)
for s, dur in zip(drift_starts, drift_durations):
    if s < 0 or s >= TIMESTEPS:
        continue
    end = min(TIMESTEPS, s + dur)
    delta = rng.uniform(0.01, 0.05)  # per timestep added bias
    bias = 0.0
    for t in range(s, end):
        bias += delta
        P_obs[t] = P_obs[t] + bias
        anomaly_flag[t] = 1
        anomaly_type[t] = 'drift'

# Apply physics_violation anomalies (set observed P far from prediction)
for s, dur in zip(physics_violation_starts, physics_violation_durations):
    if s < 0 or s >= TIMESTEPS:
        continue
    end = min(TIMESTEPS, s + dur)
    for t in range(s, end):
        # compute P_pred at t using P_true[t-1] and true deltas (we'll base on current P_true as reference)
        if t == 0:
            continue
        P_pred_t = P_true[t-1] + alpha * (T_true[t] - T_true[t-1]) + beta * (F_true[t] - F_true[t-1]) + gamma * (R_true[t] - R_true[t-1])
        delta = rng.uniform(8.0, 15.0)
        P_obs[t] = P_pred_t + delta
        anomaly_flag[t] = 1
        anomaly_type[t] = 'physics_violation'

# Apply missing/dropout anomalies
for s, dur in zip(missing_starts, missing_durations):
    if s < 0 or s >= TIMESTEPS:
        continue
    end = min(TIMESTEPS, s + dur)
    for t in range(s, end):
        P_obs[t] = np.nan
        anomaly_flag[t] = 1 if anomaly_type[t] == 'none' else 1
        # preserve earlier labels if present, otherwise mark as missing
        if anomaly_type[t] == 'none':
            anomaly_type[t] = 'missing'

# Ensure at least one of every anomaly exists
required_types = ['spike', 'stuck', 'drift', 'process_fault', 'physics_violation', 'missing']
present = set(anomaly_type)
for rtype in required_types:
    if rtype not in present:
        # force-inject a single instance near the middle
        idx = TIMESTEPS // 2
        if rtype == 'spike':
            P_obs[idx] = P_obs[idx] + 10.0
            anomaly_flag[idx] = 1
            anomaly_type[idx] = 'spike'
        elif rtype == 'stuck':
            dur = 60
            val = float(P_obs[idx])
            for t in range(idx, min(TIMESTEPS, idx+dur)):
                P_obs[t] = val
                anomaly_flag[t] = 1
                anomaly_type[t] = 'stuck'
        elif rtype == 'drift':
            dur = 300
            bias = 0.0
            delta = 0.02
            for t in range(idx, min(TIMESTEPS, idx+dur)):
                bias += delta
                P_obs[t] = P_obs[t] + bias
                anomaly_flag[t] = 1
                anomaly_type[t] = 'drift'
        elif rtype == 'process_fault':
            dur = 200
            delta_f = -25
            for t in range(idx, min(TIMESTEPS, idx+dur)):
                F_true[t] = np.clip(F_true[t] + delta_f, F_min, F_max)
            # recompute P_true forward from idx to reflect forced flow change, then set P_obs accordingly
            for t in range(idx, TIMESTEPS):
                if t == idx:
                    P_base = P_true[t-1]
                P_pred = P_base + alpha * (T_true[t] - T_true[t-1]) + beta * (F_true[t] - F_true[t-1]) + gamma * (R_true[t] - R_true[t-1])
                P_true[t] = P_pred + np.random.normal(0, sigma_proc)
                P_true[t] = np.clip(P_true[t], P_min, P_max)
                P_obs[t] = P_true[t] + np.random.normal(0, sigma_sensor)
                anomaly_flag[t] = 1
                anomaly_type[t] = 'process_fault'
        elif rtype == 'physics_violation':
            delta = 9.0
            P_obs[idx] = P_obs[idx] + delta
            anomaly_flag[idx] = 1
            anomaly_type[idx] = 'physics_violation'
        elif rtype == 'missing':
            P_obs[idx] = np.nan
            anomaly_flag[idx] = 1
            anomaly_type[idx] = 'missing'

# After anomaly injections, recompute P_pred (using P_true as reference) and physics residuals
P_pred_arr = np.zeros(TIMESTEPS)
for t in range(TIMESTEPS):
    if t == 0:
        P_pred_arr[t] = P_true[0]  # no prediction for first step aside from base
    else:
        P_pred_arr[t] = P_true[t-1] + alpha * (T_true[t] - T_true[t-1]) + beta * (F_true[t] - F_true[t-1]) + gamma * (R_true[t] - R_true[t-1])

# physics_residual based on observed (P_obs) vs predicted
for t in range(TIMESTEPS):
    if np.isnan(P_obs[t]):
        physics_residual[t] = np.nan
        physics_violation_bool[t] = False
    else:
        physics_residual[t] = abs(P_obs[t] - P_pred_arr[t])
        physics_violation_bool[t] = physics_residual[t] > 3.0

# If an anomaly type was not set (none) but physics residual > 3.0 and not marked anomalous, mark as physics_violation
for t in range(TIMESTEPS):
    if anomaly_type[t] == 'none' and not np.isnan(physics_residual[t]) and physics_residual[t] > 3.0:
        anomaly_flag[t] = 1
        anomaly_type[t] = 'physics_violation'

# Prepare final DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'T_obs': np.round(T_true, 3),
    'P_true': np.round(P_true, 3),
    'P_obs': np.round(P_obs, 3),
    'P_pred': np.round(P_pred_arr, 3),
    'F_obs': np.round(F_true, 3),
    'R_obs': np.round(R_true, 3),
    'physics_residual': np.round(physics_residual, 3),
    'physics_violation': physics_violation_bool,
    'anomaly_flag': anomaly_flag,
    'anomaly_type': anomaly_type
})

# Save to CSV
output_path = Path('/mnt/data/pressure_tpfr_dataset.csv')
df.to_csv(output_path, index=False)

# Validation summaries
normal_mask = df['anomaly_flag'] == 0
mean_P_obs = df.loc[normal_mask, 'P_obs'].mean()
std_P_obs = df.loc[normal_mask, 'P_obs'].std()

residual_median = df['physics_residual'].dropna().median()
residual_95 = df['physics_residual'].dropna().quantile(0.95)
residual_max = df['physics_residual'].dropna().max()

counts = df['anomaly_type'].value_counts()

summary = {
    'timesteps': TIMESTEPS,
    'mean_P_obs_normal': float(np.round(mean_P_obs, 3)),
    'std_P_obs_normal': float(np.round(std_P_obs, 3)),
    'residual_median': float(np.round(residual_median, 3)),
    'residual_95pct': float(np.round(residual_95, 3)),
    'residual_max': float(np.round(residual_max, 3)),
    'anomaly_counts': counts.to_dict()
}

print('=== Generation Summary ===')
for k, v in summary.items():
    print(f'{k}: {v}')

# Display first rows for preview using display function
import caas_jupyter_tools as cjt
cjt.display_dataframe_to_user('Pressure TPFR dataset preview', df.head(200))

# Provide file link for download (the assistant will include the link in its reply)
print(f"\nSaved dataset to: {output_path}")
