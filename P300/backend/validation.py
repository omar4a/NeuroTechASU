"""
P300 ERP Validation Visualizer.

Plots the Grand Average ERP waveform at channel Pz, comparing target vs. non-target
averages. Used to visually verify that a P300 component is present in the data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from signal_processing import PZ_INDEX, EPOCH_LEN

# Use portable path resolution instead of hardcoded absolute paths
from signal_processing import TRAINING_DATA_DIR

# 1. Load your data
# Assuming X_train shape is (epochs, channels, timepoints) 
# Assuming 250Hz sampling rate and 800ms epoch (200 timepoints)
X = np.load(os.path.join(TRAINING_DATA_DIR, "X_train.npy"))
y = np.load(os.path.join(TRAINING_DATA_DIR, "y_train.npy"))

# PZ_INDEX imported from signal_processing (shared single source of truth)

# 2. Separate Targets and Non-Targets
if len(X) == 0:
    raise ValueError("CRITICAL ERROR: X_train.npy is completely empty (shape is 0). You have not collected any valid data yet! Run data_collection.py and psychopy_speller.py fully before visualizing.")

if len(X.shape) < 3:
    raise ValueError(f"CRITICAL ERROR: X_train.npy has malformed shape {X.shape}. It should be (Epochs, Channels, Samples). Ensure your training session fully completed successfully.")

targets = X[y == 1]
nontargets = X[y == 0]

print(f"Total Targets: {len(targets)}")
print(f"Total Non-Targets: {len(nontargets)}")

if len(targets) == 0:
    raise ValueError("No Target flashes found in dataset!")

# 3. Calculate the Grand Average for the Pz channel
# np.mean across the epoch dimension (axis=0)
target_avg = np.mean(targets[:, PZ_INDEX, :], axis=0)
nontarget_avg = np.mean(nontargets[:, PZ_INDEX, :], axis=0)

# Create a time axis in milliseconds using shared EPOCH_LEN constant
epoch_ms = EPOCH_LEN * 1000
times = np.linspace(0, epoch_ms, X.shape[2])

# 4. Plot it
plt.figure(figsize=(10, 6))
plt.plot(times, target_avg, label='Target (y=1)', color='red', linewidth=2)
plt.plot(times, nontarget_avg, label='Non-Target (y=0)', color='blue', linewidth=2)

plt.axvline(x=300, color='gray', linestyle='--', label='Theoretical P300')
plt.title('Grand Average ERP at Channel Pz')
plt.xlabel('Time (ms) after visual flash')
plt.ylabel('Amplitude (µV)')
plt.legend()
plt.grid(True)
plt.show()