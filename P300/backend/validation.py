import numpy as np
import matplotlib.pyplot as plt
import mne

# 1. Load your data
# Assuming X_train shape is (epochs, channels, timepoints) 
# Assuming 250Hz sampling rate and 800ms epoch (200 timepoints)
X = np.load("C:\\Omar\\Education\\NeuroTech_ASU\\P300\\backend\\X_train_v2.npy")
y = np.load("C:\\Omar\\Education\\NeuroTech_ASU\\P300\\backend\\y_train_v2.npy")

# Get the index of the Pz channel (usually channel 4 in your 8-channel array)
# Update this index if your channel order is different!
PZ_INDEX = 4 

# 2. Separate Targets and Non-Targets
targets = X[y == 1]
nontargets = X[y == 0]

print(f"Total Targets: {len(targets)}")
print(f"Total Non-Targets: {len(nontargets)}")

# 3. Calculate the Grand Average for the Pz channel
# np.mean across the epoch dimension (axis=0)
target_avg = np.mean(targets[:, PZ_INDEX, :], axis=0)
nontarget_avg = np.mean(nontargets[:, PZ_INDEX, :], axis=0)

# Create a time axis in milliseconds (e.g., 0 to 800ms)
times = np.linspace(0, 800, X.shape[2])

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