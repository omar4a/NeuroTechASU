import os
import numpy as np
import pylsl
import sys

base_dir = r'c:\Omar\Education\NeuroTech_ASU\P300\backend'
sys.path.append(base_dir)

from realtime_inference import RealTimeInference
from signal_processing import MATRIX_CHARS

def main():
    rec_dir = os.path.join(base_dir, "inference_recording")
    
    print("Loading recording...")
    eeg_data = np.load(os.path.join(rec_dir, "eeg_continuous.npy"))
    eeg_times = np.load(os.path.join(rec_dir, "eeg_timestamps.npy"))
    flashes = np.load(os.path.join(rec_dir, "flash_events.npy"), allow_pickle=True)
    predictions = np.load(os.path.join(rec_dir, "predictions.npy"), allow_pickle=True)
    
    print(f"EEG Shape: {eeg_data.shape}")
    print(f"Flashes: {len(flashes)}")
    print(f"Predictions: {predictions}")
    
    # Let's manually run the decoding logic on this recorded data
    agent = RealTimeInference()
    agent.load_and_train_model()
    
    # Fill buffer
    agent.eeg_data.extend(eeg_data)
    agent.eeg_times.extend(eeg_times)
    
    agent._reset_trial_state()
    agent.current_trial_flashes = [tuple(f.values()) for f in flashes]
    
    print(f"\nEvaluating the recorded trial (total flashes = {len(flashes)})...")
    best_char, hit_threshold = agent.decode_trial(check_threshold=False)
    
    print(f"\nFinal Decode Result: {best_char} (hit_threshold={hit_threshold})")
    
if __name__ == '__main__':
    main()
