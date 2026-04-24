import os
import sys
import numpy as np

# Mock pylsl to prevent RealTimeInference from crashing when resolving streams
import sys
import types
mock_pylsl = types.ModuleType("pylsl")
mock_pylsl.StreamInfo = lambda *args, **kwargs: None
mock_pylsl.StreamOutlet = lambda *args, **kwargs: type("MockOutlet", (), {"push_sample": lambda self, x, y: None})()
mock_pylsl.StreamInlet = lambda *args, **kwargs: None
mock_pylsl.local_clock = lambda: 0.0
sys.modules["pylsl"] = mock_pylsl

from realtime_inference import RealTimeInference

def run_simulation():
    raw_dir = r"c:\Omar\Education\NeuroTech_ASU\P300\backend\raw_session"
    print("Loading raw session data...")
    data = np.load(os.path.join(raw_dir, "eeg_continuous.npy"))
    times = np.load(os.path.join(raw_dir, "eeg_timestamps.npy"))
    events = np.load(os.path.join(raw_dir, "flash_events.npy"), allow_pickle=True)
    
    agent = RealTimeInference()
    agent.load_and_train_model()
    
    # Pre-load all EEG data into the agent's buffer (simulating a full 120s ring buffer)
    # Since the session is about 6 minutes, we will just feed it the data sequentially.
    
    # Group events by target character
    trials = []
    current_trial = []
    current_target = events[0]['target']
    
    for ev in events:
        if ev['target'] != current_target:
            trials.append((current_target, current_trial))
            current_target = ev['target']
            current_trial = []
        current_trial.append(ev)
    if current_trial:
        trials.append((current_target, current_trial))
        
    print(f"\nSimulating {len(trials)} trials...")
    
    correct_count = 0
    
    for target, flashes in trials:
        print(f"\n--- Testing Target: {target} ---")
        agent._reset_trial_state()
        
        # We need to feed the agent the EEG data up to the last flash of this trial
        # plus some padding.
        last_flash_time = flashes[-1]['time']
        idx_end = np.searchsorted(times, last_flash_time + 1.0) # 1 second after last flash
        
        # Feed the buffer
        agent.eeg_data.clear()
        agent.eeg_times.clear()
        # Feed up to 120 seconds of history
        idx_start = max(0, idx_end - (250 * 120))
        agent.eeg_data.extend(data[idx_start:idx_end])
        agent.eeg_times.extend(times[idx_start:idx_end])
        
        # Feed the flashes block by block (12 flashes per block)
        blocks = [flashes[i:i+12] for i in range(0, len(flashes), 12)]
        
        hit_threshold = False
        decoded_char = None
        blocks_used = 0
        
        for block in blocks:
            blocks_used += 1
            for f in block:
                agent.current_trial_flashes.append((f['time'], f['group']))
            
            # Simulate EVALUATE after block
            pred, hit = agent.decode_trial(check_threshold=True)
            if hit:
                hit_threshold = True
                decoded_char = pred
                print(f"Dynamic Stop Hit at block {blocks_used}! Predicted: {pred}")
                break
                
        if not hit_threshold:
            # Simulate TRIAL_END
            pred, _ = agent.decode_trial(check_threshold=False)
            decoded_char = pred
            print(f"Reached Max Flashes (15 blocks). Fallback Predicted: {pred}")
            
        if decoded_char == target:
            correct_count += 1
            print(">> SUCCESS <<")
        else:
            print(f">> FAILED << (Expected {target}, got {decoded_char})")
            
    print(f"\nFINAL ACCURACY: {correct_count}/{len(trials)} ({(correct_count/len(trials))*100:.1f}%)")

if __name__ == '__main__':
    run_simulation()
