"""
P300 Training Data Collection Backend.

This script is the "recorder." It listens to two streams at the same time:
1. The Brain: EEG data from the Unicorn headset.
2. The Screen: Markers from the PsychoPy UI telling us when a light flashed.

When you finish a training session, this script matches the brain waves to the 
flashes, cleans them, and saves them as .npy files so the AI can learn from them.
"""

import os
import sys
import numpy as np
import pylsl
import asyncio

# PsychoPy sometimes uses its own "private" Python folder. We need to make sure
# it can find our scientific libraries (mne, pyriemann, sklearn).
_user_site = os.path.expanduser(r"~\AppData\Roaming\Python\Python310\site-packages")
if os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.append(_user_site)

from signal_processing import (
    FS, SAMPLES_PER_EPOCH, BASELINE_SAMPLES, ARTIFACT_THRESHOLD_UV,
    apply_preprocessing, reject_artifacts, extract_epoch
)


class DataCollector:
    def __init__(self):
        # We store everything in lists during the recording
        self.eeg_data = []      # The raw brain wave values
        self.eeg_times = []     # The exact time (in seconds) for each value
        self.flash_events = []  # A list of (timestamp, label) for every flash
        self.is_recording = False
        
    async def lsl_worker(self, inlet):
        """
        Worker 1: The EEG Listener.
        This runs in the background and constantly pulls brain data from the Unicorn.
        """
        print("Connected to Unicorn stream. Receiving data...")
        while self.is_recording:
            # pull_chunk grabs a bunch of data at once to be efficient
            chunk, timestamps = inlet.pull_chunk(max_samples=1250)
            if timestamps:
                self.eeg_data.extend(chunk)
                self.eeg_times.extend(timestamps)
            
            # Wait a tiny bit (4ms) so we don't hog the CPU
            await asyncio.sleep(0.004) 
            
    async def marker_worker(self, marker_inlet):
        """
        Worker 2: The Screen Listener.
        This listens for the PsychoPy UI telling us what flashed and when.
        """
        print("Connected to Marker stream. Receiving markers...")
        while self.is_recording:
            # timeout=0.0 means "check if there's a marker right now, don't wait"
            marker, timestamp = marker_inlet.pull_sample(timeout=0.0)
            if timestamp is not None and marker is not None:
                m_str = marker[0]
                
                # If the UI says the session is over, we stop recording
                if m_str == "SESSION_END":
                    print("SESSION_END received. Terminating recording.")
                    self.is_recording = False
                    break
                
                # Markers look like "FLASH_1_ABCDEF" (1 means Target, 0 means Non-Target)
                elif m_str.startswith("FLASH_"):
                    parts = m_str.split("_")
                    if len(parts) >= 3:
                        label = int(parts[1]) # 1 or 0
                        self.flash_events.append((timestamp, label))
            
            await asyncio.sleep(0.01)

    async def main_loop(self):
        """
        The main control loop that starts the listeners and waits for data.
        """
        print("Resolving LSL Streams...")
        loop = asyncio.get_running_loop()
        
        # Look for the Unicorn stream and the Speller stream on the network
        unicorn_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'UnicornRecorderLSLStream', 1, 10.0)
        marker_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'Speller_Markers', 1, 10.0)
        
        if not unicorn_streams or not marker_streams:
            print("ERROR: Could not find the streams. Is the headset and UI running?")
            return
            
        inlet = pylsl.StreamInlet(unicorn_streams[0], max_buflen=360)
        marker_inlet = pylsl.StreamInlet(marker_streams[0], max_buflen=120)
        
        self.is_recording = True
        print("\n[READY] Recording active. Waiting for UI markers...")
        
        # Run both workers at the same time!
        await asyncio.gather(
            self.lsl_worker(inlet),
            self.marker_worker(marker_inlet)
        )
        
        # Once recording stops, process and save everything
        self.epoch_and_save()
        
    def epoch_and_save(self):
        """
        The "Post-Processor". Matches brain waves to flashes and saves them.
        """
        print("Epoching data...")
        if not self.eeg_data or not self.flash_events:
            print("No data collected.")
            return
            
        X, y = [], []
        time_arr = np.array(self.eeg_times)
        data_arr = np.array(self.eeg_data)[:, :8] # Keep only 8 EEG channels
        
        # 1. Clean the entire signal at once (Filters)
        data_arr = apply_preprocessing(data_arr)
        
        # 2. Slice the signal into 800ms windows (Epoching)
        rejected_count = 0
        for f_time, label in self.flash_events:
            epoch = extract_epoch(data_arr, time_arr, f_time, apply_baseline=True)
            
            if epoch is None: continue
            
            # 3. Discard noisy epochs (Artifact Rejection)
            if not reject_artifacts(epoch, ARTIFACT_THRESHOLD_UV):
                rejected_count += 1
                continue
            
            X.append(epoch)
            y.append(label)
                
        # 4. Save to files that the AI can read later
        X = np.array(X)
        y = np.array(y)
        
        output_dir = os.path.dirname(os.path.abspath(__file__))
        training_dir = os.path.join(output_dir, "training_data")
        os.makedirs(training_dir, exist_ok=True)
        
        np.save(os.path.join(training_dir, "X_train.npy"), X)
        np.save(os.path.join(training_dir, "y_train.npy"), y)
        
        # Also save the RAW data so we can re-process it if we change settings later
        raw_dir = os.path.join(output_dir, "raw_session")
        os.makedirs(raw_dir, exist_ok=True)
        np.save(os.path.join(raw_dir, "eeg_continuous.npy"), data_arr)
        np.save(os.path.join(raw_dir, "eeg_timestamps.npy"), time_arr)
        flash_arr = np.array(self.flash_events, dtype=[('time', 'f8'), ('label', 'i4')])
        np.save(os.path.join(raw_dir, "flash_events.npy"), flash_arr)
        
        print(f"--- Data Collection Finalized ---")
        print(f"Saved {len(X)} clean epochs. {rejected_count} were too noisy and discarded.")

async def main():
    collector = DataCollector()
    await collector.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutdown requested.")
