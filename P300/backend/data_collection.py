"""
P300 Training Data Collection Backend.

Connects to:
  1. Unicorn EEG stream (8ch @ 250Hz) via LSL
  2. Speller_Markers stream from PsychoPy UI via LSL

Runs two async workers to buffer EEG data and flash markers concurrently.
On SESSION_END, epochs the continuous data around flash events, applies
preprocessing + artifact rejection, and saves X_train.npy / y_train.npy.
"""

import os
import sys
# PsychoPy Runner isolates the environment, stripping standard user site-packages.
# We append the standard Python 3.10 user site (where pyriemann/mne/sklearn live)
# to sys.path. We APPEND so that PsychoPy's own libraries take priority.
_user_site = os.path.expanduser(r"~\AppData\Roaming\Python\Python310\site-packages")
if os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.append(_user_site)

import numpy as np
import pylsl
import asyncio

from signal_processing import (
    FS, SAMPLES_PER_EPOCH, BASELINE_SAMPLES,
    apply_preprocessing, reject_artifacts, extract_epoch
)


class DataCollector:
    def __init__(self):
        self.eeg_data = []
        self.eeg_times = []
        self.flash_events = []
        self.is_recording = False
        
    async def lsl_worker(self, inlet):
        print("Connected to Unicorn stream. Receiving data...")
        while self.is_recording:
            chunk, timestamps = inlet.pull_chunk()
            if timestamps:
                self.eeg_data.extend(chunk)
                self.eeg_times.extend(timestamps)
            # Sleep briefly to yield loop
            await asyncio.sleep(0.01)
            
    async def marker_worker(self, marker_inlet):
        print("Connected to Marker stream. Receiving markers...")
        while self.is_recording:
            marker, timestamp = marker_inlet.pull_sample(timeout=0.0)
            if timestamp is not None and marker is not None:
                # the marker is a list of 1 string e.g., ["FLASH_1_ABCDEF"]
                m_str = marker[0]
                if m_str == "SESSION_END":
                    print("SESSION_END received. Terminating recording.")
                    self.is_recording = False
                    break
                elif m_str.startswith("FLASH_"):
                    # Extract label 
                    parts = m_str.split("_")
                    if len(parts) >= 3:
                        label = int(parts[1])
                        self.flash_events.append((timestamp, label))
            await asyncio.sleep(0.01)

    async def main_loop(self):
        print("Resolving LSL Streams...")
        loop = asyncio.get_running_loop()
        unicorn_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'UnicornRecorderLSLStream', 1, 10.0)
        marker_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'Speller_Markers', 1, 10.0)
        
        if not unicorn_streams:
            print("ERROR: Could not find Unicorn stream within 10 seconds.")
            return
            
        if not marker_streams:
            print("ERROR: Could not find Speller_Markers stream in 10 seconds.")
            print("Make sure psychopy_speller.py is running.")
            return
            
        inlet = pylsl.StreamInlet(unicorn_streams[0])
        marker_inlet = pylsl.StreamInlet(marker_streams[0])
        
        self.is_recording = True
        print("\n[READY] Recording active. Waiting for UI markers...")
        
        await asyncio.gather(
            self.lsl_worker(inlet),
            self.marker_worker(marker_inlet)
        )
        
        self.epoch_and_save()
        
    def epoch_and_save(self):
        print("Epoching data...")
        if not self.eeg_data or not self.flash_events:
            print("No data collected.")
            return
            
        X, y = [], []
        time_arr = np.array(self.eeg_times)
        data_arr = np.array(self.eeg_data)  # Format: (Samples, Channels)
        
        # Keep only the first 8 channels (EEG) from the Unicorn dataset
        data_arr = data_arr[:, :8]
        
        # Apply preprocessing (Bandpass + Notch) to continuous signal to prevent edge artifacts
        data_arr = apply_preprocessing(data_arr)
        
        rejected_count = 0
        for f_time, label in self.flash_events:
            epoch = extract_epoch(data_arr, time_arr, f_time, apply_baseline=True)
            
            if epoch is None:
                continue
            
            # Artifact rejection — skip epochs with extreme amplitudes (e.g., blinks)
            if not reject_artifacts(epoch):
                rejected_count += 1
                continue
            
            X.append(epoch)
            y.append(label)
                
        X = np.array(X)
        y = np.array(y)
        
        output_dir = os.path.dirname(os.path.abspath(__file__))
        x_path = os.path.join(output_dir, "X_train.npy")
        y_path = os.path.join(output_dir, "y_train.npy")
        
        np.save(x_path, X)
        np.save(y_path, y)
        
        # Save raw continuous data for offline re-epoching with different delay values.
        # This allows empirical measurement of the true OSCAR delay rather than guessing.
        raw_dir = os.path.join(output_dir, "raw_session")
        os.makedirs(raw_dir, exist_ok=True)
        np.save(os.path.join(raw_dir, "eeg_continuous.npy"), data_arr)  # Already 8ch, preprocessed
        np.save(os.path.join(raw_dir, "eeg_timestamps.npy"), time_arr)
        # Flash events: list of (timestamp, label) tuples
        flash_arr = np.array(self.flash_events, dtype=[('time', 'f8'), ('label', 'i4')])
        np.save(os.path.join(raw_dir, "flash_events.npy"), flash_arr)
        print(f"Saved raw session data to {raw_dir}/ for delay calibration.")
        
        print(f"--- Data Collection Finalized ---")
        print(f"Saved {x_path} (Shape: {X.shape})")
        print(f"Saved {y_path} (Shape: {y.shape})")
        if rejected_count > 0:
            print(f"Artifact rejection: {rejected_count} epochs rejected (threshold: {int(ARTIFACT_THRESHOLD_UV)}uV)")

async def main():
    collector = DataCollector()
    await collector.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutdown requested.")
