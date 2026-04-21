import os
import numpy as np
import pylsl
import asyncio
import websockets
import json

FS = 250
EPOCH_LEN = 0.8  # 800 ms
SAMPLES_PER_EPOCH = int(FS * EPOCH_LEN)

class DataCollector:
    def __init__(self):
        self.eeg_data = []
        self.eeg_times = []
        self.flash_events = []
        self.target_word = "TECHNOLOGY"
        self.is_recording = False
        
        # Create an LSL stream for our Web UI markers
        info = pylsl.StreamInfo('Speller_Markers', 'Markers', 1, 0, 'string', 'my_alienware_marker_123')
        self.marker_outlet = pylsl.StreamOutlet(info)
        print("LSL Marker Outlet created successfully.")
        
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
                if m_str.startswith("FLASH_"):
                    # Extract label 
                    parts = m_str.split("_")
                    if len(parts) >= 3:
                        label = int(parts[1])
                        self.flash_events.append((timestamp, label))
            await asyncio.sleep(0.01)
            
    async def ws_handler(self, websocket, path=None):
        print("Speller UI Connected! Starting data collection...")
        
        # 1. Resolve LSL before allowing UI to start
        print("Resolving Unicorn LSL and Marker streams...")
        loop = asyncio.get_running_loop()
        unicorn_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'UnicornRecorderLSLStream', 1, 5.0)
        marker_streams = await loop.run_in_executor(None, pylsl.resolve_byprop, 'name', 'Speller_Markers', 1, 5.0)
        
        if not unicorn_streams:
            print("ERROR: Could not find Unicorn stream within 5 seconds.")
            print("Aborting WebSocket session. Please ensure Unicorn Suite is broadcasting first!!!")
            return
            
        if not marker_streams:
            print("ERROR: Could not find our LSL marker stream. This should not happen!")
            return
            
        inlet = pylsl.StreamInlet(unicorn_streams[0])
        marker_inlet = pylsl.StreamInlet(marker_streams[0])
        
        # 2. Everything is good, start recording!
        self.is_recording = True
        lsl_task = asyncio.create_task(self.lsl_worker(inlet))
        marker_task = asyncio.create_task(self.marker_worker(marker_inlet))
        
        # Tell UI to begin spelling
        await websocket.send(json.dumps({
            "command": "start_spelling",
            "word": self.target_word
        }))
        
        try:
            async for message in websocket:
                data = json.loads(message)
                event = data.get("event")
                
                if event == "experiment_start":
                    print(f"UI acknowledged experiment start. Target word: {data.get('word')}")
                    
                elif event == "flash":
                    group = data.get("target_group", [])
                    current_target = data.get("current_target")
                    
                    # Target tagging logic
                    label = 1 if current_target in group else 0
                    
                    # Instantly push to LSL Marker outlet
                    marker_string = f"FLASH_{label}_{''.join(group)}"
                    self.marker_outlet.push_sample([marker_string], pylsl.local_clock())
                    print(f"Pushed Marker -> Target: {current_target} | Group: {group} | Tag (y): {label}")
                    
                elif event == "experiment_complete":
                    print("Experiment complete flag received from UI. Concluding session...")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print("UI Disconnected or Experiment finished.")
            
        self.is_recording = False
        await asyncio.gather(lsl_task, marker_task)
        
        self.epoch_and_save()
        
    def epoch_and_save(self):
        print("Epoching data...")
        if not self.eeg_data or not self.flash_events:
            print("No data collected.")
            return
            
        X, y = [], []
        time_arr = np.array(self.eeg_times)
        data_arr = np.array(self.eeg_data) # Format: (Samples, Channels)
        
        # Keep only the first 8 channels (EEG) from the Unicorn dataset
        data_arr = data_arr[:, :8]
        
        for f_time, label in self.flash_events:
            idx = np.searchsorted(time_arr, f_time)
            
            # Ensure we have enough data past the flash for a full epoch
            if idx + SAMPLES_PER_EPOCH < len(data_arr):
                epoch = data_arr[idx : idx + SAMPLES_PER_EPOCH]
                
                # Model pipelines generally expect (Epochs, Channels, Samples)
                epoch = epoch.T
                
                X.append(epoch)
                y.append(label)
                
        X = np.array(X)
        y = np.array(y)
        
        output_dir = os.path.dirname(os.path.abspath(__file__))
        x_path = os.path.join(output_dir, "X_train.npy")
        y_path = os.path.join(output_dir, "y_train.npy")
        
        np.save(x_path, X)
        np.save(y_path, y)
        print(f"--- Data Collection Finalized ---")
        print(f"Saved {x_path} (Shape: {X.shape})")
        print(f"Saved {y_path} (Shape: {y.shape})")

async def main():
    collector = DataCollector()
    print("Backend WebSocket Server listening on ws://localhost:8765")
    async with websockets.serve(collector.ws_handler, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutdown requested.")
