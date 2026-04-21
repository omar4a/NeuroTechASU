import os
import numpy as np
import scipy.signal
import pylsl
import asyncio
import websockets
import json

try:
    from pyriemann.spatialfilters import Xdawn
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.classification import MDM
    from mne.decoding import Vectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
except ImportError:
    print("Warning: mne, pyriemann or sklearn not installed.")

FS = 250
EPOCH_LEN = 0.8  # 800 ms
SAMPLES_PER_EPOCH = int(FS * EPOCH_LEN)

matrixChars = [
    'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '_'
]

def butter_bandpass_filter(data, lowcut=1.0, highcut=20.0, fs=250.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.filtfilt(b, a, data, axis=-1)
    return y

class RealTimeInference:
    def __init__(self):
        self.eeg_data = []
        self.eeg_times = []
        
        self.current_trial_flashes = []
        self.is_running = True
        
        self.active_algo = "xdawn_lda"
        self.model_lda = None
        self.model_mdm = None
        
        # Create an LSL stream for our Web UI markers
        info = pylsl.StreamInfo('Speller_Markers', 'Markers', 1, 0, 'string', 'my_alienware_marker_123')
        self.marker_outlet = pylsl.StreamOutlet(info)
        print("LSL Marker Outlet created for Inference.")
        
    def load_and_train_model(self):
        print("Loading training data to train the live classifier...")
        output_dir = os.path.dirname(os.path.abspath(__file__))
        x_path = os.path.join(output_dir, "X_train.npy")
        y_path = os.path.join(output_dir, "y_train.npy")
        
        if not os.path.exists(x_path):
            raise FileNotFoundError("X_train.npy required to train the real-time model. Do data collection first.")
            
        X = np.load(x_path)
        y = np.load(y_path)
        
        X_filt = butter_bandpass_filter(X)
        print("Training xDAWN + LDA Pipeline...")
        self.model_lda = make_pipeline(
            Xdawn(nfilter=3),
            Vectorizer(),
            LinearDiscriminantAnalysis()
        )
        self.model_lda.fit(X_filt, y)
        
        print("Training Riemannian MDM Pipeline...")
        self.model_mdm = make_pipeline(
            XdawnCovariances(nfilter=3, estimator="oas"),
            MDM()
        )
        self.model_mdm.fit(X_filt, y)
        print("Models Trained Successfully.")

    async def lsl_worker(self, inlet):
        print("Connected to Unicorn stream. Listening for data...")
        while self.is_running:
            chunk, timestamps = inlet.pull_chunk()
            if timestamps:
                self.eeg_data.extend(chunk)
                self.eeg_times.extend(timestamps)
                
                # Keep buffer magnitude manageable (keep last 60 seconds)
                if len(self.eeg_times) > FS * 60:
                    self.eeg_data = self.eeg_data[-FS * 60:]
                    self.eeg_times = self.eeg_times[-FS * 60:]
                    
            await asyncio.sleep(0.01)

    async def marker_worker(self, marker_inlet):
        print("Connected to Marker stream. Receiving markers...")
        while self.is_running:
            marker, timestamp = marker_inlet.pull_sample(timeout=0.0)
            if timestamp is not None and marker is not None:
                m_str = marker[0]
                if m_str.startswith("FLASH_"):
                    # Format: FLASH_GROUP_ABC
                    parts = m_str.split("_")
                    if len(parts) >= 3:
                        group = list(parts[2])
                        self.current_trial_flashes.append((timestamp, group))
            await asyncio.sleep(0.01)

    async def ws_handler(self, websocket, path=None):
        print("Speller UI Connected for INFERENCE!")
        
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
        self.is_running = True
        lsl_task = asyncio.create_task(self.lsl_worker(inlet))
        marker_task = asyncio.create_task(self.marker_worker(marker_inlet))
        
        # Tell UI to start a freestyle session with 5 letters
        await websocket.send(json.dumps({
            "command": "start_inference",
            "num_trials": 5
        }))
        
        try:
            async for message in websocket:
                data = json.loads(message)
                event = data.get("event")
                
                if event == "inference_session_start":
                    self.active_algo = data.get("algorithm", "xdawn_lda")
                    print(f"Inference Session Started. Algorithm: {self.active_algo}")
                    
                elif event == "flash":
                    group = data.get("target_group", [])
                    marker_string = f"FLASH_GROUP_{''.join(group)}"
                    self.marker_outlet.push_sample([marker_string], pylsl.local_clock())
                    
                elif event == "evaluate_block":
                    predicted_char, hit_threshold = self.decode_trial(check_threshold=True)
                    if hit_threshold and predicted_char:
                        print(f"*** DYNAMIC STOP *** PREDICTED: {predicted_char} (Confidence > 0.85)")
                        await websocket.send(json.dumps({
                            "command": "type_char",
                            "char": predicted_char
                        }))
                        self.current_trial_flashes = []
                    
                elif event == "inference_done":
                    print("Trial ended (Max blocks reached). Decoding final fallback signal...")
                    await asyncio.sleep(0.9)
                    
                    predicted_char, _ = self.decode_trial(check_threshold=False)
                    if predicted_char:
                        print(f"---------> FINAL PREDICTED CHARACTER: {predicted_char} <---------")
                        await websocket.send(json.dumps({
                            "command": "type_char",
                            "char": predicted_char
                        }))
                    else:
                        print("Failed to decode trial.")
                        
                    self.current_trial_flashes = []

        except websockets.exceptions.ConnectionClosed:
            print("UI Disconnected.")
            
        self.is_running = False
        await asyncio.gather(lsl_task, marker_task)

    def decode_trial(self, check_threshold=False):
        if not self.current_trial_flashes:
            return None, False
            
        if not self.eeg_data:
            print("ERROR: No EEG data in buffer! Is the Unicorn Suite actively broadcasting?")
            return None, False
            
        X_test = []
        groups = []
        
        time_arr = np.array(self.eeg_times)
        data_arr = np.array(self.eeg_data)
        
        if data_arr.ndim < 2:
            print("ERROR: EEG data array is malformed or empty.")
            return None, False
            
        data_arr = data_arr[:, :8] # Get 8 channels
        
        # Epoch our flashes
        for f_time, group in self.current_trial_flashes:
            idx = np.searchsorted(time_arr, f_time)
            
            if idx + SAMPLES_PER_EPOCH < len(data_arr):
                epoch = data_arr[idx : idx + SAMPLES_PER_EPOCH].T
                X_test.append(epoch)
                groups.append(group)
                
        if not X_test:
            print("No valid epochs found in buffer!")
            return None, False
            
        X_test = np.array(X_test)
        X_test_filt = butter_bandpass_filter(X_test)
        
        # Get target probability scores from chosen algorithm
        if self.active_algo == "riemann_mdm":
            y_probs = self.model_mdm.predict_proba(X_test_filt)[:, 1]
        else:
            y_probs = self.model_lda.predict_proba(X_test_filt)[:, 1]
        
        # Bayesian probability aggregation
        char_probs = {c: 1.0/len(matrixChars) for c in matrixChars}
        
        for i, group in enumerate(groups):
            p_target = y_probs[i]
            for c in matrixChars:
                if c in group:
                    char_probs[c] *= p_target
                else:
                    char_probs[c] *= (1.0 - p_target)
                    
        # Normalize sum
        total_p = sum(char_probs.values())
        if total_p == 0:
            return None, False
        for c in char_probs:
            char_probs[c] /= total_p
        
        # Best character is the one with highest posterior cumulative probability
        best_char = max(char_probs.items(), key=lambda x: x[1])
        
        if check_threshold:
            if best_char[1] >= 0.95:
                return best_char[0], True
            return None, False
            
        return best_char[0], True

async def main():
    agent = RealTimeInference()
    agent.load_and_train_model()
    
    print("Backend WebSocket Server listening on ws://localhost:8765")
    async with websockets.serve(agent.ws_handler, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown.")
