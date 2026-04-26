import threading
import time
from collections import deque
import numpy as np
import scipy.signal as signal
from pylsl import StreamInlet, resolve_byprop
import joblib
import serial

# --- System Configuration ---
SFREQ = 250
BUFFER_SECONDS = 4.0
WINDOW_SAMPLES = int(BUFFER_SECONDS * SFREQ) # 1000 samples

SLICE_SECONDS = 2.0
SLICE_SAMPLES = int(SLICE_SECONDS * SFREQ) # 500 samples
INFERENCE_INTERVAL = 0.25 # 250 ms (exactly 62.5 samples)

MODEL_PATH = 'eeg_mi_pipeline.pkl'
SERIAL_PORT = 'COM3' # User should assign this properly
BAUD_RATE = 115200

# Thread-safe circular buffer (8 channels, 1000 samples)
buffer_lock = threading.Lock()
ring_buffer = np.zeros((8, WINDOW_SAMPLES))

def lsl_worker():
    """Background thread to continuously populate the buffer so the stream never stalls."""
    global ring_buffer
    print("[LSL] Resolving EEG stream...")
    
    # Find the LSL stream. Change 'type' to 'name' if you want to strictly match 'UnicornMock'
    streams = resolve_byprop('type', 'EEG')
    inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=1)
    print(f"[LSL] Connected to stream: {streams[0].name()}")
    
    while True:
        # Pull incoming chunk of data
        chunk, timestamps = inlet.pull_chunk()
        if chunk:
            chunk = np.array(chunk).T # Transpose to (n_channels, n_samples)
            
            # Match channel count to ring_buffer (e.g., take first 8 channels if stream has 9)
            n_channels_to_copy = min(chunk.shape[0], ring_buffer.shape[0])
            chunk = chunk[:n_channels_to_copy, :]
            
            n_new = chunk.shape[1]
            
            with buffer_lock:
                if n_new >= WINDOW_SAMPLES:
                    # If chunk is larger than buffer, keep latest
                    ring_buffer[:n_channels_to_copy, :] = chunk[:, -WINDOW_SAMPLES:]
                else:
                    # Roll buffer left and append new samples
                    ring_buffer[:n_channels_to_copy, :-n_new] = ring_buffer[:n_channels_to_copy, n_new:]
                    ring_buffer[:n_channels_to_copy, -n_new:] = chunk

def main():
    """
    realtime_inference.py (The Core Engine)
    Asynchronous loop that pulls live data, runs inferences, and handles serial hardware communication.
    """
    global ring_buffer

    # 1. Load Trained Pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("[Engine] Pipeline loaded successfully.")
    except Exception as e:
        print(f"[Engine] WARNING: Failed to load pipeline. Running in pass-through mock mode. Error: {e}")
        pipeline = None

    # 2. Initialize Serial Connection
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
        print(f"[Serial] Connected to port {SERIAL_PORT}")
    except Exception as e:
        print(f"[Serial] WARNING: Failed to open serial port {SERIAL_PORT}. Error: {e}")
        ser = None

    # 3. Precompute Butterworth Filter Coefficients (4th-order IIR, 7-30Hz)
    nyq = 0.5 * SFREQ
    low = 7.0 / nyq
    high = 30.0 / nyq
    b, a = signal.butter(4, [low, high], btype='band')

    # 4. Start LSL Background Thread
    lsl_thread = threading.Thread(target=lsl_worker, daemon=True)
    lsl_thread.start()

    # 5. Inference State Variables
    prediction_queue = deque(maxlen=3) # Algorithmic Debouncing queue
    target_hardware_state = -1 # Starts at invalid state
    
    print("[Engine] Starting inference loop...")
    
    while True:
        start_time = time.time()
        
        # --- Data Slicing ---
        with buffer_lock:
            # Slice the most recent 2.0 seconds (500 samples)
            data_slice = ring_buffer[:, -SLICE_SAMPLES:].copy()
            
        # Ensure the buffer is actually receiving data
        if not np.all(data_slice == 0):
            # --- Live Preprocessing ---
            # 1. Apply Common Average Reference (CAR)
            car_mean = np.mean(data_slice, axis=0)
            data_slice = data_slice - car_mean
            
            # 2. Apply 4th-order IIR Butterworth bandpass filter
            data_filtered = signal.filtfilt(b, a, data_slice, axis=1)
            
            # 3. Reshape array to (1, 8, 500) for scikit-learn
            X = data_filtered.reshape(1, 8, SLICE_SAMPLES)
            
            # --- Inference & Thresholding ---
            if pipeline is not None:
                probs = pipeline.predict_proba(X)[0]
                max_prob = np.max(probs)
                pred_class = np.argmax(probs)
                
                # Hardcoded Rule: probability must be > 0.70
                if max_prob <= 0.70:
                    pred_class = 0 # Override to Class 0 (Rest)
            else:
                pred_class = 0 # Mock prediction if pipeline missing
                
            # --- Algorithmic Debouncing ---
            prediction_queue.append(pred_class)
            
            # Hardcoded Rule: Only update target_hardware_state if all 3 ints in queue are identical
            if len(prediction_queue) == 3 and len(set(prediction_queue)) == 1:
                new_state = prediction_queue[0]
                
                if new_state != target_hardware_state:
                    target_hardware_state = new_state
                    
                    # --- Serial Output ---
                    if ser and ser.is_open:
                        if target_hardware_state == 0:
                            ser.write(b'\x00') # Rest
                            state_str = "RESTING"
                        elif target_hardware_state == 1:
                            ser.write(b'\x01') # Open (Left MI)
                            state_str = "OPENING FIST"
                        elif target_hardware_state == 2:
                            ser.write(b'\x02') # Close (Right MI)
                            state_str = "CLOSING FIST"
                    else:
                        # Map state to string for console output when serial is disconnected
                        state_str = ["RESTING", "OPENING FIST", "CLOSING FIST"][target_hardware_state]
                        
                    # Standard Out transmission (Parsed by main_dashboard.py)
                    print(f"STATE: {state_str}", flush=True)

        # Sleep to strictly enforce the 0.25s interval
        elapsed = time.time() - start_time
        sleep_time = INFERENCE_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()
