import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, iirnotch
import os

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Using lfilter for basic directional filtering
    y = lfilter(b, a, data)
    return y

def notch_filter(data, fs, freq0=60.0, Q=30.0):
    """Apply a notch filter to remove powerline noise."""
    nyq = 0.5 * fs
    w0 = freq0 / nyq
    b, a = iirnotch(w0, Q)
    y = lfilter(b, a, data)
    return y

def process_eeg_file(input_csv, output_csv, fs=250.0):
    """Read raw EEG CSV, apply DSP pipeline, and save to new CSV."""
    df = pd.read_csv(input_csv)
    
    # Identify EEG columns (Unicorn typically uses 'EEG 1', 'EEG 2', etc.)
    eeg_cols = [col for col in df.columns if 'EEG' in col.upper()]
    
    # Fallback to the first 8 columns if no implicit header matches
    if not eeg_cols:
        eeg_cols = df.columns[:8]
        
    processed_df = df.copy()
    
    for col in eeg_cols:
        raw_signal = df[col].astype(float).values
        
        # Step 1: Band-pass filter (0.5 to 45 Hz) 
        # This removes the massive DC offset (0 Hz) and high freq noise
        bp_signal = bandpass_filter(raw_signal, lowcut=0.5, highcut=45.0, fs=fs, order=4)
        
        # Step 2: Notch filter (60 Hz powerline interference)
        clean_signal = notch_filter(bp_signal, fs=fs, freq0=60.0)
        
        # Update the dataframe with the clean microvolt values
        processed_df[col] = clean_signal
        
    processed_df.to_csv(output_csv, index=False)
    print(f"Success: Processed {input_csv} -> {output_csv}")

if __name__ == "__main__":
    base_dir = r"c:\Omar\Education\NeuroTech_ASU\Signal Quality Algorithm\Unicorn_Recordings"
    
    raw_files = [
        "UnicornRawDataRecorder_1.csv", 
        "UnicornRawDataRecorder_2.csv"
    ]
    
    for raw_file in raw_files:
        input_path = os.path.join(base_dir, raw_file)
        # Create output file name indicating it is now in target microvolts
        output_name = raw_file.replace(".csv", "_MicrovoltsFiltered.csv")
        output_path = os.path.join(base_dir, output_name)
        
        if os.path.exists(input_path):
            print(f"Applying DSP pipeline to {raw_file}...")
            process_eeg_file(input_path, output_path)
        else:
            print(f"File not found: {input_path}")
