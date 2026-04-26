import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

def main():
    """
    mock_lsl_stream.py (Testing Environment)
    Initializes a mock LSL stream for testing the inference pipeline
    without needing the actual hardware connected.
    """
    print("Initializing UnicornMock LSL stream...")
    # 8 channels, 250 Hz, float32
    info = StreamInfo('UnicornMock', 'EEG', 8, 250, 'float32', 'unicorn_mock_123')
    outlet = StreamOutlet(info)
    
    print("Streaming random noise. Press Ctrl+C to stop.")
    try:
        while True:
            # Generate 8 channels of random noise
            sample = np.random.rand(8).tolist()
            outlet.push_sample(sample)
            
            # Push every 4 milliseconds to simulate 250 Hz
            time.sleep(0.004)
    except KeyboardInterrupt:
        print("Stopping mock stream.")

if __name__ == '__main__':
    main()
