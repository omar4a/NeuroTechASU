from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

class QualityEngine:
    def __init__(self, sampling_rate: float = 250.0, debounce_seconds: int = 3) -> None:
        self.fs = sampling_rate
        self.scale_factor = 4_500_000.0 / 8_388_608.0
        self.railing_threshold_uv = 180_000.0
        
        # 1. Bandpass filter to isolate 50 Hz powerline noise (The "Antenna" test)
        self.sos_50hz = butter(4, [48, 52], btype='bandpass', fs=self.fs, output='sos')
        
        # 2. Highpass filter to remove DC offset for checking actual brainwave drift
        self.sos_clean = butter(4, 1.0, btype='highpass', fs=self.fs, output='sos')

        # Debouncing state machine
        self.history_length = debounce_seconds
        self.channel_history: dict[str, list[bool]] = {}
        self.channel_states: dict[str, bool] = {}

    def process_window(self, channel_label: str, raw_counts: list[float] | np.ndarray) -> dict[str, float | bool]:
        """
        Input: Channel label and a 1D array of raw ADC counts.
        Output: Debounced quality metrics.
        """
        data_uv = np.asarray(raw_counts, dtype=float) * self.scale_factor
        offset_uv = float(np.mean(data_uv))
        
        # --- 1. HARDWARE FAULT CHECKS ---
        # Is the signal clamped to the maximum/minimum limits, or completely dead?
        is_railed = abs(offset_uv) > self.railing_threshold_uv
        is_flatline = float(np.ptp(data_uv)) < 0.1
        
        if is_railed or is_flatline:
            instantaneous_good = False
            noise_rms = 999.0
        else:
            # --- 2. THE ANTENNA TEST (50 Hz Noise) ---
            # Extract only the powerline noise. High RMS means bad impedance/contact.
            noise_50hz = sosfiltfilt(self.sos_50hz, data_uv)
            
            # Calculate Root Mean Square (RMS) of the noise
            # Formula: $$ RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2} $$
            noise_rms = float(np.sqrt(np.mean(noise_50hz**2)))
            
            # --- 3. THE DRIFT TEST ---
            # Strip the DC offset to see the actual signal drift
            clean_data = sosfiltfilt(self.sos_clean, data_uv)
            v_pp_clean = float(np.ptp(clean_data))
            
            # Evaluation: Good contact has low 50 Hz noise and reasonable peak-to-peak drift
            # (You may need to tweak these specific thresholds based on your environment)
            is_noise_low = noise_rms < 15.0
            is_drift_reasonable = v_pp_clean < 250.0
            
            instantaneous_good = is_noise_low and is_drift_reasonable

        # --- 4. DEBOUNCING (Anti-Flicker Logic) ---
        if channel_label not in self.channel_history:
            # Initialize the channel with its first reading
            self.channel_history[channel_label] = [instantaneous_good] * self.history_length
            self.channel_states[channel_label] = instantaneous_good
            
        history = self.channel_history[channel_label]
        history.pop(0)
        history.append(instantaneous_good)
        
        # Only switch the official state if ALL recent checks agree
        if all(history):
            self.channel_states[channel_label] = True
        elif not any(history):
            self.channel_states[channel_label] = False
            
        final_good_state = self.channel_states[channel_label]

        return {
            "is_good": final_good_state,
            "display_uv": noise_rms,  # Passing 50 Hz RMS back as a useful debugging metric
        }