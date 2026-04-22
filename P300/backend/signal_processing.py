"""
Shared Signal Processing Module for the P300 BCI Speller.

Centralizes all preprocessing, constants, and artifact rejection logic
used by both data_collection.py and realtime_inference.py.
Filter coefficients are pre-computed once at module load for performance.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# ============================================================================
# Hardware Constants — g.tec Unicorn Hybrid Black
# ============================================================================
FS = 250                            # Sampling rate (Hz) — hardware-locked
EPOCH_LEN = 0.6                     # Epoch duration (seconds) post-stimulus
                                    # Reduced from 0.8 — diagnostic showed 200-600ms
                                    # window outperforms full 800ms (less noise features)
SAMPLES_PER_EPOCH = int(FS * EPOCH_LEN)  # 150 samples
BASELINE_SAMPLES = int(FS * 0.1)         # 25 samples (100 ms pre-stimulus)

# Artifact rejection threshold (peak-to-peak µV per channel)
ARTIFACT_THRESHOLD_UV = 100.0

# OSCAR (Online Signal Conditioning and Artifact Removal) delay compensation.
# The Unicorn's OSCAR filter introduces processing latency that shifts EEG
# timestamps relative to marker timestamps. The exact delay depends on your
# firmware/config and MUST be measured empirically, not guessed.
#
# WORKFLOW:
#   1. Collect training data with OSCAR_DELAY_S = 0.0 (raw, no compensation)
#   2. Run calibrate_oscar_delay.py to sweep 0-500ms and find the true delay
#   3. Set OSCAR_DELAY_S to the calibrated value
#   4. Re-run data_collection.py or bci_classifiers.py with correct alignment
#
# Set to 0.0 if OSCAR is DISABLED in the Unicorn Suite.
OSCAR_DELAY_S = 0.125  # OSCAR is DISABLED — no delay compensation needed

# Unicorn Hybrid Black electrode montage (8 channels)
# Positions correspond to the 10-20 international system
UNICORN_CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
PZ_INDEX = 4  # Index of the Pz channel in the Unicorn montage

# 6x6 Farwell-Donchin character matrix (flattened for inference voting)
MATRIX_CHARS = [
    'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '_'
]

# 6x6 matrix grid (for UI)
MATRIX_GRID = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '_']
]

# ============================================================================
# Pre-computed Filter Coefficients (computed ONCE at module load)
# ============================================================================
_nyq = 0.5 * FS

# Bandpass: 0.5 – 30 Hz, 4th-order Butterworth
# Widened from 20Hz — 20Hz was too aggressive and attenuated high-frequency
# ERP components that carry discriminative information for the P300.
_bp_low = 0.5 / _nyq
_bp_high = 30.0 / _nyq
BP_B, BP_A = butter(4, [_bp_low, _bp_high], btype='band')

# Notch: 50 Hz (European mains), Q=30
_notch_freq = 50.0 / _nyq
NOTCH_B, NOTCH_A = iirnotch(_notch_freq, 30.0)


# ============================================================================
# Preprocessing Functions
# ============================================================================

def apply_preprocessing(data):
    """
    Apply bandpass (0.5-20 Hz) and 50 Hz notch filtering to continuous EEG data.
    Uses pre-computed filter coefficients for maximum performance.
    
    Parameters
    ----------
    data : np.ndarray, shape (samples, channels)
        Raw EEG data in time-major format.
    
    Returns
    -------
    np.ndarray : Filtered EEG data, same shape as input.
    """
    # Zero-phase bandpass filter
    filtered = filtfilt(BP_B, BP_A, data, axis=0)
    # Zero-phase notch filter
    filtered = filtfilt(NOTCH_B, NOTCH_A, filtered, axis=0)
    return filtered


def reject_artifacts(epoch, threshold_uv=ARTIFACT_THRESHOLD_UV):
    """
    Check if an epoch exceeds the artifact rejection threshold.
    Uses peak-to-peak amplitude per channel — if ANY channel exceeds
    the threshold, the entire epoch is rejected.
    
    Parameters
    ----------
    epoch : np.ndarray, shape (channels, samples) or (samples, channels)
        Single epoch of EEG data.
    threshold_uv : float
        Maximum allowed peak-to-peak amplitude in µV.
    
    Returns
    -------
    bool : True if the epoch is CLEAN (should be kept), False if it should be rejected.
    """
    # Ensure we compute peak-to-peak along the time axis
    if epoch.ndim == 2:
        # If shape is (channels, samples), compute along axis=1
        if epoch.shape[0] < epoch.shape[1]:
            ptp = np.ptp(epoch, axis=1)
        else:
            ptp = np.ptp(epoch, axis=0)
    else:
        ptp = np.array([np.ptp(epoch)])
    
    return np.all(ptp < threshold_uv)


def extract_epoch(data_arr, time_arr, flash_time, apply_baseline=True):
    """
    Extract a single epoch from continuous data given a flash timestamp.
    
    Compensates for OSCAR processing delay: the Unicorn's OSCAR filter delays
    EEG data by ~250ms relative to marker timestamps. Without compensation,
    the epoch captures [-250ms, +550ms] instead of [0ms, +800ms], shifting
    the P300 from its expected ~300ms position to ~550ms.
    
    Parameters
    ----------
    data_arr : np.ndarray, shape (samples, channels)
        Continuous EEG data (already preprocessed).
    time_arr : np.ndarray, shape (samples,)
        Corresponding timestamps.
    flash_time : float
        LSL timestamp of the flash onset.
    apply_baseline : bool
        Whether to apply baseline correction.
    
    Returns
    -------
    epoch : np.ndarray, shape (channels, samples) or None
        Baseline-corrected epoch transposed to (channels, samples).
        Returns None if insufficient data for the epoch window.
    """
    # Compensate for OSCAR delay: the EEG sample that was actually recorded
    # at flash_time has timestamp (flash_time + OSCAR_DELAY_S) in the stream,
    # because OSCAR delayed it before transmission.
    adjusted_time = flash_time + OSCAR_DELAY_S
    
    idx = np.searchsorted(time_arr, adjusted_time)
    
    # Check bounds: need BASELINE_SAMPLES before and SAMPLES_PER_EPOCH after
    if idx - BASELINE_SAMPLES < 0 or idx + SAMPLES_PER_EPOCH >= len(data_arr):
        return None
    
    baseline = data_arr[idx - BASELINE_SAMPLES : idx]
    epoch = data_arr[idx : idx + SAMPLES_PER_EPOCH]
    
    if apply_baseline:
        baseline_mean = np.mean(baseline, axis=0)
        epoch = epoch - baseline_mean
    
    # Transpose to (channels, samples) — expected by pyriemann/sklearn pipelines
    return epoch.T
