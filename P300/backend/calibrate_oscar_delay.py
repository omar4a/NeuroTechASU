"""
OSCAR Delay Calibration Script.

After collecting training data, this script re-epochs the raw continuous data
with different OSCAR delay values (0ms to 500ms in 25ms steps) and evaluates
classification accuracy at each delay. The delay that produces the highest
AUC IS the true system latency.

Usage:
    python calibrate_oscar_delay.py

Requires: raw_session/eeg_continuous.npy, eeg_timestamps.npy, flash_events.npy
          (auto-saved by data_collection.py)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
    from pyriemann.spatialfilters import Xdawn
    from mne.decoding import Vectorizer
except ImportError:
    print("ERROR: pyriemann and mne are required. pip install pyriemann mne")
    sys.exit(1)

from signal_processing import (
    FS, SAMPLES_PER_EPOCH, BASELINE_SAMPLES, ARTIFACT_THRESHOLD_UV,
    reject_artifacts
)

output_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(output_dir, "raw_session")


def epoch_with_delay(data_arr, time_arr, flash_events, delay_s):
    """Re-epoch continuous data with a specific OSCAR delay compensation."""
    X, y = [], []
    
    for event in flash_events:
        f_time = float(event['time'])
        label = int(event['label'])
        
        adjusted_time = f_time + delay_s
        idx = np.searchsorted(time_arr, adjusted_time)
        
        if idx - BASELINE_SAMPLES < 0 or idx + SAMPLES_PER_EPOCH >= len(data_arr):
            continue
        
        baseline = data_arr[idx - BASELINE_SAMPLES : idx]
        epoch = data_arr[idx : idx + SAMPLES_PER_EPOCH]
        
        baseline_mean = np.mean(baseline, axis=0)
        epoch = epoch - baseline_mean
        epoch = epoch.T  # (channels, samples)
        
        if not reject_artifacts(epoch, ARTIFACT_THRESHOLD_UV):
            continue
        
        X.append(epoch)
        y.append(label)
    
    return np.array(X), np.array(y)


def evaluate_auc(X, y):
    """Quick cross-validated AUC evaluation."""
    if len(X) < 20 or np.sum(y == 1) < 5:
        return 0.0
    
    n_splits = min(5, np.min(np.bincount(y)))
    if n_splits < 2:
        return 0.0
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipe = make_pipeline(Xdawn(nfilter=2), Vectorizer(), LinearDiscriminantAnalysis())
    
    try:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        return scores.mean()
    except Exception:
        return 0.0


def main():
    # Load raw session data
    print("Loading raw session data...")
    eeg_path = os.path.join(raw_dir, "eeg_continuous.npy")
    ts_path = os.path.join(raw_dir, "eeg_timestamps.npy")
    ev_path = os.path.join(raw_dir, "flash_events.npy")
    
    for p in [eeg_path, ts_path, ev_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found!")
            print("Run data_collection.py first to generate raw session data.")
            return
    
    data_arr = np.load(eeg_path)
    time_arr = np.load(ts_path)
    flash_events = np.load(ev_path, allow_pickle=True)
    
    print(f"Continuous EEG: {data_arr.shape} ({data_arr.shape[0]/FS:.1f}s)")
    print(f"Flash events: {len(flash_events)}")
    print(f"Targets: {np.sum(flash_events['label'] == 1)}, "
          f"Non-targets: {np.sum(flash_events['label'] == 0)}")
    
    # Sweep delays from 0ms to 500ms in 25ms steps
    delays_ms = list(range(0, 525, 25))
    results = []
    
    print(f"\nSweeping {len(delays_ms)} delay values (0-500ms)...")
    print(f"{'Delay (ms)':>12} | {'Epochs':>8} | {'Targets':>8} | {'AUC':>8}")
    print("-" * 50)
    
    for delay_ms in delays_ms:
        delay_s = delay_ms / 1000.0
        X, y = epoch_with_delay(data_arr, time_arr, flash_events, delay_s)
        
        if len(X) == 0:
            auc = 0.0
            n_targ = 0
        else:
            n_targ = np.sum(y == 1)
            auc = evaluate_auc(X, y)
        
        results.append((delay_ms, len(X), n_targ, auc))
        marker = " <-- BEST" if auc == max(r[3] for r in results) and auc > 0.5 else ""
        print(f"{delay_ms:>12} | {len(X):>8} | {n_targ:>8} | {auc:>8.4f}{marker}")
    
    # Find best delay
    best = max(results, key=lambda r: r[3])
    best_delay_ms = best[0]
    best_auc = best[3]
    
    print(f"\n{'='*50}")
    print(f"OPTIMAL OSCAR DELAY: {best_delay_ms}ms (AUC = {best_auc:.4f})")
    print(f"{'='*50}")
    
    if best_auc < 0.55:
        print("\nWARNING: Even the best delay produces near-chance AUC.")
        print("This suggests the problem is NOT (only) OSCAR delay.")
        print("Check: electrode contact, user attention, training data size.")
    else:
        print(f"\nTo apply: set OSCAR_DELAY_S = {best_delay_ms/1000.0:.3f} in signal_processing.py")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 5))
    delays = [r[0] for r in results]
    aucs = [r[3] for r in results]
    
    ax.plot(delays, aucs, 'b-o', linewidth=2, markersize=5)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance level')
    ax.axvline(x=best_delay_ms, color='green', linestyle='--', alpha=0.7,
               label=f'Best: {best_delay_ms}ms (AUC={best_auc:.3f})')
    ax.set_xlabel('OSCAR Delay Compensation (ms)', fontsize=12)
    ax.set_ylabel('Cross-Validated ROC AUC', fontsize=12)
    ax.set_title('OSCAR Delay Calibration Sweep', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.35, max(0.8, best_auc + 0.1))
    
    plot_path = os.path.join(output_dir, "oscar_delay_calibration.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nCalibration plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
