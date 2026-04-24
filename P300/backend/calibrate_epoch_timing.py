"""
Epoch Timing Calibration Script.

Sweeps the epoch start offset (relative to flash onset) in 10ms steps to find
the optimal extraction window for P300 classification. The P300 component can
peak anywhere from ~200ms to 600ms post-stimulus depending on the subject,
hardware, and task. This script finds the best alignment empirically.

Usage:
    python calibrate_epoch_timing.py

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
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.classification import MDM
    from mne.decoding import Vectorizer
except ImportError:
    print("ERROR: pyriemann and mne are required. pip install pyriemann mne")
    sys.exit(1)

from signal_processing import (
    FS, BASELINE_SAMPLES, ARTIFACT_THRESHOLD_UV, reject_artifacts
)

output_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(output_dir, "raw_session")

# Sweep parameters
OFFSET_MIN_MS = 50       # Start from flash onset
OFFSET_MAX_MS = 300     # Max start offset (ms after flash)
OFFSET_STEP_MS = 10     # 50ms resolution to speed up grid search
EPOCH_DURATION_MS = 800  # Fixed epoch duration (ms)


def epoch_with_offset(data_arr, time_arr, flash_events, offset_s, epoch_samples):
    """Extract epochs with a specific start offset from flash onset."""
    X, y = [], []

    for event in flash_events:
        f_time = float(event['time'])
        label = int(event['label'])

        # Shift the epoch start by offset_s AFTER the flash
        start_time = f_time + offset_s
        idx = np.searchsorted(time_arr, start_time)

        if idx - BASELINE_SAMPLES < 0 or idx + epoch_samples >= len(data_arr):
            continue

        baseline = data_arr[idx - BASELINE_SAMPLES : idx]
        epoch = data_arr[idx : idx + epoch_samples]

        baseline_mean = np.mean(baseline, axis=0)
        epoch = epoch - baseline_mean
        epoch = epoch.T  # (channels, samples)

        if not reject_artifacts(epoch, ARTIFACT_THRESHOLD_UV):
            continue

        X.append(epoch)
        y.append(label)

    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)


def evaluate_auc(X, y, pipeline_type='mdm'):
    """Cross-validated AUC evaluation."""
    if len(X) < 20 or np.sum(y == 1) < 5:
        return 0.0

    n_splits = min(5, np.min(np.bincount(y)))
    if n_splits < 2:
        return 0.0

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if pipeline_type == 'mdm':
        pipe = make_pipeline(
            XdawnCovariances(nfilter=2, estimator='oas'),
            MDM()
        )
    else:  # lda
        pipe = make_pipeline(
            Xdawn(nfilter=2),
            Vectorizer(),
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9)
        )

    try:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        return scores.mean()
    except Exception:
        return 0.0


def main():
    print("=" * 60)
    print("  EPOCH TIMING CALIBRATION")
    print("  Sweeping epoch start offset to find optimal P300 window")
    print("=" * 60)

    # Load raw session data
    print("\nLoading raw session data...")
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

    if len(flash_events) > 1000:
        print(f"\nDownsampling {len(flash_events)} flashes to 1000 to speed up calibration...")
        np.random.seed(42)
        idx = np.random.choice(len(flash_events), 1000, replace=False)
        idx.sort()
        flash_events = flash_events[idx]
        
    epoch_samples = int(EPOCH_DURATION_MS / 1000.0 * FS)
    offsets_ms = list(range(OFFSET_MIN_MS, OFFSET_MAX_MS + 1, OFFSET_STEP_MS))

    print(f"\nSweeping {len(offsets_ms)} offset values ({OFFSET_MIN_MS}-{OFFSET_MAX_MS}ms, step={OFFSET_STEP_MS}ms)")
    print(f"Epoch duration: {EPOCH_DURATION_MS}ms ({epoch_samples} samples)")
    print(f"Each test epochs window: [offset, offset + {EPOCH_DURATION_MS}ms]")
    print()
    print(f"{'Offset (ms)':>12} | {'Window':>18} | {'Epochs':>8} | {'Targets':>8} | {'MDM AUC':>10} | {'LDA AUC':>10}")
    print("-" * 80)

    results = []

    for offset_ms in offsets_ms:
        offset_s = offset_ms / 1000.0
        X, y = epoch_with_offset(data_arr, time_arr, flash_events, offset_s, epoch_samples)

        if len(X) == 0:
            results.append((offset_ms, 0, 0, 0.0, 0.0))
            print(f"{offset_ms:>12} | {'N/A':>18} | {0:>8} | {0:>8} | {'N/A':>10} | {'N/A':>10}")
            continue

        n_targ = int(np.sum(y == 1))
        auc_mdm = evaluate_auc(X, y, 'mdm')
        auc_lda = evaluate_auc(X, y, 'lda')

        results.append((offset_ms, len(X), n_targ, auc_mdm, auc_lda))

        # Mark the current best
        best_mdm = max(r[3] for r in results)
        best_lda = max(r[4] for r in results)
        markers = []
        if auc_mdm == best_mdm and auc_mdm > 0.5:
            markers.append("MDM*")
        if auc_lda == best_lda and auc_lda > 0.5:
            markers.append("LDA*")
        marker_str = f" <-- {' '.join(markers)}" if markers else ""

        print(f"{offset_ms:>12} | {f'{offset_ms}-{offset_ms+EPOCH_DURATION_MS}ms':>18} | {len(X):>8} | {n_targ:>8} | {auc_mdm:>10.4f} | {auc_lda:>10.4f}{marker_str}")

    # Find best offsets
    best_mdm = max(results, key=lambda r: r[3])
    best_lda = max(results, key=lambda r: r[4])

    print(f"\n{'=' * 60}")
    print(f"OPTIMAL EPOCH TIMING:")
    print(f"  MDM:  Start at +{best_mdm[0]}ms -> window [{best_mdm[0]}, {best_mdm[0]+EPOCH_DURATION_MS}]ms  (AUC = {best_mdm[3]:.4f})")
    print(f"  LDA:  Start at +{best_lda[0]}ms -> window [{best_lda[0]}, {best_lda[0]+EPOCH_DURATION_MS}]ms  (AUC = {best_lda[3]:.4f})")
    print(f"{'=' * 60}")

    best_overall = max(results, key=lambda r: max(r[3], r[4]))
    best_auc = max(best_overall[3], best_overall[4])

    if best_auc < 0.55:
        print("\nWARNING: Even the best timing produces near-chance AUC.")
        print("The problem is likely NOT timing -- check signal quality.")
    else:
        print(f"\nTo apply: set EPOCH_START_OFFSET_S = {best_overall[0]/1000.0:.3f} in signal_processing.py")
        print(f"(This shifts epoch extraction to start {best_overall[0]}ms after flash onset)")

    # Plot results
    fig, ax = plt.subplots(figsize=(14, 6))
    offsets = [r[0] for r in results]
    aucs_mdm = [r[3] for r in results]
    aucs_lda = [r[4] for r in results]

    ax.plot(offsets, aucs_mdm, 'b-o', linewidth=2, markersize=4, label=f'Riemannian MDM (best: {best_mdm[3]:.3f} @ {best_mdm[0]}ms)')
    ax.plot(offsets, aucs_lda, 'r-s', linewidth=2, markersize=4, label=f'xDAWN+LDA (best: {best_lda[3]:.3f} @ {best_lda[0]}ms)')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level')
    ax.axvline(x=best_mdm[0], color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=best_lda[0], color='red', linestyle=':', alpha=0.5)

    ax.set_xlabel('Epoch Start Offset (ms after flash)', fontsize=12)
    ax.set_ylabel('Cross-Validated ROC AUC', fontsize=12)
    ax.set_title('Epoch Timing Calibration: Finding the Optimal P300 Window', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, max(0.85, best_auc + 0.05))

    # Add secondary x-axis showing epoch end
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] + EPOCH_DURATION_MS, ax.get_xlim()[1] + EPOCH_DURATION_MS)
    ax2.set_xlabel(f'Epoch End (ms after flash, duration={EPOCH_DURATION_MS}ms)', fontsize=10, color='gray')
    ax2.tick_params(axis='x', colors='gray')

    plot_path = os.path.join(output_dir, "epoch_timing_calibration.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nCalibration plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
