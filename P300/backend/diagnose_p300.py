"""
P300 Diagnostic Script — Comprehensive analysis of why classification is failing.

Analyzes all dataset versions to identify:
1. Whether a P300 component is present at the expected latency
2. The impact of OSCAR delay on epoch alignment
3. Whether training data size is sufficient
4. Per-channel ERP quality
5. Cross-validated classification performance
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
    from pyriemann.spatialfilters import Xdawn
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.classification import MDM
    from mne.decoding import Vectorizer
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False
    print("WARNING: pyriemann/mne not available. Skipping xDAWN pipelines.")

output_dir = os.path.dirname(os.path.abspath(__file__))

CHANNEL_NAMES = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
FS = 250
EPOCH_LEN_S = 0.8
SAMPLES = int(FS * EPOCH_LEN_S)  # 200

def load_dataset(version_suffix=""):
    """Load a dataset version. Returns X (epochs, ch, samples), y (labels)."""
    xf = os.path.join(output_dir, f"X_train{version_suffix}.npy")
    yf = os.path.join(output_dir, f"y_train{version_suffix}.npy")
    if not os.path.exists(xf):
        return None, None
    return np.load(xf), np.load(yf)


def analyze_erp_all_channels(X, y, version_name, save_dir):
    """Plot ERP for all 8 channels, not just Pz."""
    targets = X[y == 1]
    nontargets = X[y == 0]
    
    times = np.linspace(0, EPOCH_LEN_S * 1000, X.shape[2])
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    fig.suptitle(f'{version_name}: Grand Average ERP — All Channels\n'
                 f'Targets={len(targets)}, Non-Targets={len(nontargets)}',
                 fontsize=14, fontweight='bold')
    
    for i, (ax, ch_name) in enumerate(zip(axes.flat, CHANNEL_NAMES)):
        targ_avg = np.mean(targets[:, i, :], axis=0)
        nontarg_avg = np.mean(nontargets[:, i, :], axis=0)
        diff = targ_avg - nontarg_avg
        
        ax.plot(times, targ_avg, 'r-', linewidth=1.5, label='Target')
        ax.plot(times, nontarg_avg, 'b-', linewidth=1.5, label='Non-Target')
        ax.plot(times, diff, 'g--', linewidth=1, alpha=0.7, label='Difference')
        ax.axvline(x=300, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=550, color='orange', linestyle=':', alpha=0.5, label='300ms+OSCAR')
        ax.set_title(f'{ch_name} (ch{i})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Amplitude (µV)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'erp_all_ch_{version_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved ERP plot for {version_name}")


def compute_snr_metrics(X, y, version_name):
    """Compute signal-to-noise ratio of the P300 component."""
    targets = X[y == 1]
    nontargets = X[y == 0]
    
    print(f"\n  --- SNR Analysis for {version_name} ---")
    print(f"  Targets: {len(targets)}, Non-Targets: {len(nontargets)}")
    
    # Check multiple time windows for the P300 peak
    windows = {
        '200-400ms (standard P300)': (int(0.200*FS), int(0.400*FS)),
        '300-500ms (late P300)': (int(0.300*FS), int(0.500*FS)),
        '400-600ms (OSCAR-shifted?)': (int(0.400*FS), int(0.600*FS)),
        '500-700ms (heavily shifted?)': (int(0.500*FS), int(0.700*FS)),
    }
    
    for window_name, (s_start, s_end) in windows.items():
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            targ_window = targets[:, ch_idx, s_start:s_end]
            nontarg_window = nontargets[:, ch_idx, s_start:s_end]
            
            targ_mean = np.mean(targ_window)
            nontarg_mean = np.mean(nontarg_window)
            diff = targ_mean - nontarg_mean
            noise_std = np.std(nontarg_window)
            
            snr = abs(diff) / noise_std if noise_std > 0 else 0
            
            if ch_name == 'Pz' or snr > 0.3:
                print(f"    {window_name} | {ch_name}: "
                      f"Target={targ_mean:.3f}µV, NonTarget={nontarg_mean:.3f}µV, "
                      f"Diff={diff:.3f}µV, SNR={snr:.3f}")


def simulate_oscar_shift(X, y, version_name):
    """
    Simulate what happens if we shift the epoch window to compensate for OSCAR delay.
    
    Since we can't re-epoch from continuous data, we CROP the existing epochs
    to simulate different alignment offsets. If the real P300 is shifted by
    250ms due to OSCAR, then the discriminative signal should be concentrated
    in the LATER part of the epoch.
    """
    targets = X[y == 1]
    nontargets = X[y == 0]
    
    print(f"\n  --- OSCAR Shift Simulation for {version_name} ---")
    print(f"  Testing where discriminative signal is strongest...")
    
    # Split epoch into early and late halves
    half = SAMPLES // 2  # 100 samples = 400ms
    
    # Early half (0-400ms) — should contain P300 if NO OSCAR delay
    early_targ = np.mean(targets[:, :, :half], axis=(0, 2))
    early_nontarg = np.mean(nontargets[:, :, :half], axis=(0, 2))
    early_diff = np.abs(early_targ - early_nontarg)
    
    # Late half (400-800ms) — should contain P300 if OSCAR shifted it
    late_targ = np.mean(targets[:, :, half:], axis=(0, 2))
    late_nontarg = np.mean(nontargets[:, :, half:], axis=(0, 2))
    late_diff = np.abs(late_targ - late_nontarg)
    
    print(f"  Early half (0-400ms) mean |diff| per channel: {early_diff.round(3)}")
    print(f"  Late  half (400-800ms) mean |diff| per channel: {late_diff.round(3)}")
    print(f"  Early total: {early_diff.sum():.3f}, Late total: {late_diff.sum():.3f}")
    
    if late_diff.sum() > early_diff.sum():
        print(f"  >>> LATE half has MORE discriminative signal — consistent with OSCAR delay!")
    else:
        print(f"  >>> EARLY half has more signal — OSCAR delay may not be the issue here.")
    
    # Now test classification on early-only vs late-only vs full
    if HAS_PYRIEMANN and len(targets) >= 10:
        results = {}
        
        subsets = {
            'Full epoch (0-800ms)': X,
            'Early only (0-400ms)': X[:, :, :half],
            'Late only (400-800ms)': X[:, :, half:],
            'Middle (200-600ms)': X[:, :, 50:150],
        }
        
        for label, X_sub in subsets.items():
            try:
                pipe = make_pipeline(Xdawn(nfilter=2), Vectorizer(), LinearDiscriminantAnalysis())
                cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=42)
                scores = cross_val_score(pipe, X_sub, y, cv=cv, scoring='roc_auc')
                results[label] = scores.mean()
                print(f"    {label}: AUC = {scores.mean():.4f} (±{scores.std():.4f})")
            except Exception as e:
                print(f"    {label}: FAILED — {e}")
        
        if results:
            best = max(results, key=results.get)
            print(f"  >>> Best subset: {best} (AUC={results[best]:.4f})")


def run_classification(X, y, version_name):
    """Run cross-validated classification with both pipelines."""
    if not HAS_PYRIEMANN:
        print("  Skipping classification (pyriemann not available)")
        return
    
    n_targets = np.sum(y == 1)
    n_nontargets = np.sum(y == 0)
    
    print(f"\n  --- Classification for {version_name} ---")
    print(f"  Targets: {n_targets}, Non-Targets: {n_nontargets}, Total: {len(y)}")
    
    if n_targets < 10:
        print(f"  SKIPPED: Too few targets ({n_targets}) for cross-validation")
        return
    
    n_splits = min(5, n_targets)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Pipeline A: xDAWN + LDA
    try:
        pipe_a = make_pipeline(Xdawn(nfilter=3), Vectorizer(), LinearDiscriminantAnalysis())
        scores_a = cross_val_score(pipe_a, X, y, cv=cv, scoring='roc_auc')
        print(f"  xDAWN+LDA:  AUC = {scores_a.mean():.4f} (±{scores_a.std():.4f})")
    except Exception as e:
        print(f"  xDAWN+LDA: FAILED — {e}")
    
    # Pipeline B: Riemannian MDM
    try:
        pipe_b = make_pipeline(XdawnCovariances(nfilter=3, estimator="oas"), MDM())
        scores_b = cross_val_score(pipe_b, X, y, cv=cv, scoring='roc_auc')
        print(f"  Riemann MDM: AUC = {scores_b.mean():.4f} (±{scores_b.std():.4f})")
    except Exception as e:
        print(f"  Riemann MDM: FAILED — {e}")


def check_amplitude_distribution(X, y, version_name):
    """Check if data looks like real EEG or is corrupted."""
    print(f"\n  --- Amplitude Distribution for {version_name} ---")
    print(f"  Shape: {X.shape}")
    print(f"  Global range: [{X.min():.2f}, {X.max():.2f}] µV")
    print(f"  Global std: {X.std():.2f} µV")
    
    for i, ch in enumerate(CHANNEL_NAMES):
        ch_data = X[:, i, :]
        print(f"    {ch}: mean={ch_data.mean():.3f}, std={ch_data.std():.3f}, "
              f"range=[{ch_data.min():.2f}, {ch_data.max():.2f}]")


def main():
    print("=" * 70)
    print("P300 BCI DIAGNOSTIC REPORT")
    print("=" * 70)
    
    versions = [
        ("", "Current"),
        ("_v0", "V0"),
        ("_v1", "V1"),
        ("_v2", "V2"),
    ]
    
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    
    for suffix, name in versions:
        X, y = load_dataset(suffix)
        if X is None:
            print(f"\n{'='*40}\nDataset {name}: NOT FOUND\n{'='*40}")
            continue
        
        print(f"\n{'='*70}")
        print(f"DATASET: {name} (X_train{suffix}.npy)")
        print(f"{'='*70}")
        
        check_amplitude_distribution(X, y, name)
        analyze_erp_all_channels(X, y, name, diag_dir)
        compute_snr_metrics(X, y, name)
        simulate_oscar_shift(X, y, name)
        run_classification(X, y, name)
    
    # Summary recommendations
    print(f"\n{'='*70}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    print("""
KEY FINDINGS TO CHECK:
1. Do ERP plots show Target/Non-Target separation at 300ms (expected) or 550ms (OSCAR-shifted)?
2. Is the late-half discrimination stronger than early-half? → OSCAR delay confirmation
3. Are amplitudes in realistic EEG range (std ~5-30µV per channel)?
4. Are there enough target epochs for reliable classification (minimum ~100)?

Diagnostic plots saved to: {diag_dir}
""".format(diag_dir=diag_dir))


if __name__ == "__main__":
    main()
