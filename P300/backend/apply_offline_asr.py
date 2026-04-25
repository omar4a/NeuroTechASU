import os
import sys
import numpy as np
import asrpy
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# pyriemann imports
from pyriemann.spatialfilters import Xdawn
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

# mne import
from mne.decoding import Vectorizer

try:
    from signal_processing import FS, ACTIVE_CHANNEL_INDICES, apply_preprocessing, EPOCH_LEN
except ImportError:
    FS = 250
    ACTIVE_CHANNEL_INDICES = list(range(8))
    EPOCH_LEN = 0.8
    apply_preprocessing = lambda x: x

def get_epoch_samples():
    return int(EPOCH_LEN * FS)

def evaluate_models_on_data(X, y):
    pipelines = {}
    for n in range(1, 6):
        pipelines[f"xDAWN({n}) + LDA"] = make_pipeline(
            Xdawn(nfilter=n),
            Vectorizer(),
            LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        )
    
    pipelines["MDM"] = make_pipeline(
        XdawnCovariances(nfilter=3, estimator='oas'),
        MDM()
    )
    
    pipelines["TS-LogR (New)"] = make_pipeline(
        XdawnCovariances(nfilter=3, estimator='oas'),
        TangentSpace(),
        LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced')
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    import warnings
    warnings.filterwarnings("ignore")

    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        results.append((name, np.mean(scores), np.std(scores)))

    results.sort(key=lambda x: x[1], reverse=True)
    for i, (name, auc, std) in enumerate(results, 1):
        print(f"{i:<5} | {name:<25} | {auc:.4f}     | {std:.3f}")

def main():
    print("\n" + "="*80)
    print(" APPLYING OFFLINE ASR TO Omar_final2")
    print("="*80)

    # 1. Load Raw Data
    raw_dir = os.path.join(os.path.dirname(__file__), "raw_session")
    eeg_path = os.path.join(raw_dir, "eeg_continuous_omar_final2.npy")
    ts_path = os.path.join(raw_dir, "eeg_timestamps_omar_final2.npy")
    ev_path = os.path.join(raw_dir, "flash_events_omar_final2.npy")

    if not all(os.path.exists(p) for p in [eeg_path, ts_path, ev_path]):
        print("Error: Raw session files not found.")
        return

    eeg_raw = np.load(eeg_path)
    timestamps = np.load(ts_path)
    flash_events = np.load(ev_path, allow_pickle=True)

    # Keep only active channels
    eeg_raw = eeg_raw[:, ACTIVE_CHANNEL_INDICES]
    
    print(f"Loaded Raw Data Shape: {eeg_raw.shape} (Samples x Channels)")
    print(f"Total Flash Events: {len(flash_events)}")

    # Pre-filter before ASR (bandpass and notch are crucial for ASR to work well)
    eeg_filtered = apply_preprocessing(eeg_raw)

    # 2. Apply ASR
    # asrpy expects (n_channels, n_samples)
    eeg_transposed = eeg_filtered.T
    
    import mne
    
    print("\nApplying ASR (Cutoff=20)...")
    asr = asrpy.ASR(sfreq=FS, cutoff=20)
    
    # Wrap in MNE RawArray as required by asrpy
    info = mne.create_info(ch_names=[f'CH{i}' for i in range(8)], sfreq=FS, ch_types='eeg')
    raw = mne.io.RawArray(eeg_transposed, info)
    
    # Fit ASR to the data (finds clean reference portions automatically)
    asr.fit(raw)
    
    # [TASK 1] Serialize the fitted ASR object for real-time state preservation
    train_dir = os.path.join(os.path.dirname(__file__), "training_data")
    os.makedirs(train_dir, exist_ok=True)
    asr_state_path = os.path.join(train_dir, "asr_state.pkl")
    with open(asr_state_path, 'wb') as f:
        pickle.dump(asr, f)
    print(f"Successfully saved ASR state to: {asr_state_path}")
    
    # Apply ASR to scrub artifacts
    raw_clean = asr.transform(raw)
    eeg_asr_transposed = raw_clean.get_data()
    
    # Transpose back to (n_samples, n_channels)
    eeg_cleaned = eeg_asr_transposed.T
    print(f"Cleaned Data Shape: {eeg_cleaned.shape}")

    # 3. Extract Epochs
    epoch_samples = get_epoch_samples()
    baseline_samples = int(0.1 * FS) # 100ms baseline before flash
    offset_s = 0.06 # Strict 60ms offset
    offset_samples = int(offset_s * FS)
    
    X_list = []
    y_list = []

    print(f"\nExtracting Epochs with strict {offset_s*1000}ms offset...")
    
    for event in flash_events:
        flash_time = event['time']
        is_target = event['label']
        
        # Find closest index
        idx = np.searchsorted(timestamps, flash_time)
        
        start_idx = idx + offset_samples
        end_idx = start_idx + epoch_samples
        
        # Baseline window
        base_start = idx - baseline_samples
        base_end = idx
        
        if base_start >= 0 and end_idx < len(eeg_cleaned):
            # Extract
            epoch = eeg_cleaned[start_idx:end_idx, :].copy()
            # Baseline correct
            baseline = np.mean(eeg_cleaned[base_start:base_end, :], axis=0)
            epoch -= baseline
            
            # Transpose epoch to (channels, samples) for ML pipelines
            X_list.append(epoch.T)
            y_list.append(is_target)

    X_train_asr = np.array(X_list)
    y_train_asr = np.array(y_list)
    
    print(f"Successfully extracted {len(X_train_asr)} epochs.")

    # 4. Save Outputs
    train_dir = os.path.join(os.path.dirname(__file__), "training_data")
    out_x = os.path.join(train_dir, "X_train_asr_Omar_final2.npy")
    out_y = os.path.join(train_dir, "y_train_asr_Omar_final2.npy")
    
    np.save(out_x, X_train_asr)
    np.save(out_y, y_train_asr)
    print(f"\nSaved ASR-cleaned epochs to:\n- {out_x}\n- {out_y}")

    # 5. Evaluate Before and After
    print("\n" + "="*60)
    print(" PERFORMANCE COMPARISON: BEFORE vs AFTER ASR")
    print("="*60)
    
    # Load the "Before" data (the one currently without ASR)
    X_before = np.load(os.path.join(train_dir, "X_train_Omar_final2.npy"))
    y_before = np.load(os.path.join(train_dir, "y_train_Omar_final2.npy"))

    print("\n--- BEFORE ASR ---")
    print(f"RANK  | PIPELINE                  | MEAN AUC   | STD")
    print("-" * 60)
    evaluate_models_on_data(X_before, y_before)

    print("\n--- AFTER ASR ---")
    print(f"RANK  | PIPELINE                  | MEAN AUC   | STD")
    print("-" * 60)
    evaluate_models_on_data(X_train_asr, y_train_asr)

if __name__ == "__main__":
    main()
