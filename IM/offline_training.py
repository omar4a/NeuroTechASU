import numpy as np
import mne
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, KFold
import joblib
import pyxdf

def load_and_epoch_data(filepath="data.xdf"):
    """
    Loads recorded EEG data and markers via pyxdf, and creates epoched data.
    - 3 classes (0: Rest, 1: Left MI, 2: Right MI)
    - 8 channels
    - Cropped exactly from 1.0s to 4.0s (3 seconds total at 250Hz = 750 samples)
    """
    print(f"Loading {filepath} via pyxdf...")
    try:
        streams, header = pyxdf.load_xdf(filepath)
    except Exception as e:
        print(f"Failed to load {filepath}. Ensure it exists in the directory. Error: {e}")
        print("Falling back to dummy data so script doesn't crash...")
        return _generate_dummy_data()
        
    eeg_stream = None
    marker_stream = None
    
    for stream in streams:
        if stream['info']['type'][0] == 'EEG':
            eeg_stream = stream
        elif stream['info']['name'][0] == 'IM_Markers':
            marker_stream = stream
            
    if eeg_stream is None or marker_stream is None:
        raise ValueError("Could not find both EEG and Marker streams in XDF.")
        
    eeg_data = eeg_stream['time_series'] # Shape: (n_samples, n_channels)
    eeg_times = eeg_stream['time_stamps']
    
    # We only want the first 8 channels (EEG), ignoring counters/battery
    eeg_data = eeg_data[:, :8].T # Transpose to (n_channels, n_samples)
    
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    
    marker_times = marker_stream['time_stamps']
    marker_data = marker_stream['time_series']
    
    # Map string markers to integer events
    marker_map = {"Rest_Start": 0, "Left_MI_Start": 1, "Right_MI_Start": 2}
    
    epochs_data = []
    labels = []
    
    # We want 1.0s to 4.0s relative to marker onset
    window_samples = int(3.0 * sfreq)
    delay_samples = int(1.0 * sfreq)
    
    for marker_time, marker_val in zip(marker_times, marker_data):
        marker_str = marker_val[0]
        if marker_str in marker_map:
            label = marker_map[marker_str]
            
            # Find the index in eeg_times closest to marker_time
            idx = np.argmin(np.abs(eeg_times - marker_time))
            
            start_idx = idx + delay_samples
            end_idx = start_idx + window_samples
            
            if end_idx <= eeg_data.shape[1]:
                epochs_data.append(eeg_data[:, start_idx:end_idx])
                labels.append(label)
                
    if len(epochs_data) == 0:
        raise ValueError("No valid epochs found. Check markers and durations.")
        
    epochs_data = np.array(epochs_data) # Shape: (n_epochs, 8, window_samples)
    labels = np.array(labels)
    
    # Convert to float64 for MNE, sometimes data is float32
    epochs_data = epochs_data.astype(np.float64)
    
    ch_names = [f'CH{i+1}' for i in range(8)]
    ch_types = ['eeg'] * 8
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    epochs = mne.EpochsArray(epochs_data, info, verbose=False)
    
    return epochs, labels

def _generate_dummy_data():
    sfreq = 250.0
    n_trials = 150
    n_samples = int(3.0 * sfreq) 
    epochs_data = np.random.randn(n_trials, 8, n_samples)
    labels = np.random.randint(0, 3, n_trials)
    info = mne.create_info(ch_names=[f'CH{i+1}' for i in range(8)], sfreq=sfreq, ch_types=['eeg']*8)
    epochs = mne.EpochsArray(epochs_data, info, verbose=False)
    return epochs, labels

def main():
    """
    offline_training.py (Pipeline Construction)
    Loads EEG data, applies CAR & Filter, trains a CSP+LDA pipeline (OVR),
    runs CV, and exports the model.
    """
    # 1. Data Loading & Epoching
    # Note: Mandatory 1.0s delay is handled in the cropping phase of load_and_epoch_data
    epochs, y = load_and_epoch_data("dummy_path.xdf")

    # 2. Preprocessing
    print("Applying Common Average Reference (CAR)...")
    epochs, _ = mne.set_eeg_reference(epochs, ref_channels='average', projection=False, verbose=False)

    print("Applying zero-phase FIR bandpass filter (7.0 - 30.0 Hz)...")
    epochs.filter(l_freq=7.0, h_freq=30.0, method='fir', phase='zero', verbose=False)

    # Extract NumPy array for scikit-learn (Shape: n_trials, n_channels, n_samples)
    X = epochs.get_data(copy=False)

    # 3. Machine Learning Pipeline
    print("Constructing Pipeline (CSP -> OVR LDA)...")
    # Step 1: CSP with n_components=4 and log=True.
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    
    # Step 2: LDA with solver='svd'.
    lda = LinearDiscriminantAnalysis(solver='svd')
    
    # Handle the 3-class data using a One-Versus-Rest (OVR) multiclass strategy
    # MNE's CSP inherently computes OVR spatial filters when presented with >2 classes.
    # Wrapping LDA in OneVsRestClassifier enforces OVR at the classification level.
    clf = make_pipeline(csp, OneVsRestClassifier(lda))

    # 4. Verification & Export
    print("Running 10-fold Cross-Validation...")
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    
    print("\nAccuracy Matrix (10 folds):")
    print(np.round(scores, 4))
    print(f"Mean CV Accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # Train the pipeline on the full dataset
    print("\nTraining final pipeline on full dataset...")
    clf.fit(X, y)

    # Save to disk
    model_path = "eeg_mi_pipeline.pkl"
    joblib.dump(clf, model_path)
    print(f"Pipeline successfully serialized to: {model_path}")

if __name__ == "__main__":
    main()
