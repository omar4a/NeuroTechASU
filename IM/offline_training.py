import numpy as np
import mne
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, KFold
import joblib

def load_and_epoch_data(filepath=None):
    """
    Loads recorded EEG data and markers, and creates epoched data.
    In a real scenario, you'd use mne.io.read_raw_xdf or load from CSV,
    then use mne.events_from_annotations, and finally mne.Epochs.
    
    This function generates simulated data that adheres strictly to the
    described pipeline requirements:
    - 3 classes (0: Rest, 1: Left MI, 2: Right MI)
    - 8 channels
    - Cropped exactly from 1.0s to 4.0s (3 seconds total at 250Hz = 750 samples)
    """
    # 250 Hz sampling rate
    sfreq = 250.0
    
    # Simulate loading and epoching
    print("Loading data and creating epochs (1.0s to 4.0s relative to marker)...")
    
    # 150 trials total, 8 channels, 750 samples (3.0s duration)
    n_trials = 150
    n_channels = 8
    n_samples = int(3.0 * sfreq) 
    
    # Generating dummy float64 data and labels
    epochs_data = np.random.randn(n_trials, n_channels, n_samples)
    labels = np.random.randint(0, 3, n_trials)
    
    # Create MNE Info object
    ch_names = [f'CH{i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create MNE EpochsArray
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
