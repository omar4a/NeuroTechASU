import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Pipeline requirements
# Ensure you pip install mne pyriemann scikit-learn scipy numpy
try:
    from pyriemann.spatialfilters import Xdawn
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.classification import MDM
    from mne.decoding import Vectorizer
except ImportError:
    print("Warning: mne or pyriemann not installed.")

# Pre-processing is handled by data_collection.py via the shared signal_processing module

def evaluate_pipeline(X, y, pipeline, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    print(f"{name} -> Mean ROC AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    for i, s in enumerate(scores):
        print(f"   Fold {i+1}: {s:.4f}")
    return scores

def main():
    parser = argparse.ArgumentParser(description="P300 Offline Classifier Evaluation")
    parser.add_argument('--pipeline', type=str, choices=['A', 'B', 'Both'], default='Both',
                        help="Choose classification pipeline to evaluate")
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    x_path = os.path.join(output_dir, "X_train.npy")
    y_path = os.path.join(output_dir, "y_train.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print("Data files not found. Please run data_collection.py with the Speller UI first to generate X_train.npy and y_train.npy")
        return

    # Load Data
    print("Loading datasets...")
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"X shape: {X.shape}") # Expected (Epochs, Channels, Samples)
    print(f"y shape: {y.shape}")

    # Check classes distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    # Preprocessing note
    print("Dataset is already Bandpass Filtered (0.5-20Hz), Notch Filtered (50Hz), and Baseline Corrected.")
    
    # Sweep nfilter values to find optimal spatial filter count
    # With limited training data, fewer filters reduce overfitting risk
    nfilter_values = [2, 3, 4]
    
    # Pipeline A: xDAWN + LDA
    if args.pipeline in ['A', 'Both']:
        print("\n--- Evaluating Pipeline A (xDAWN + LDA) ---")
        for nf in nfilter_values:
            try:
                pipe_a = make_pipeline(
                    Xdawn(nfilter=nf),
                    Vectorizer(),
                    LinearDiscriminantAnalysis()
                )
                evaluate_pipeline(X, y, pipe_a, f"Pipeline A (xDAWN+LDA, nfilter={nf})")
            except Exception as e:
                print(f"Failed nfilter={nf}: {e}")

    # Pipeline B: Riemannian MDM
    if args.pipeline in ['B', 'Both']:
        print("\n--- Evaluating Pipeline B (Riemannian MDM) ---")
        for nf in nfilter_values:
            try:
                pipe_b = make_pipeline(
                    XdawnCovariances(nfilter=nf, estimator="oas"),
                    MDM()
                )
                evaluate_pipeline(X, y, pipe_b, f"Pipeline B (MDM, nfilter={nf})")
            except Exception as e:
                print(f"Failed nfilter={nf}: {e}")

if __name__ == "__main__":
    main()
