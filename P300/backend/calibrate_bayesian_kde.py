import os
import numpy as np
import pickle
from scipy.stats import gaussian_kde
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from pyriemann.spatialfilters import Xdawn
from mne.decoding import Vectorizer

from signal_processing import TRAINING_DATA_DIR, ACTIVE_CHANNEL_INDICES

def main():
    print("=====================================================")
    print(" CALIBRATING BAYESIAN KDE FOR DYNAMIC STOPPING")
    print("=====================================================")

    # 1. Load ASR-Cleaned Data
    x_path = os.path.join(TRAINING_DATA_DIR, "X_train_asr_Omar_final2.npy")
    y_path = os.path.join(TRAINING_DATA_DIR, "y_train_asr_Omar_final2.npy")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Error: Could not find training data at {x_path}")
        return

    X = np.load(x_path)[:, ACTIVE_CHANNEL_INDICES, :]
    y = np.load(y_path)
    
    print(f"Loaded Data Shape: {X.shape}, Labels Shape: {y.shape}")

    # 2. Define Optimized Pipeline
    clf = make_pipeline(
        Xdawn(nfilter=4),
        Vectorizer(),
        LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    )

    # 3. Generate Out-of-Fold Decision Scores
    # We use decision_function to get unbounded scores [-inf, inf]
    # which is strictly required for stable Gaussian KDE fitting.
    print("Running Stratified 5-Fold Cross-Validation to extract out-of-fold scores...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # method='decision_function' returns the continuous LDA score
    scores = cross_val_predict(clf, X, y, cv=cv, method='decision_function', n_jobs=-1)

    # 4. Separate Scores
    scores_target = scores[y == 1]
    scores_nontarget = scores[y == 0]
    
    print(f"Target Scores: N={len(scores_target)}, Mean={scores_target.mean():.3f}, Std={scores_target.std():.3f}")
    print(f"Non-Target Scores: N={len(scores_nontarget)}, Mean={scores_nontarget.mean():.3f}, Std={scores_nontarget.std():.3f}")

    # 5. Fit KDEs
    print("Fitting Gaussian KDEs...")
    kde_target = gaussian_kde(scores_target)
    kde_nontarget = gaussian_kde(scores_nontarget)

    # 6. Save KDEs
    kde_path = os.path.join(TRAINING_DATA_DIR, "kde_distributions.pkl")
    with open(kde_path, 'wb') as f:
        pickle.dump({
            'target': kde_target,
            'nontarget': kde_nontarget
        }, f)
        
    print(f"Successfully saved KDE distributions to: {kde_path}")
    print("=====================================================")

if __name__ == "__main__":
    main()
