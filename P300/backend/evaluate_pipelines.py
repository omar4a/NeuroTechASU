import os
import numpy as np
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

def evaluate_models():
    print("============================================================")
    print(" P300 CLASSIFIER MODEL SWEEP (omar_final_2)")
    print("============================================================")
    
    # 1. Load Data
    train_dir = os.path.join(os.path.dirname(__file__), "training_data")
    x_path = os.path.join(train_dir, "X_train_Omar_final2.npy")
    y_path = os.path.join(train_dir, "y_train_Omar_final2.npy")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Error: Data not found in {train_dir}")
        print("Please ensure X_train_Omar_final2.npy and y_train_Omar_final2.npy exist.")
        return

    print("Loading training data...")
    X = np.load(x_path)
    y = np.load(y_path)
    
    print(f"Data Loaded: {X.shape} epochs (Flashes)")
    print(f"Class Distribution: Targets={np.sum(y==1)}, Non-Targets={np.sum(y==0)}")
    print("Keeping all 8 active channels for classification.\n")

    # 2. Define Pipelines
    pipelines = {}
    for n in range(1, 6):
        pipelines[f"xDAWN({n}) + LDA"] = make_pipeline(
            Xdawn(nfilter=n),
            Vectorizer(),
            LinearDiscriminantAnalysis(solver='svd') # SVD is more stable than LSQR for this sweep
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

    # 3. Evaluate using Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, pipeline in pipelines.items():
        print(f"Evaluating {name}...")
        # We use roc_auc to account for the class imbalance
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        mean_auc = np.mean(scores)
        std_auc = np.std(scores)
        results.append((name, mean_auc, std_auc))

    # 4. Print Clean Terminal Report
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by Mean AUC
    
    print("\n" + "="*60)
    print(f"{'RANK':<5} | {'PIPELINE':<25} | {'MEAN AUC':<10} | {'STD':<8}")
    print("-" * 60)
    
    for i, (name, auc, std) in enumerate(results, 1):
        print(f"{i:<5} | {name:<25} | {auc:.4f}     | {std:.3f}")
    print("="*60)


if __name__ == "__main__":
    evaluate_models()
