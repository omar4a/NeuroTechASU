import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# BCI specific
try:
    from pyriemann.spatialfilters import Xdawn
    from pyriemann.estimation import XdawnCovariances, ERPCovariances
    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace
    from mne.decoding import Vectorizer
except ImportError:
    print("Error: Missing pyriemann or mne. Install with: pip install pyriemann mne")

# ============================================================================
# Deep Learning: EEGNet Implementation
# ============================================================================
class EEGNet(nn.Module):
    def __init__(self, n_channels=8, n_samples=200, n_classes=1, dropout_rate=0.5, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        # Block 2
        self.separable = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.conv2 = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        # Output
        flat_features = F2 * (n_samples // 32)
        self.fc = nn.Linear(flat_features, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.separable(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        return self.sigmoid(self.fc(x)).squeeze()

class EEGNetWrapper:
    def __init__(self, name, params, device):
        self.name = name
        self.params = params
        self.device = device
        self.model = None
    
    def fit(self, X, y):
        X_pt = torch.from_numpy(X[:, np.newaxis, :, :]).float().to(self.device)
        y_pt = torch.from_numpy(y).float().to(self.device)
        self.model = EEGNet(n_channels=X.shape[1], n_samples=X.shape[2], **self.params).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.005, weight_decay=0.01)
        criterion = nn.BCELoss()
        self.model.train()
        ds = TensorDataset(X_pt, y_pt)
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        for epoch in range(50):
            for bx, by in dl:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.model(bx)
                if out.ndim == 0: out = out.unsqueeze(0)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_pt = torch.from_numpy(X[:, np.newaxis, :, :]).float().to(self.device)
            out = self.model(X_pt).cpu().numpy()
            if out.ndim == 0: out = np.array([out])
        return np.vstack([1-out, out]).T

# ============================================================================
# Main Sweep Logic
# ============================================================================
def main():
    print("="*60)
    print("P300 CLASSIFIER MODEL SWEEP - SOTA PIPELINES")
    print("="*60)

    data_dir = r'c:\Omar\Education\NeuroTech_ASU\P300\backend\training_data'
    x_path = os.path.join(data_dir, 'X_train.npy')
    y_path = os.path.join(data_dir, 'y_train.npy')
    
    if not os.path.exists(x_path):
        print(f"Error: Could not find {x_path}")
        return

    X = np.load(x_path)
    y = np.load(y_path)
    print(f"Data Loaded: {X.shape} epochs (Flashes)", flush=True)
    print(f"Class Distribution: Targets={np.sum(y)}, Non-Targets={len(y)-np.sum(y)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Define 15 Configurations
    pipelines = [
        ("xDAWN(2)+LDA", make_pipeline(Xdawn(nfilter=2, estimator='oas'), Vectorizer(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))),
        ("xDAWN(3)+LDA", make_pipeline(Xdawn(nfilter=3, estimator='oas'), Vectorizer(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))),
        ("xDAWN(4)+LDA", make_pipeline(Xdawn(nfilter=4, estimator='oas'), Vectorizer(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))),
        ("xDAWN(3)+LogReg", make_pipeline(Xdawn(nfilter=3, estimator='oas'), Vectorizer(), LogisticRegression(class_weight='balanced', max_iter=10000, solver='lbfgs'))),
        ("MDM(riemann, n=3)", make_pipeline(XdawnCovariances(nfilter=3, estimator='oas'), MDM(metric='riemann'))),
        ("MDM(logeuclid, n=3)", make_pipeline(XdawnCovariances(nfilter=3, estimator='oas'), MDM(metric=dict(mean='logeuclid', distance='logeuclid')))),
        ("TS(3)+LDA", make_pipeline(XdawnCovariances(nfilter=3, estimator='oas'), TangentSpace(metric='riemann'), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))),
        ("TS(3)+LogReg", make_pipeline(XdawnCovariances(nfilter=3, estimator='oas'), TangentSpace(metric='riemann'), LogisticRegression(class_weight='balanced', max_iter=10000))),
        ("TS(4)+LogReg", make_pipeline(XdawnCovariances(nfilter=4, estimator='oas'), TangentSpace(metric='riemann'), LogisticRegression(class_weight='balanced', max_iter=10000))),
        ("Xdawn(3)+Ridge", make_pipeline(Xdawn(nfilter=3, estimator='oas'), Vectorizer(), RidgeClassifier(class_weight='balanced'))),
        ("ERPCov+MDM", make_pipeline(ERPCovariances(estimator='oas'), MDM(metric='riemann'))),
        ("TS(3)+LogReg(C=0.1)", make_pipeline(XdawnCovariances(nfilter=3, estimator='oas'), TangentSpace(metric='riemann'), LogisticRegression(C=0.1, class_weight='balanced', max_iter=10000))),
        ("EEGNet-Standard", EEGNetWrapper("EEGNet-Std", {"F1":8, "D":2, "F2":16, "dropout_rate":0.5}, device)),
        ("EEGNet-Compact", EEGNetWrapper("EEGNet-Cmp", {"F1":4, "D":2, "F2":8, "dropout_rate":0.25}, device)),
        ("Ensemble(ML+DL)", "ENSM") 
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for i, (name, pipe) in enumerate(pipelines):
        print(f"\n[{i+1}/{len(pipelines)}] STARTING: {name}", flush=True)
        fold_aucs = []
        inf_times = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                if name == "Ensemble(ML+DL)":
                    m1 = pipelines[1][1] # xDAWN(3)+LDA
                    m2 = pipelines[12][1] # EEGNet-Standard
                    m1.fit(X_train, y_train)
                    m2.fit(X_train, y_train)
                    p1 = m1.predict_proba(X_test)[:, 1]
                    p2 = m2.predict_proba(X_test)[:, 1]
                    y_pred = (p1 + p2) / 2.0
                else:
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict_proba(X_test)[:, 1]

                auc = roc_auc_score(y_test, y_pred)
                fold_aucs.append(auc)
                
                # Real-time Inference Speed Test (1 epoch)
                single_epoch = X_test[0:1]
                t0 = time.perf_counter()
                if name == "Ensemble(ML+DL)":
                    _ = (m1.predict_proba(single_epoch) + m2.predict_proba(single_epoch)) / 2
                else:
                    _ = pipe.predict_proba(single_epoch)
                inf_times.append((time.perf_counter() - t0) * 1000)
                
                print(f"   Fold {fold+1}: AUC={auc:.4f} | Inf={inf_times[-1]:.2f}ms", flush=True)
            except Exception as e:
                print(f"   Fold {fold+1} FAILED: {e}", flush=True)

        if fold_aucs:
            avg_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            avg_inf = np.mean(inf_times)
            results.append((name, avg_auc, std_auc, avg_inf))
            print(f">> COMPLETED {name}: Mean AUC = {avg_auc:.4f} | Avg Inf = {avg_inf:.2f}ms", flush=True)

    # Final Leaderboard
    results.sort(key=lambda x: x[1], reverse=True)
    print("\n" + "="*85)
    print(f"{'RANK':<4} | {'PIPELINE':<25} | {'MEAN AUC':<10} | {'STD':<8} | {'AVG INF TIME':<15} | {'REALTIME'}")
    print("-" * 85)
    for i, (name, auc, std, inf) in enumerate(results):
        rt_status = "OK (<50ms)" if inf < 50 else "CAUTION"
        print(f"{i+1:<4} | {name:<25} | {auc:.4f}     | {std:.3f}    | {inf:>8.2f} ms     | {rt_status}")
    print("="*85)

if __name__ == "__main__":
    main()
