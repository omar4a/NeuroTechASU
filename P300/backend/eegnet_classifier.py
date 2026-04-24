import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# --- EEGNet Implementation in PyTorch ---
# Based on Lawhern et al. 2018: "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces"
class EEGNet(nn.Module):
    def __init__(self, n_channels=8, n_samples=200, n_classes=1, dropout_rate=0.5, kern_length=125, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        # Block 1: Temporal Convolution + Depthwise Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 2: Separable Convolution
        self.separable = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.conv2 = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Classification Block
        # Calculate flat features: n_samples // 4 // 8
        flat_features = F2 * (n_samples // 32)
        self.fc = nn.Linear(flat_features, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (Batch, 1, Channels, Samples)
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
        x = self.fc(x)
        return self.sigmoid(x).squeeze()

def train_and_evaluate():
    # Load Data
    data_dir = os.path.join(os.path.dirname(__file__), "training_data")
    X = np.load(os.path.join(data_dir, "X_train.npy"))
    y = np.load(os.path.join(data_dir, "y_train.npy"))
    
    # Preprocessing for PyTorch: (Epochs, 1, Channels, Samples)
    X = X[:, np.newaxis, :, :].astype(np.float32)
    y = y.astype(np.float32)
    
    print(f"Dataset Loaded: X={X.shape}, y={y.shape}")
    print(f"Target ratio: {np.mean(y):.2%}")
    
    # 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_aucs = []
    fold_accs = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # DataLoaders
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        model = EEGNet(n_channels=8, n_samples=200).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Training Loop
        model.train()
        for epoch in range(50):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_X_torch = torch.from_numpy(X_test).to(device)
            y_pred = model(test_X_torch).cpu().numpy()
            
            auc = roc_auc_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred > 0.5)
            
            fold_aucs.append(auc)
            fold_accs.append(acc)
            print(f"Fold {fold+1}: AUC = {auc:.4f}, ACC = {acc:.4f}")
            
    print("\n--- Final Results (EEGNet) ---")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
    print(f"Mean Accuracy: {np.mean(fold_accs):.4f}")

if __name__ == "__main__":
    train_and_evaluate()
