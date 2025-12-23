import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# ============================================================
# 1. CONFIG (BANGALORE-SPECIFIC)
# ============================================================

CSV_PATH = "dragonfruit_bangalore_dataset.csv"
WINDOW_SIZE = 45          # longer window for BLR seasonal stability
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
MODEL_PATH = "lstm_bangalore_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

X = df.drop(columns=["Yield"]).values
y = df["Yield"].values

# ============================================================
# 3. FEATURE SCALING
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4. SLIDING WINDOW SEQUENCES
# ============================================================

def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y, WINDOW_SIZE)

# ============================================================
# 5. TRAIN / VAL / TEST SPLIT (NO SHUFFLE)
# ============================================================

train_size = int(0.7 * len(X_seq))
val_size = int(0.15 * len(X_seq))

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]

X_val = X_seq[train_size:train_size + val_size]
y_val = y_seq[train_size:train_size + val_size]

X_test = X_seq[train_size + val_size:]
y_test = y_seq[train_size + val_size:]

# ============================================================
# 6. DATASET CLASS
# ============================================================

class BangaloreDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(BangaloreDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(BangaloreDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(BangaloreDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 7. LSTM MODEL (BLR-TUNED)
# ============================================================

class BangaloreLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=96,     # stronger temporal capacity
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(96, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

model = BangaloreLSTM(X.shape[1]).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================================================
# 8. TRAINING LOOP
# ============================================================

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * Xb.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            preds = model(Xb)
            val_loss += criterion(preds, yb).item() * Xb.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ============================================================
# 9. TEST EVALUATION
# ============================================================

model.eval()
preds, targets = [], []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(DEVICE)
        pred = model(Xb).cpu().numpy()
        preds.append(pred)
        targets.append(yb.numpy())

preds = np.concatenate(preds)
targets = np.concatenate(targets)

rmse = sqrt(mean_squared_error(targets, preds))
mae = mean_absolute_error(targets, preds)

print("\n===== BANGALORE LSTM TEST RESULTS =====")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")

# ============================================================
# 10. SAVE MODEL
# ============================================================

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved as {MODEL_PATH}")
