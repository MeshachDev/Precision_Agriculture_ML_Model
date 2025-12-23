import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

# ============================================================
# 1. CONFIG
# ============================================================

CSV_PATH = "./dragonfruit_generic_dataset.csv"  # or dragonfruit_generic.csv
WINDOW_SIZE = 30
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
MODEL_PATH = "lstm_yield_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

X = df.drop(columns=["Yield"]).values
y = df["Yield"].values

# ============================================================
# 3. SCALING
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4. CREATE SLIDING WINDOWS
# ============================================================

def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y, WINDOW_SIZE)

# ============================================================
# 5. TRAIN / VAL / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X_seq, y_seq, test_size=0.3, shuffle=False
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, shuffle=False
)

# ============================================================
# 6. DATASET CLASS
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TimeSeriesDataset(X_train, y_train)
val_ds   = TimeSeriesDataset(X_val, y_val)
test_ds  = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 7. LSTM MODEL
# ============================================================

class LSTMYieldModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

model = LSTMYieldModel(input_size=X.shape[1]).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================================================
# 8. TRAINING LOOP
# ============================================================

def train_model(model, train_loader, val_loader):
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

train_model(model, train_loader, val_loader)

# ============================================================
# 9. EVALUATION
# ============================================================

model.eval()
preds, targets = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        pred = model(X_batch).cpu().numpy()
        preds.append(pred)
        targets.append(y_batch.numpy())

preds = np.concatenate(preds)
targets = np.concatenate(targets)

rmse = sqrt(mean_squared_error(targets, preds))
mae = mean_absolute_error(targets, preds)

print("\n===== TEST RESULTS =====")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")

# ============================================================
# 10. SAVE MODEL
# ============================================================

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved as {MODEL_PATH}")
