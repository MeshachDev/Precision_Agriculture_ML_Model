import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

WINDOW_SIZE = 45
MODEL_PATH = "lstm_bangalore_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = [
    "Temperature", "Humidity", "Rainfall", "Sunlight", "SoilMoisture",
    "SoilPH", "WindSpeed", "PlantAge", "Branches", "StemDiameter",
    "FlowerCount", "PrevFruitCount", "PestIndex", "DiseaseIndex",
    "Fertilizer", "Irrigation", "PruningFreq", "OrganicMatter"
]

NUM_FEATURES = len(FEATURE_NAMES)

# ============================================================
# LSTM MODEL (same as training)
# ============================================================

class BangaloreLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(96, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

model = BangaloreLSTM(NUM_FEATURES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ============================================================
# SCALER (MUST MATCH TRAINING DISTRIBUTION)
# ============================================================

scaler = StandardScaler()
scaler.fit(np.random.uniform(0, 1, size=(2000, NUM_FEATURES)))

# ============================================================
# SINGLE SENSOR INPUT (EXAMPLE)
# ============================================================

single_sensor_reading = {
    "Temperature": 27.5,
    "Humidity": 70,
    "Rainfall": 1.2,
    "Sunlight": 7.8,
    "SoilMoisture": 42,
    "SoilPH": 6.6,
    "WindSpeed": 3.5,
    "PlantAge": 14,
    "Branches": 10,
    "StemDiameter": 4.1,
    "FlowerCount": 8,
    "PrevFruitCount": 15,
    "PestIndex": 0.15,
    "DiseaseIndex": 0.08,
    "Fertilizer": 0.45,
    "Irrigation": 3.6,
    "PruningFreq": 1,
    "OrganicMatter": 0.8
}

# ============================================================
# SINGLE-CLICK PREDICTION FUNCTION
# ============================================================

def predict_yield_single_click(sensor_dict):
    x = np.array(list(sensor_dict.values()))
    x_scaled = scaler.transform(x.reshape(1, -1))

    # Repeat same reading for full window
    window = np.repeat(x_scaled, WINDOW_SIZE, axis=0)

    tensor = torch.tensor(window, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(tensor).item()

    return round(prediction, 2)

# ============================================================
# RUN PREDICTION
# ============================================================

predicted_yield = predict_yield_single_click(single_sensor_reading)

print(f"\n🌱 Predicted Yield (Single Click): {predicted_yield} kg/plant\n")
