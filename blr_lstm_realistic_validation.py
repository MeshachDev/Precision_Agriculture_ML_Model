import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from collections import deque
import random

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

NUM_FEATURES = len(FEATURE_NAMES)  # MUST be 18

# ============================================================
# LSTM MODEL (IDENTICAL TO TRAINING)
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
# SCALER (SIMULATED – SAME FEATURE SPACE)
# ============================================================

scaler = StandardScaler()
scaler.fit(np.random.uniform(0, 1, size=(2000, NUM_FEATURES)))

# ============================================================
# REALISTIC AGRONOMIC FEATURE GENERATOR (18 FEATURES)
# ============================================================

def generate_realistic_features(day):
    temperature = random.uniform(22, 32)
    humidity = random.uniform(55, 85)
    rainfall = random.uniform(0, 10)
    sunlight = random.uniform(6, 9)
    soil_moisture = random.uniform(30, 60)
    soil_ph = random.uniform(6.0, 7.2)
    wind_speed = random.uniform(1, 6)

    plant_age = min(18, day // 30)
    branches = random.randint(6, 15)
    stem_diameter = random.uniform(2.5, 5.5)

    flower_count = random.randint(3, 15)
    prev_fruit_count = max(0, flower_count - random.randint(0, 3))

    pest_index = random.uniform(0, 0.4)
    disease_index = random.uniform(0, 0.3)

    fertilizer = random.uniform(0.2, 0.7)     # kg/plant/month
    irrigation = random.uniform(2, 5)         # liters/day
    pruning_freq = random.randint(0, 2)
    organic_matter = random.uniform(0.3, 1.2)

    return [
        temperature, humidity, rainfall, sunlight, soil_moisture,
        soil_ph, wind_speed, plant_age, branches, stem_diameter,
        flower_count, prev_fruit_count, pest_index, disease_index,
        fertilizer, irrigation, pruning_freq, organic_matter
    ]

# ============================================================
# PHYSIOLOGY-BASED GROUND TRUTH YIELD
# ============================================================

def calculate_true_yield(f):
    (
        _, _, _, sunlight, soil_moisture, _, _, plant_age,
        branches, stem_dia, flowers, prev_fruits,
        pest, disease, fertilizer, irrigation, pruning, organic
    ) = f

    flower_effect = flowers * random.uniform(0.8, 1.2)
    maturity_bonus = min(plant_age / 12, 1.2)
    nutrient_score = (fertilizer * 8 + organic * 4)
    irrigation_score = irrigation * 1.2

    stress_penalty = (pest * 10 + disease * 12)
    structure_bonus = (branches * 0.3 + stem_dia * 0.6)
    climate_bonus = sunlight * 0.6

    base_yield = 5
    yield_kg = (
        base_yield +
        flower_effect +
        nutrient_score +
        irrigation_score +
        structure_bonus +
        climate_bonus
    ) * maturity_bonus - stress_penalty

    return max(5, min(yield_kg, 35))

# ============================================================
# TIME-SERIES BUFFER
# ============================================================

buffer = deque(maxlen=WINDOW_SIZE)

# ============================================================
# VALIDATION LOOP
# ============================================================

print("\n🌱 REALISTIC BLR VALIDATION (18 FEATURES)\n")

for day in range(1, 90):
    features = generate_realistic_features(day)
    buffer.append(features)

    true_yield = calculate_true_yield(features)

    if len(buffer) == WINDOW_SIZE:
        X = np.array(buffer)
        X_scaled = scaler.transform(X)

        tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            predicted_yield = model(tensor).item()

        print(f"Day {day}")
        print(f"True Yield      : {round(true_yield, 2)} kg")
        print(f"Predicted Yield : {round(predicted_yield, 2)} kg")
        print(f"Error           : {round(abs(true_yield - predicted_yield), 2)} kg\n")
