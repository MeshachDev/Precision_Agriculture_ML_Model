import numpy as np
import pandas as pd

def generate_bangalore_dragonfruit(n=10000):
    data = {
        "Temperature": np.random.uniform(18, 34, n),        # Bangalore climate
        "Humidity": np.random.uniform(40, 75, n),
        "Rainfall": np.random.uniform(0, 35, n),
        "SunlightHours": np.random.uniform(6, 9, n),
        "SoilMoisture": np.random.uniform(25, 55, n),
        "SoilPH": np.random.uniform(6.0, 7.2),
        "WindSpeed": np.random.uniform(4, 12),

        "PlantAge": np.random.uniform(6, 48, n),
        "Branches": np.random.randint(12, 40, n),
        "StemDiameter": np.random.uniform(3.2, 5.8, n),
        "FlowerCount": np.random.randint(20, 70, n),
        "FruitPrev": np.random.randint(8, 25, n),

        "PestIndex": np.random.randint(0, 4, n),
        "DiseaseIndex": np.random.randint(0, 3, n),

        "Fertilizer": np.random.uniform(0.8, 1.8, n),
        "Irrigation": np.random.uniform(2, 6, n),
        "Pruning": np.random.randint(1, 3, n),
        "OrganicMatter": np.random.uniform(0.5, 1.5, n),
    }

    df = pd.DataFrame(data)

    # Bangalore-tuned Yield Model
    df["Yield"] = (
        0.18 * df["FlowerCount"]
        + 0.22 * df["FruitPrev"]
        + 0.14 * df["Fertilizer"]
        + 0.07 * df["Irrigation"]
        - 0.30 * df["PestIndex"]
        - 0.22 * df["DiseaseIndex"]
        + np.random.normal(0, 1.1, n)
    )

    return df


if __name__ == "__main__":
    df = generate_bangalore_dragonfruit(4000)
    df.to_csv("dragonfruit_bangalore_dataset.csv", index=False)
    print("Bangalore dataset saved as dragonfruit_bangalore_dataset.csv")