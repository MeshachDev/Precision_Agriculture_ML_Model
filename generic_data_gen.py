import numpy as np
import pandas as pd

def generate_generic_dragonfruit(n=10000):
    data = {
        "Temperature": np.random.normal(29, 2, n),          # °C
        "Humidity": np.random.normal(70, 8, n),             # %
        "Rainfall": np.random.uniform(0, 30, n),            # mm/day
        "SunlightHours": np.random.uniform(6, 10, n),       # hrs
        "SoilMoisture": np.random.uniform(35, 60, n),       # %
        "SoilPH": np.random.uniform(5.8, 7.0, n),
        "WindSpeed": np.random.uniform(5, 15, n),           # km/h

        "PlantAge": np.random.uniform(6, 48, n),            # months
        "Branches": np.random.randint(10, 40, n),
        "StemDiameter": np.random.uniform(3.0, 5.8, n),     # cm
        "FlowerCount": np.random.randint(15, 70, n),
        "FruitPrev": np.random.randint(5, 25, n),

        "PestIndex": np.random.randint(0, 5, n),
        "DiseaseIndex": np.random.randint(0, 5, n),

        "Fertilizer": np.random.uniform(0.5, 1.8, n),       # kg/plant/month
        "Irrigation": np.random.uniform(2.5, 6.0, n),       # L/day
        "Pruning": np.random.randint(0, 4, n),
        "OrganicMatter": np.random.uniform(0.3, 1.5, n),
    }

    df = pd.DataFrame(data)

    # Synthetic Yield Equation (Generic)
    df["Yield"] = (
        0.15 * df["FlowerCount"]
        + 0.22 * df["FruitPrev"]
        + 0.12 * df["Fertilizer"]
        + 0.06 * df["Irrigation"]
        - 0.25 * df["PestIndex"]
        - 0.18 * df["DiseaseIndex"]
        + np.random.normal(0, 1.2, n)
    )

    return df


if __name__ == "__main__":
    df = generate_generic_dragonfruit(5000)
    df.to_csv("dragonfruit_generic_dataset.csv", index=False)
    print("Generic dataset saved as dragonfruit_generic_dataset.csv")