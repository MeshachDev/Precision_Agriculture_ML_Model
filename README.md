# 🌱 Bangalore-Based Dragon Fruit Yield Prediction Model

A machine learning project for **predicting dragon fruit yield under Bangalore-specific agro-climatic conditions** using synthetic, region-tuned agricultural data.  
This repository is designed for **research, experimentation, and precision agriculture applications**.

---

## 📌 Project Overview

Dragon fruit yield is influenced by a combination of **environmental conditions, plant growth characteristics, pest/disease pressure, and farm management practices**.  
Generic yield models often fail to perform well in Bangalore due to its **semi-tropical high-altitude climate, seasonal monsoons, and red loamy soils**.

This project addresses that gap by providing:
- A **Bangalore-specific dataset generator**
- A **generic (global) dataset generator**
- Clean, ML-ready CSV outputs suitable for regression, LSTM, or Transformer models

---

## 📊 Features Used

### 🌦 Environmental Features
- Temperature (°C)
- Humidity (%)
- Rainfall (mm/day)
- Sunlight Hours (hours/day)
- Soil Moisture (%)
- Soil pH
- Wind Speed (km/h)

### 🌿 Plant & Field Features
- Plant Age (months)
- Number of Branches
- Stem Diameter (cm)
- Flower Count
- Previous Cycle Fruit Count

### 🐛 Biotic Stress Indicators
- Pest Severity Index
- Disease Severity Index

### 🚜 Farm Management Practices
- Fertilizer Application (kg/plant/month)
- Irrigation Volume (liters/day)
- Pruning Frequency
- Organic Matter Added (kg)

---

## 🎯 Target Variable

- **Yield (kg per plant per production cycle)**

---

## 🧠 Modeling Approach

- Supervised machine learning formulation
- Yield learned as a function of **growth dynamics and management decisions**, not climate alone
- Supports:
  - Tabular ML models (Linear Regression, Random Forest, XGBoost)
  - Time-series models (LSTM, GRU, Transformers)
- Easily extendable to real-world **IoT sensor data**

---

## ⚙️ How to Run

### Install Dependencies
```bash
pip install numpy pandas
```
### Generate the Datasets 
```bash
python generate_generic_dataset.py
python generate_bangalore_dataset.py
```
CSV files will be saved automatically in the project directory.
---

## 📦 Output Files :

dragonfruit_generic.csv → Global / location-agnostic dataset
dragonfruit_bangalore.csv → Bangalore-tuned dataset

## 🚀 Applications :

Dragon fruit yield forecasting
Fertilizer and irrigation optimization
Pest and disease impact analysis
Precision agriculture research
Smart farming decision-support systems
