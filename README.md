# Predictive Irrigation Decision Support System

Honours thesis project — University of Technology Sydney
Student: Rei Jiang Wong (14220103)
Supervisor: Mingshan Jia

## Overview
A machine learning-based irrigation decision support system 
that predicts next-day soil moisture and generates irrigation 
recommendations from environmental sensor data.

## Results
- Best model: Gradient Boosting (NSE = 0.873, R² = 0.873)
- Data: NASA POWER MERRA-2, Yanco NSW, 10 years (2016–2025)
- Benchmark: NSE > 0.75 = Very Good (Moriasi et al., 2007)

## How to Run
1. python 00_read_nasapower.py   # Process NASA POWER data
2. python 02_ml_pipeline.py      # Train models, generate figures
3. python 03_predict_today.py    # Run daily recommendation demo

## Requirements
pip install -r requirements.txt

## Data
Download NASA POWER data for Yanco, NSW from:
https://power.larc.nasa.gov/data-access-viewer/
Coordinates: -34.99, 146.42
