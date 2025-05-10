
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Define the input schema using Pydantic
class EnergyInput(BaseModel):
    Capacity_Utilization: float
    Temperature_C: float
    Cloud_Cover: float
    Wind_Speed: float
    Is_Peak_Hour: int
    Is_Weekend: int
    Site_Avg_Output: float
    Site_Temp_Deviation: float
    Temp_Wind_Index: float
    Clear_Sky_Index: float
    Solar_Potential: float
    Utilization_Efficiency: float
    Prev_Energy: float
    Rolling_Mean_3: float
    Rolling_Std_3: float

# Load trained model and scaler
model = joblib.load("gradient_boosting_energy_model.joblib")
scaler = joblib.load("energy_model_scaler.joblib")

app = FastAPI(title="Energy Generation Predictor")

@app.post("/predict")
def predict_energy(data: EnergyInput):
    features = np.array([[ 
        data.Capacity_Utilization,
        data.Temperature_C,
        data.Cloud_Cover,
        data.Wind_Speed,
        data.Is_Peak_Hour,
        data.Is_Weekend,
        data.Site_Avg_Output,
        data.Site_Temp_Deviation,
        data.Temp_Wind_Index,
        data.Clear_Sky_Index,
        data.Solar_Potential,
        data.Utilization_Efficiency,
        data.Prev_Energy,
        data.Rolling_Mean_3,
        data.Rolling_Std_3
    ]])
    
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return {"predicted_energy_output_MWh": round(float(prediction[0]), 2)}

