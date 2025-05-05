from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model
model = joblib.load("greenmind_model.joblib")

class Telemetry(BaseModel):
    temperature: float
    humidity: float
    soilMoisture: float

@app.post("/predict")
async def predict(data: Telemetry):
    features = [[data.temperature, data.humidity, data.soilMoisture]]
    prediction = model.predict(features)[0]

    # Determine prediction type
    if isinstance(prediction, str):
        # Handle single label case (e.g., "fan_on")
        fan = "on" if "fan_on" in prediction else "off"
        pump = "on" if "pump_on" in prediction else "off"
    elif isinstance(prediction, (list, tuple)):
        # Handle multi-output binary classification (e.g., [1, 0])
        fan = "on" if prediction[0] == 1 else "off"
        pump = "on" if prediction[1] == 1 else "off"
    else:
        fan = "unknown"
        pump = "unknown"

    return {"fanStatus": fan, "pumpStatus": pump}
