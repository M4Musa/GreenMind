from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("greenmind_model.joblib")

class Telemetry(BaseModel):
    temperature: float
    humidity: float
    soilMoisture: float

@app.post("/predict")
async def predict(data: Telemetry):
    features = [[data.temperature, data.humidity, data.soilMoisture]]
    prediction = model.predict(features)[0]  # should return [1, 0] or [0, 1]
    fan = "on" if prediction[0] == 1 else "off"
    pump = "on" if prediction[1] == 1 else "off"
    return {"fanStatus": fan, "pumpStatus": pump}
