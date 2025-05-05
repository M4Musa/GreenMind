from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins for testing (you can tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app = FastAPI()

model = joblib.load("greenmind_model.joblib")


class Telemetry(BaseModel):
    temperature: float
    humidity: float
    soilMoisture: float

@app.post("/predict")
async def predict(data: Telemetry):
    features = [[data.temperature, data.humidity, data.soilMoisture]]
    prediction = model.predict(features)[0]

    # SAFELY interpret prediction output
    if isinstance(prediction, str):
        fan = "on" if "fan_on" in prediction else "off"
        pump = "on" if "pump_on" in prediction else "off"
    elif isinstance(prediction, (list, tuple)):
        fan = "on" if prediction[0] == 1 else "off"
        pump = "on" if prediction[1] == 1 else "off"
    else:
        fan = pump = "unknown"

    return {"fanStatus": fan, "pumpStatus": pump}
