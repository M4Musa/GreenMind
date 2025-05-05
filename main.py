from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

try:
    model = joblib.load("greenmind_model.joblib")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)

class Telemetry(BaseModel):
    temperature: float
    humidity: float
    soilMoisture: float

@app.post("/predict")
async def predict(data: Telemetry):
    try:
        print("📥 Received data:", data)
        features = [[data.temperature, data.humidity, data.soilMoisture]]
        print("📊 Features:", features)
        prediction = model.predict(features)[0]
        print("🔮 Prediction:", prediction)
        fan = "on" if prediction[0] == 1 else "off"
        pump = "on" if prediction[1] == 1 else "off"
        return {"fanStatus": fan, "pumpStatus": pump}
    except Exception as e:
        print("❌ Prediction error:", e)
        return {"error": str(e)}
