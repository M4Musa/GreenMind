from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GreenMind API")

# Get the absolute path to the model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greenmind_model.joblib")
model = None

class Telemetry(BaseModel):
    temperature: float
    humidity: float
    soilMoisture: float

@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info(f"Attempting to load model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        model = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(data: Telemetry):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        logger.info(f"📥 Received data: {data}")
        features = [[data.temperature, data.humidity, data.soilMoisture]]
        logger.info(f"📊 Features: {features}")
        
        prediction = model.predict(features)[0]
        logger.info(f"🔮 Prediction: {prediction}")
        
        fan = "on" if prediction[0] == 1 else "off"
        pump = "on" if prediction[1] == 1 else "off"
        
        return {"fanStatus": fan, "pumpStatus": pump}
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))