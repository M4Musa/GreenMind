from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# More robust model loading with better error handling
try:
    # Try different possible paths for the model
    model_paths = [
        "greenmind_model.joblib",
        "./greenmind_model.joblib",
        os.path.join(os.path.dirname(__file__), "greenmind_model.joblib")
    ]
    
    model = None
    for path in model_paths:
        try:
            logger.info(f"Attempting to load model from: {path}")
            if os.path.exists(path):
                model = joblib.load(path)
                logger.info(f"✅ Model loaded successfully from {path}")
                break
            else:
                logger.warning(f"Path does not exist: {path}")
        except Exception as e:
            logger.error(f"Failed to load from {path}: {str(e)}")
    
    if model is None:
        logger.error("❌ Could not load model from any path")
except Exception as e:
    logger.error(f"❌ Error in model loading process: {str(e)}")
    model = None

class Telemetry(BaseModel):
    temperature: float
    humidity: float
    soilMoisture: float

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "API is running", "model_loaded": model is not None}

@app.post("/predict")
async def predict(data: Telemetry):
    try:
        if model is None:
            logger.error("Prediction attempted but model is not loaded")
            raise HTTPException(status_code=500, detail="Model not loaded")
        
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