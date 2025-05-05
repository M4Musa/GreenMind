from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging
import sys
from pathlib import Path
import traceback

# Configure logging to output to stdout for Azure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
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
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Directory contents: {os.listdir('.')}")
        logger.info(f"Attempting to load model from: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at path: {MODEL_PATH}")
            return
            
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        model = None

@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "working_directory": os.getcwd(),
            "directory_contents": os.listdir('.')
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(data: Telemetry):
    try:
        if model is None:
            logger.error("Model is not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        logger.info(f"📥 Received data: {data}")
        features = [[data.temperature, data.humidity, data.soilMoisture]]
        logger.info(f"📊 Features: {features}")
        
        try:
            prediction = model.predict(features)[0]
            logger.info(f"🔮 Prediction: {prediction}")
        except Exception as pred_error:
            logger.error(f"Prediction error: {str(pred_error)}")
            logger.error(f"Model type: {type(model)}")
            logger.error(f"Features shape: {len(features)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(pred_error)}")
        
        fan = "on" if prediction[0] == 1 else "off"
        pump = "on" if prediction[1] == 1 else "off"
        
        return {"fanStatus": fan, "pumpStatus": pump}
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))