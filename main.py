from fastapi import FastAPI, HTTPException, Body, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, validator
import uvicorn
import logging
import os
from utils import decode_audio
from model import VoiceDetector

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security Scheme
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Ideally get this from environment variable
# For this buildathon, we can set a default or require env
EXPECTED_API_KEY = os.getenv("API_KEY", "secret-123-xyz")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

app = FastAPI(title="AI Voice Detector API", description="Hybrid Ensemble AI Voice Detection System")

# Initialize model mechanism
# (Lazy loading could be better for serverless, but detailed here for performance)
try:
    detector = VoiceDetector()
    logger.info("Voice Detector model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Voice Detector: {e}")
    # We don't crash the app, but endpoint will fail
    detector = None

class DetectionRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio file (wav/mp3/etc). Include header like 'data:audio/mp3;base64,' if preferred, or raw base64.")

    @validator('audio_base64')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('audio_base64 cannot be empty')
        return v

class DetectionResponse(BaseModel):
    classification: str = Field(..., description="Classification result: 'AI-Generated' or 'Human'")
    confidence_score: float = Field(..., description="Probability score (0.0 - 1.0) of being AI-Generated")
    explanation: str = Field(..., description="Human-readable explanation of the result")

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": detector is not None}

@app.post("/detect", response_model=DetectionResponse)
async def detect_voice(request: DetectionRequest = Body(...), api_key: str = Depends(get_api_key)):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")
    
    try:
        # Decode base64
        audio_file = decode_audio(request.audio_base64)
        
        # Run detection
        result = detector.detect(audio_file)
        
        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during detection.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
