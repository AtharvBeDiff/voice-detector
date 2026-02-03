import io
import numpy as np
import soundfile as sf
import logging
from model import VoiceDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_wav():
    """Create a 5-second dummy sine wave audio in memory."""
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    # Generate a simple sine wave (440 Hz)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='WAV')
    buffer.seek(0)
    return buffer

def test_pipeline():
    logger.info("Starting internal pipeline test...")
    
    try:
        # 1. Initialize Detector
        logger.info("Initializing VoiceDetector...")
        detector = VoiceDetector()
        
        # 2. Create Dummy Audio
        logger.info("Creating dummy audio...")
        audio = create_dummy_wav()
        
        # 3. Run Detection
        logger.info("Running detection...")
        result = detector.detect(audio)
        
        # 4. Print Results
        logger.info("Detection Result:")
        print(result)
        
        # Basic assertions
        assert "classification" in result
        assert "confidence_score" in result
        assert "explanation" in result
        
        logger.info("Test PASSED successfully.")
        
    except Exception as e:
        logger.error(f"Test FAILED: {e}")
        raise

if __name__ == "__main__":
    test_pipeline()
