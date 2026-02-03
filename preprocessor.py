import librosa
import numpy as np
import io
import soundfile as sf

TARGET_SR = 16000
CHUNK_DURATION = 3.0  # seconds

def preprocess_audio(audio_file: io.BytesIO) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Loads audio, normalizes it, and splits it into 3-second chunks.
    
    Args:
        audio_file (io.BytesIO): The audio file object.
        
    Returns:
        tuple: (full_audio_array, list_of_chunks)
    """
    # Load audio with librosa (resamples to 16kHz automatically)
    # librosa.load can handle mp3/wav/etc if ffmpeg/backend is available.
    try:
        y, sr = librosa.load(audio_file, sr=TARGET_SR, mono=True)
    except Exception as e:
         raise ValueError(f"Failed to load audio: {e}")

    # Trim silence (removes leading/trailing silence)
    # top_db=20 is a common default, can be adjusted
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalize amplitude
    y_norm = librosa.util.normalize(y_trimmed)

    # Calculate chunk size in samples
    chunk_samples = int(CHUNK_DURATION * TARGET_SR)
    
    chunks = []
    # Pad if shorter than one chunk
    if len(y_norm) < chunk_samples:
        y_padded = np.pad(y_norm, (0, chunk_samples - len(y_norm)), mode='constant')
        chunks.append(y_padded)
    else:
        # Split into chunks (overlapping could be added, but simple splitting for now)
        for i in range(0, len(y_norm), chunk_samples):
            chunk = y_norm[i : i + chunk_samples]
            # Ensure last chunk is full length if needed, or just discard/pad
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            chunks.append(chunk)
            
    return y_norm, chunks
