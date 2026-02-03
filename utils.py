import base64
import io
import binascii

def decode_audio(audio_base64: str) -> io.BytesIO:
    """
    Decodes a base64 string into an in-memory BytesIO object.
    
    Args:
        audio_base64 (str): The base64 encoded audio string.
        
    Returns:
        io.BytesIO: The decoded audio data.
        
    Raises:
        ValueError: If the input is not a valid base64 string.
    """
    try:
        # Strip header if present (e.g., "data:audio/wav;base64,")
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")[1]
            
        decoded_data = base64.b64decode(audio_base64, validate=True)
        return io.BytesIO(decoded_data)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 encoding: {str(e)}")
