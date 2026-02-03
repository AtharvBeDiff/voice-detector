import io
import pytest
import base64
from utils import decode_audio

def test_decode_valid_wav():
    # Minimal WAV header + PCM data
    # RIFF header
    # Just a small valid base64 string rep
    dummy_data = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVEfmt " + b"\x10\x00\x00\x00" + b"\x01\x00" + b"\x01\x00" + b"\x44\xac\x00\x00" + b"\x88\x58\x01\x00" + b"\x02\x00" + b"\x10\x00" + b"data" + b"\x00\x00\x00\x00"
    b64_str = base64.b64encode(dummy_data).decode('utf-8')
    
    decoded = decode_audio(b64_str)
    assert isinstance(decoded, io.BytesIO)
    assert decoded.read(4) == b"RIFF"

def test_decode_with_header():
    dummy_data = b"DATA"
    b64_str = "data:audio/wav;base64," + base64.b64encode(dummy_data).decode('utf-8')
    
    decoded = decode_audio(b64_str)
    assert decoded.read() == dummy_data

def test_invalid_base64():
    with pytest.raises(ValueError):
        decode_audio("!!!INVALID!!!")
