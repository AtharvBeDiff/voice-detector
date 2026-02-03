import io
from preprocessor import preprocess_audio
from classifier import HybridClassifier
from explainer import explain_result

class VoiceDetector:
    def __init__(self):
        self.classifier = HybridClassifier()

    def detect(self, audio_file: io.BytesIO) -> dict:
        """
        Orchestrates the detection pipeline: Preprocessing -> Classification -> Explanation.
        
        Args:
            audio_file (io.BytesIO): The audio file stream.
            
        Returns:
            dict: Structured result ready for API response.
        """
        # 1. Preprocessing
        full_audio, chunks = preprocess_audio(audio_file)
        
        if not chunks:
             # Handle too short audio logic if needed, preprocessor usually handles it by padding
             pass

        # 2. Hybrid Classification
        result_metrics = self.classifier.hybrid_predict(full_audio, chunks)
        final_score = result_metrics["score"]
        pitch_entropy = result_metrics.get("pitch_entropy", 0.0)

        # 3. Determine Label
        classification = "AI-Generated" if final_score > 0.5 else "Human"

        # 4. Generate Explanation
        explanation = explain_result(final_score, pitch_entropy)

        return {
            "classification": classification,
            "confidence_score": round(final_score, 4),
            "explanation": explanation,
            # detailed_metrics could be added if needed for debugging
        }
