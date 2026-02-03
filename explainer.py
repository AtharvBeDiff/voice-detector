def explain_result(score: float, pitch_entropy: float) -> str:
    """
    Generates a text explanation based on the confidence score and signal traits.
    
    Args:
        score (float): The final probability score of being AI-generated (0.0 to 1.0).
        pitch_entropy (float): The calculated pitch entropy.
        
    Returns:
        str: A human-readable explanation.
    """
    explanation_parts = []
    
    # Confidence-based explanation
    if score > 0.90:
        explanation_parts.append("Strong indicators of neural vocoder artifacts detected.")
    elif score > 0.70:
        explanation_parts.append("Inconsistencies in spectral balance suggest potential synthesis.")
    elif score > 0.50:
        explanation_parts.append("Ambiguous signal traits, but leaning towards artificial generation.")
    else:
        explanation_parts.append("Natural acoustic properties detected.")

    # Pitch stability explanation (Entropy threshold is heuristic)
    # Lower entropy means MORE stable (less variance), which is typical of AI
    if pitch_entropy < 3.0: 
        explanation_parts.append("Voice lacks natural human micro-tremors (unnaturally stable pitch).")
    elif pitch_entropy > 5.0 and score < 0.5:
        explanation_parts.append("High pitch variance consistent with natural human speech.")

    return " ".join(explanation_parts)
