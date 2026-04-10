
#AURAFORGE: Decision Engine
#Maps AI probability score: business action (ALLOW / FLAG / BLOCK).




def decide(score: float) -> str:
  
    if score < 0.40:
        return "ALLOW"
    elif score < 0.70:
        return "FLAG"
    else:
        return "BLOCK"


def confidence_label(score: float) -> str:


    """
    Returns: "HIGH" | "MEDIUM" | "LOW"
    Confidence is highest at extremes, lowest near decision boundaries.
    
    """

    
    if score < 0.20 or score > 0.85:
        return "HIGH"
    elif score < 0.35 or score > 0.70:
        return "MEDIUM"
    else:
        return "LOW"
