#loads trained model and runs prediction

import os
import numpy as np
import joblib
import logging
from pathlib import Path

logger = logging.getLogger("auraforge.inference")

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")


class AuraForgeModel:
    
    #Wrapper around trained sklearn pipeline.
    #Designed to be loaded once at startup and reused.

    

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.is_loaded = False

    def load(self):

        """Load model from disk. Call at API startup."""

        if not Path(self.model_path).exists():
            logger.warning(
                f"Model not found at {self.model_path}. "
                
            )
            self.is_loaded = False
            return

        self.pipeline = joblib.load(self.model_path)
        self.is_loaded = True
        logger.info(f"Model loaded from {self.model_path}")

    def predict(self, features: np.ndarray) -> float:

        
        """
        Run inference on a feature vector.
        
        Args:
            features: np.ndarray of shape (54,)
        
        Returns:
            float: AI probability score (0.0 - 1.0)

        """

        if not self.is_loaded:
            raise RuntimeError(

                "Model not loaded."
            )

        features_2d = features.reshape(1, -1)
        probability = self.pipeline.predict_proba(features_2d)[0][1]
        return float(probability)

    def predict_batch(self, feature_matrix: np.ndarray) -> list[float]:

        """Batch prediction for multiple tracks."""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        probabilities = self.pipeline.predict_proba(feature_matrix)[:, 1]
        return probabilities.tolist()
