"""
AURAFORGE Training Pipeline
Train and evaluate the audio AI detection model.

Usage:
    python ml/train.py --ai_dir data/ai --human_dir data/human --output model.pkl
"""

import os
import argparse
import numpy as np
import joblib
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score
)
from sklearn.pipeline import Pipeline

from features import extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auraforge.train")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def load_dataset(ai_dir: str, human_dir: str):
    """
    Load audio files from AI and human directories.
    Labels: AI=1, Human=0
    
    Args:
        ai_dir: Directory containing AI-generated audio files
        human_dir: Directory containing human-made audio files
    
    Returns:
        X: feature matrix (n_samples, 54)
        y: labels (n_samples,)
        file_names: list of file names
    """
    X, y, file_names = [], [], []

    for label, directory in [(1, ai_dir), (0, human_dir)]:
        label_name = "AI" if label == 1 else "Human"
        files = [
            f for f in Path(directory).iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        logger.info(f"Loading {len(files)} {label_name} files from {directory}")

        for f in files:
            try:
                features = extract_features(str(f), max_seconds=30)
                X.append(features)
                y.append(label)
                file_names.append(f.name)
                logger.debug(f"  ✓ {f.name}")
            except Exception as e:
                logger.warning(f"  ✗ Skipping {f.name}: {e}")

    if not X:
        raise ValueError("No audio files could be loaded. Check your data directories.")

    logger.info(f"\nDataset: {y.count(1)} AI + {y.count(0)} Human = {len(y)} total")

    return np.array(X), np.array(y), file_names


def train_model(X: np.ndarray, y: np.ndarray, model_type: str = "random_forest"):
    """
    Train classification pipeline with scaler + model.
    
    Args:
        X: feature matrix
        y: labels
        model_type: 'random_forest' or 'gradient_boosting'
    
    Returns:
        Trained sklearn Pipeline
    """
    if model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
    else:
        # Default: RandomForest (best for small datasets)
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])

    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """Print detailed evaluation metrics."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "="*50)
    print("AURAFORGE MODEL EVALUATION")
    print("="*50)
    print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train AURAFORGE model")
    parser.add_argument("--ai_dir", required=True, help="Directory of AI audio files")
    parser.add_argument("--human_dir", required=True, help="Directory of human audio files")
    parser.add_argument("--output", default="model.pkl", help="Output model path")
    parser.add_argument("--model_type", default="random_forest",
                        choices=["random_forest", "gradient_boosting"])
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    # Load data
    X, y, _ = load_dataset(args.ai_dir, args.human_dir)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Train
    logger.info(f"Training {args.model_type} model...")
    pipeline = train_model(X_train, y_train, model_type=args.model_type)
    pipeline.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Evaluate
    evaluate_model(pipeline, X_test, y_test)

    # Save
    joblib.dump(pipeline, args.output)
    logger.info(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
