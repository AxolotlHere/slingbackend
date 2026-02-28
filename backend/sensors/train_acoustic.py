"""
backend/sensors/train_acoustic.py

CLI script to train the acoustic surface classifier.

Usage:
    python backend/sensors/train_acoustic.py \\
        --bc-audio data/bc_road.wav \\
        --gravel-audio data/gravel_road.wav \\
        --output models/acoustic_model.pkl

    Optional extra classes:
        --wbm-audio data/wbm_road.wav \\
        --concrete-audio data/concrete_road.wav

Training takes < 5 minutes on any CPU.
Model is saved as a scikit-learn RandomForest pickle.
"""

import argparse
import sys
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_audio_windows(path: str, sr: int = 22050, window_sec: float = 1.0) -> list[np.ndarray]:
    """
    Load a WAV file and split into fixed-length windows.

    Args:
        path:       Path to WAV file.
        sr:         Target sample rate.
        window_sec: Window length in seconds.

    Returns:
        List of numpy arrays, each of length sr * window_sec.
    """
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as exc:
        logger.error(f"Failed to load audio file {path}: {exc}")
        sys.exit(1)

    window_len = int(sr * window_sec)
    windows = []
    for start in range(0, len(audio) - window_len + 1, window_len):
        windows.append(audio[start: start + window_len])

    logger.info(f"  Loaded {path}: {len(windows)} windows of {window_sec}s each.")
    return windows


def main():
    parser = argparse.ArgumentParser(
        description="Train acoustic road surface classifier (RandomForest)."
    )
    parser.add_argument("--bc-audio",       required=False, help="WAV file: Bituminous Concrete road")
    parser.add_argument("--wbm-audio",      required=False, help="WAV file: Water Bound Macadam road")
    parser.add_argument("--gravel-audio",   required=False, help="WAV file: Gravel/Granular road")
    parser.add_argument("--concrete-audio", required=False, help="WAV file: Concrete (Rigid) road")
    parser.add_argument("--output",  default="models/acoustic_model.pkl",
                        help="Output path for trained model (default: models/acoustic_model.pkl)")
    parser.add_argument("--sr",      type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--trees",   type=int, default=100,   help="Number of RF trees (default: 100)")
    args = parser.parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────
    class_files = {
        "BC":       args.bc_audio,
        "WBM":      args.wbm_audio,
        "Granular": args.gravel_audio,
        "Concrete": args.concrete_audio,
    }
    provided = {cls: path for cls, path in class_files.items() if path}

    if len(provided) < 2:
        logger.error("Need at least 2 surface types to train. Provide --bc-audio and at least one other.")
        sys.exit(1)

    # ── Import dependencies ────────────────────────────────────────────────
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        import joblib
    except ImportError as exc:
        logger.error(f"Missing dependency: {exc}. Run: pip install scikit-learn joblib")
        sys.exit(1)

    try:
        import librosa  # noqa: F401
    except ImportError:
        logger.error("librosa not installed. Run: pip install librosa")
        sys.exit(1)

    # Import feature extractor
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from backend.sensors.acoustic_classifier import AcousticSurfaceClassifier

    clf = AcousticSurfaceClassifier(sample_rate=args.sr)

    # ── Build dataset ─────────────────────────────────────────────────────
    X, y = [], []

    logger.info("Extracting audio features...")
    for label, audio_path in provided.items():
        windows = load_audio_windows(audio_path, sr=args.sr)
        for window in windows:
            features = clf.extract_features(window)
            if features is not None:
                X.append(features)
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    logger.info(f"Dataset: {len(X)} samples across {len(provided)} classes: {list(provided.keys())}")

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info(f"Training RandomForest ({args.trees} estimators)...")
    rf = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation (if enough samples)
    if len(X) >= 20:
        cv_scores = cross_val_score(rf, X, y, cv=min(5, len(X) // 4), scoring="accuracy")
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    rf.fit(X, y)
    logger.info("Training complete.")

    # Feature importances
    feature_names = (
        [f"MFCC_{i}" for i in range(26)]
        + ["SpectralCentroid", "SpectralRolloff", "ZCR", "RMS"]
    )
    top_idx = np.argsort(rf.feature_importances_)[::-1][:5]
    logger.info("Top 5 features:")
    for i in top_idx:
        logger.info(f"  {feature_names[i]}: {rf.feature_importances_[i]:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, str(output_path))
    logger.info(f"Model saved to {output_path}")
    logger.info("Done! Load with: AcousticSurfaceClassifier(model_path='models/acoustic_model.pkl')")


if __name__ == "__main__":
    main()
