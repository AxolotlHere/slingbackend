"""
backend/sensors/acoustic_classifier.py

Channel 4: Microphone → Road Surface Type Classification

Road surface type produces a characteristic tire-road contact noise:
  - BC (Bituminous Concrete): Quiet, low rumble (dominant 200-800 Hz)
  - WBM (Water Bound Macadam): Characteristic crunch (1-3 kHz peaks)
  - Granular / Gravel: Broadband noise, high ZCR
  - Concrete (Rigid): Very low noise, slight whine

Features: MFCC (26) + Spectral Centroid + Spectral Rolloff + ZCR + RMS
Model: Random Forest (fast, interpretable, CPU-only, trainable in 5 min)
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Attempt librosa import (optional for feature extraction)
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.warning("librosa not installed — acoustic classification disabled.")

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


class AcousticSurfaceClassifier:
    """
    Classifies road surface type from tire-road contact noise.

    Supported surface classes: BC, WBM, Granular, Concrete
    Falls back to Unknown if model not loaded or librosa unavailable.

    Usage:
        clf = AcousticSurfaceClassifier(model_path="models/acoustic_model.pkl")
        result = clf.classify(audio_chunk_1sec)
    """

    SURFACE_CLASSES = ["BC", "WBM", "Granular", "Concrete"]
    FEATURE_SIZE = 30  # 26 MFCCs + 4 spectral/energy features

    def __init__(
        self,
        sample_rate: int = 22050,
        model_path: Optional[str] = None,
    ):
        self.sr = sample_rate
        self.model = None

        if model_path and _JOBLIB_AVAILABLE:
            self._load_model(model_path)

    def _load_model(self, path: str) -> bool:
        """Load pre-trained sklearn model from disk."""
        model_file = Path(path)
        if not model_file.exists():
            logger.warning(
                f"Acoustic model not found at {path}. "
                "Run backend/sensors/train_acoustic.py to train a model."
            )
            return False
        try:
            self.model = joblib.load(str(model_file))
            logger.info(f"Acoustic model loaded from {path}.")
            return True
        except Exception as exc:
            logger.error(f"Failed to load acoustic model: {exc}")
            return False

    def extract_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract acoustic features from a 1-second audio window.

        Args:
            audio: 1D float32/float64 array at self.sr Hz.
                   Should be ~1 second of audio (sr samples).

        Returns:
            Feature vector of shape (FEATURE_SIZE,), or None if extraction fails.
        """
        if not _LIBROSA_AVAILABLE:
            return None

        try:
            audio = audio.astype(np.float32)

            # Normalise amplitude
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                audio = audio / max_amp

            # ── MFCC (26 coefficients) ────────────────────────────────────
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=26)
            mfcc_mean = mfcc.mean(axis=1)  # shape (26,)

            # ── Spectral features ─────────────────────────────────────────
            spectral_centroid = float(
                librosa.feature.spectral_centroid(y=audio, sr=self.sr).mean()
            )
            spectral_rolloff = float(
                librosa.feature.spectral_rolloff(y=audio, sr=self.sr, roll_percent=0.85).mean()
            )

            # ── Temporal features ─────────────────────────────────────────
            zcr = float(librosa.feature.zero_crossing_rate(audio).mean())
            rms = float(librosa.feature.rms(y=audio).mean())

            feature_vec = np.concatenate([
                mfcc_mean,                              # 26 features
                [spectral_centroid, spectral_rolloff,   # 2 features
                 zcr, rms],                             # 2 features → total 30
            ])
            return feature_vec

        except Exception as exc:
            logger.debug(f"Feature extraction failed: {exc}")
            return None

    def classify(self, audio_chunk: np.ndarray) -> dict:
        """
        Classify road surface from an audio chunk.

        Args:
            audio_chunk: 1D float array of audio samples at self.sr Hz.

        Returns:
            dict with keys:
                surface_type_acoustic (str): e.g. "BC", "WBM", "Granular", "Unknown"
                confidence (float): 0.0–1.0
                probabilities (dict): per-class probabilities (if model loaded)
        """
        features = self.extract_features(audio_chunk)

        if features is None:
            return {
                "surface_type_acoustic": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
            }

        if self.model is None:
            return {
                "surface_type_acoustic": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
                "note": "No acoustic model loaded. Run train_acoustic.py first.",
            }

        try:
            features_2d = features.reshape(1, -1)
            prediction = self.model.predict(features_2d)[0]
            proba = self.model.predict_proba(features_2d)[0]
            classes = list(self.model.classes_)

            return {
                "surface_type_acoustic": str(prediction),
                "confidence": float(max(proba)),
                "probabilities": {cls: float(p) for cls, p in zip(classes, proba)},
            }
        except Exception as exc:
            logger.error(f"Acoustic classification inference failed: {exc}")
            return {
                "surface_type_acoustic": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
            }

    @property
    def is_ready(self) -> bool:
        """True if model is loaded and librosa is available."""
        return _LIBROSA_AVAILABLE and self.model is not None
