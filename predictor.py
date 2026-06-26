"""
DermoScan - CNN Prediction module
Handles model loading and image prediction.
Falls back to a mock predictor if the model file is not yet trained.
"""

import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "skin_model.h5")
IMG_SIZE   = (224, 224)
CLASSES    = ["acne", "eczema", "normal", "psoriasis", "ringworm"]

# Disease information shown in result page
DISEASE_INFO = {
    "acne": {
        "description": "Acne vulgaris is a chronic inflammatory condition of the pilosebaceous units. "
                       "It typically presents on the face, neck, chest, and back with comedones, "
                       "papules, pustules, or nodules.",
        "treatment":   "Topical retinoids, benzoyl peroxide, antibiotics, or oral isotretinoin "
                       "for severe cases. Consult a dermatologist.",
        "severity":    "moderate",
        "icon":        "fa-bacteria",
    },
    "eczema": {
        "description": "Atopic dermatitis (eczema) causes dry, red, and itchy skin patches. "
                       "It is a chronic condition that tends to flare periodically.",
        "treatment":   "Moisturisers, topical corticosteroids, and avoiding triggers. "
                       "Prescription immunomodulators for persistent cases.",
        "severity":    "moderate",
        "icon":        "fa-allergies",
    },
    "normal": {
        "description": "No signs of skin infection were detected. Your skin appears healthy.",
        "treatment":   "Maintain good skincare hygiene, stay hydrated, and use sunscreen daily.",
        "severity":    "none",
        "icon":        "fa-check-circle",
    },
    "psoriasis": {
        "description": "Psoriasis is a chronic autoimmune condition causing rapid skin-cell buildup, "
                       "leading to scaling on the skin surface.",
        "treatment":   "Topical treatments, phototherapy, or systemic medications. "
                       "Consult a dermatologist for a personalised plan.",
        "severity":    "high",
        "icon":        "fa-virus",
    },
    "ringworm": {
        "description": "Tinea corporis (ringworm) is a contagious fungal infection characterised by "
                       "a ring-shaped rash with raised, scaly borders.",
        "treatment":   "Topical antifungal creams (clotrimazole, terbinafine). Oral antifungals "
                       "for extensive or resistant cases.",
        "severity":    "moderate",
        "icon":        "fa-circle-notch",
    },
}


class SkinPredictor:
    """Loads the trained Keras model and runs inference on a single image."""

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[Predictor] Model not found at {MODEL_PATH}. Run train_model.py first.")
            return
        try:
            from tensorflow.keras.models import load_model  # type: ignore
            self.model = load_model(MODEL_PATH)
            print(f"[Predictor] Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[Predictor] Failed to load model: {e}")

    def _preprocess(self, image_path: str) -> np.ndarray:
        """Read, resize, and normalise an image to model input format."""
        try:
            import cv2  # type: ignore
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
        except Exception:
            # Fallback to Pillow if OpenCV not available
            from PIL import Image
            img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
            img = np.array(img)
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)  # shape (1, 224, 224, 3)

    def predict(self, image_path: str) -> dict:
        """
        Returns a dict:
          predicted_class : str
          confidence      : float  (0-1)
          all_scores      : {class_name: float}
          info            : disease info dict
        """
        if self.model is None:
            return self._mock_predict(image_path)

        img = self._preprocess(image_path)
        preds = self.model.predict(img, verbose=0)[0]          # shape (5,)
        idx = int(np.argmax(preds))
        predicted_class = CLASSES[idx]
        confidence = float(preds[idx])
        all_scores = {cls: round(float(preds[i]) * 100, 2) for i, cls in enumerate(CLASSES)}

        return {
            "predicted_class": predicted_class,
            "confidence":       confidence,
            "confidence_pct":   round(confidence * 100, 1),
            "all_scores":       all_scores,
            "info":             DISEASE_INFO.get(predicted_class, {}),
        }

    # ------------------------------------------------------------------
    # Mock predictor – used when model.h5 is not yet available so that
    # the web app can still be demonstrated / UI-tested.
    # ------------------------------------------------------------------
    def _mock_predict(self, image_path: str) -> dict:
        """Return a deterministic fake prediction based on filename hash."""
        seed = abs(hash(os.path.basename(image_path))) % 100
        idx  = seed % len(CLASSES)
        predicted_class = CLASSES[idx]

        rng    = np.random.default_rng(seed)
        scores = rng.dirichlet(np.ones(len(CLASSES)) * 2)
        # Boost the chosen class to make it look realistic
        scores[idx] = scores[idx] + 0.4
        scores       = scores / scores.sum()

        confidence = float(scores[idx])
        all_scores = {cls: round(float(scores[i]) * 100, 2) for i, cls in enumerate(CLASSES)}

        return {
            "predicted_class": predicted_class,
            "confidence":       confidence,
            "confidence_pct":   round(confidence * 100, 1),
            "all_scores":       all_scores,
            "info":             DISEASE_INFO.get(predicted_class, {}),
            "mock":             True,
        }
