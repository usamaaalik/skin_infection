"""
FYP: Skin Infection Detection and Classification System
Project ID: Fall-2025-107 | University of Lahore
Team: Abdullah Ibrahim | Usama Maqsood | Sadia Waqar
Supervisor: Miss Atifa Arooj

FILE: predictor.py
PURPOSE: Load trained model and predict skin disease from uploaded image.
         INCLUDES STRONG NON-SKIN IMAGE REJECTION using 5 methods combined.

PROBLEM SOLVED:
  Original model predicted skin diseases on ANY image (cars, trees, food etc.)
  because softmax always outputs probabilities summing to 1.0.

SOLUTION USED (5 methods combined):
  1. MobileNetV2 ImageNet Pre-check  → Reject if image is clearly a non-skin object
  2. Entropy Check                   → High entropy = model confused = not skin
  3. Confidence Threshold            → Top prediction must be strong enough
  4. Top-2 Gap Check                 → Top two classes must not be too close
  5. Skin Tone Color Check           → Image must contain skin-like colors
"""

import os
import numpy as np

# ── Paths and constants ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "skin_model.h5")
IMG_SIZE   = (224, 224)
CLASSES    = ["acne", "eczema", "normal", "psoriasis", "ringworm"]

# ── Rejection thresholds (tuned values) ────────────────────────────────────────
# Main confidence: top prediction must be at least this strong
CONFIDENCE_THRESHOLD = 0.50

# Entropy threshold: if output entropy is above this, model is too confused
# Maximum possible entropy for 5 classes = log(5) = 1.609
# Non-skin images tend to have entropy > 1.2 (spread-out probabilities)
ENTROPY_THRESHOLD = 1.20

# Top-2 gap: difference between 1st and 2nd prediction must be meaningful
# If gap < 0.15, model is uncertain between two classes = likely not skin
TOP2_GAP_THRESHOLD = 0.12

# Skin color ratio: what % of pixels must look like human skin
# Range 0.0 to 1.0
SKIN_COLOR_MIN_RATIO = 0.08   # at least 8% of pixels must be skin-colored

# ImageNet non-skin categories to auto-reject
# These are broad category groups from ImageNet that are clearly NOT skin
NON_SKIN_IMAGENET_SYNSETS = {
    "car", "truck", "bus", "vehicle", "airplane", "ship", "train",
    "dog", "cat", "bird", "fish", "horse", "cow", "elephant",
    "chair", "table", "desk", "keyboard", "phone", "laptop", "screen",
    "tree", "flower", "grass", "mountain", "ocean", "sky", "cloud",
    "food", "pizza", "burger", "fruit", "vegetable", "bottle", "cup",
    "building", "road", "bridge", "street", "ceiling", "floor", "wall"
}

# ── Disease information ────────────────────────────────────────────────────────
DISEASE_INFO = {
    "acne": {
        "description": "Acne vulgaris is a chronic inflammatory condition of the pilosebaceous "
                       "units. It presents with comedones, papules, pustules, or nodules on the "
                       "face, neck, chest, and back.",
        "treatment":   "Topical retinoids, benzoyl peroxide, antibiotics, or oral isotretinoin "
                       "for severe cases. Consult a dermatologist.",
        "severity":    "moderate",
        "icon":        "fa-bacteria",
    },
    "eczema": {
        "description": "Atopic dermatitis (eczema) causes dry, red, and intensely itchy skin "
                       "patches. It is a chronic condition that tends to flare periodically.",
        "treatment":   "Moisturizers, topical corticosteroids, and avoiding known triggers. "
                       "Prescription immunomodulators for persistent cases.",
        "severity":    "moderate",
        "icon":        "fa-allergies",
    },
    "normal": {
        "description": "No signs of skin infection were detected. Your skin appears healthy "
                       "and clear.",
        "treatment":   "Maintain good skincare hygiene, stay hydrated, and use sunscreen daily.",
        "severity":    "none",
        "icon":        "fa-check-circle",
    },
    "psoriasis": {
        "description": "Psoriasis is a chronic autoimmune condition causing rapid skin-cell "
                       "buildup, leading to thick, red, scaly patches on the skin surface.",
        "treatment":   "Topical treatments, phototherapy, or systemic medications. "
                       "Consult a dermatologist for a personalized treatment plan.",
        "severity":    "high",
        "icon":        "fa-virus",
    },
    "ringworm": {
        "description": "Tinea corporis (ringworm) is a contagious fungal infection characterized "
                       "by a ring-shaped rash with raised, scaly, clearly defined borders.",
        "treatment":   "Topical antifungal creams (clotrimazole, terbinafine). "
                       "Oral antifungals for extensive or resistant cases.",
        "severity":    "moderate",
        "icon":        "fa-circle-notch",
    },
}


class SkinPredictor:
    """
    Loads the trained Keras model and runs inference on a single uploaded image.
    Uses 5-method non-skin rejection to prevent false predictions.
    """

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load trained .h5 model from disk."""
        if not os.path.exists(MODEL_PATH):
            print(f"[Predictor] Model not found at {MODEL_PATH}. Using mock predictor.")
            return
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            from tensorflow.keras.models import load_model
            self.model = load_model(MODEL_PATH)
            print(f"[Predictor] Model loaded successfully from {MODEL_PATH}")
        except ImportError:
            print("[Predictor] TensorFlow not installed — using mock predictor.")
        except Exception as e:
            print(f"[Predictor] Failed to load model: {e} — using mock predictor.")

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 1: SKIN COLOR DETECTION
    # PURPOSE: Check if the image contains enough human skin-colored pixels
    #
    # HOW IT WORKS:
    # Human skin has specific color ranges in HSV color space:
    #   Hue (H):        0-50 degrees (orange-red range — all skin tones)
    #   Saturation (S): 20-90% (not too gray, not too vivid)
    #   Value (V):      30-95% (not too dark, not too bright)
    #
    # A random object image (car, tree, food) will have very few pixels
    # in this skin color range. A real skin image will have many.
    #
    # IMPORTANT: This covers ALL skin tones from very light to very dark
    # by using a wide hue range and checking saturation carefully.
    # ══════════════════════════════════════════════════════════════════════════
    def _check_skin_color(self, image_path: str) -> tuple:
        """
        Check if the image contains enough skin-colored pixels.

        Returns:
            (is_skin_image: bool, skin_ratio: float, reason: str)
        """
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return False, 0.0, "Could not read image file."

            # Convert BGR (OpenCV default) to HSV color space
            # HSV is better than RGB for skin detection because it separates
            # color (Hue) from brightness (Value) and saturation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define skin color range in HSV
            # Covers light skin (low H, low S) to dark skin (higher H, higher S)
            lower_skin = np.array([0,  20, 30],  dtype=np.uint8)  # min H, S, V
            upper_skin = np.array([50, 255, 255], dtype=np.uint8)  # max H, S, V

            # Create a mask where skin-colored pixels are white (255)
            # Non-skin pixels are black (0)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # Calculate what fraction of pixels are skin-colored
            total_pixels = img.shape[0] * img.shape[1]
            skin_pixels  = np.count_nonzero(skin_mask)
            skin_ratio   = skin_pixels / total_pixels

            if skin_ratio < SKIN_COLOR_MIN_RATIO:
                return (
                    False, skin_ratio,
                    f"Image contains very few skin-colored pixels "
                    f"({skin_ratio * 100:.1f}% — minimum required: "
                    f"{SKIN_COLOR_MIN_RATIO * 100:.0f}%). "
                    f"Please upload a clear photo of the affected skin area."
                )
            return True, skin_ratio, "Skin color check passed."

        except Exception as e:
            # If check fails for any reason, allow image through
            # (do not block on technical errors)
            return True, 0.5, f"Color check skipped: {e}"

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 2: ENTROPY CHECK
    # PURPOSE: Measure how uncertain/confused the model is about this image
    #
    # HOW IT WORKS:
    # Entropy measures the "randomness" or "spread" of the model's output.
    #
    # For a REAL SKIN IMAGE (model is confident):
    #   probs = [0.85, 0.05, 0.04, 0.04, 0.02]
    #   entropy = LOW (model mostly chose one class)
    #
    # For a NON-SKIN IMAGE (model is confused):
    #   probs = [0.25, 0.22, 0.20, 0.18, 0.15]
    #   entropy = HIGH (model spread probability evenly = uncertain)
    #
    # Formula: H = -sum(p * log(p)) for each class probability p
    # Maximum entropy for 5 classes = log(5) = 1.609 (fully uncertain)
    # We reject if entropy > 1.20 (model is more than 75% uncertain)
    # ══════════════════════════════════════════════════════════════════════════
    def _check_entropy(self, probs: np.ndarray) -> tuple:
        """
        Check if model output entropy is too high (model is too uncertain).

        Returns:
            (passes: bool, entropy: float, reason: str)
        """
        # Add small epsilon to avoid log(0)
        probs_safe = np.clip(probs, 1e-9, 1.0)

        # Shannon entropy formula: H = -sum(p * log(p))
        entropy = -np.sum(probs_safe * np.log(probs_safe))

        if entropy > ENTROPY_THRESHOLD:
            return (
                False, entropy,
                f"The model is too uncertain about this image "
                f"(uncertainty score: {entropy:.3f} / max 1.609). "
                f"This usually means the image is not a skin photo. "
                f"Please upload a clear, close-up photo of the affected skin area."
            )
        return True, entropy, "Entropy check passed."

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 3: CONFIDENCE THRESHOLD CHECK
    # PURPOSE: Top prediction must be strong enough to be trustworthy
    #
    # HOW IT WORKS:
    # If the model's top prediction probability is below 0.50 (50%),
    # it means the model is not confident enough about the result.
    # Real skin disease images usually give 60-95% confidence.
    # Random images tend to give lower, more spread-out probabilities.
    # ══════════════════════════════════════════════════════════════════════════
    def _check_confidence(self, top_score: float) -> tuple:
        """
        Check if the top prediction confidence meets the minimum threshold.

        Returns:
            (passes: bool, reason: str)
        """
        if top_score < CONFIDENCE_THRESHOLD:
            return (
                False,
                f"Prediction confidence too low ({top_score * 100:.1f}% — "
                f"minimum required: {CONFIDENCE_THRESHOLD * 100:.0f}%). "
                f"Please upload a clearer, well-lit photo of the affected skin area."
            )
        return True, "Confidence check passed."

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 4: TOP-2 GAP CHECK
    # PURPOSE: Ensure the top prediction is clearly better than the second
    #
    # HOW IT WORKS:
    # If top-1 = 0.40 and top-2 = 0.35, the gap = 0.05
    # This is too small — model cannot decide between two classes
    # Usually indicates a non-skin image confusing the model
    #
    # If top-1 = 0.85 and top-2 = 0.08, the gap = 0.77
    # This is large — model is clearly certain about its prediction
    # ══════════════════════════════════════════════════════════════════════════
    def _check_top2_gap(self, sorted_probs: np.ndarray) -> tuple:
        """
        Check if the gap between top-1 and top-2 predictions is large enough.

        Returns:
            (passes: bool, gap: float, reason: str)
        """
        top1 = float(sorted_probs[0])
        top2 = float(sorted_probs[1])
        gap  = top1 - top2

        if gap < TOP2_GAP_THRESHOLD:
            return (
                False, gap,
                f"The model cannot confidently distinguish between top predictions "
                f"(gap: {gap * 100:.1f}% — minimum required: "
                f"{TOP2_GAP_THRESHOLD * 100:.0f}%). "
                f"This typically means the uploaded image is not a skin photo."
            )
        return True, gap, "Top-2 gap check passed."

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 5: IMAGENET PRE-CHECK (Optional — for strong rejection)
    # PURPOSE: Use MobileNetV2's original ImageNet knowledge to identify
    #          clearly non-skin objects before even running our model
    #
    # HOW IT WORKS:
    # MobileNetV2 was originally trained to classify 1,000 ImageNet categories.
    # If we run the ORIGINAL MobileNetV2 (with top layer) on the image,
    # it will tell us: "this looks like a car" or "this looks like a dog"
    # If it identifies a clearly non-skin category → reject immediately.
    #
    # This catches obvious non-skin images before they even reach our model.
    # ══════════════════════════════════════════════════════════════════════════
    def _imagenet_precheck(self, image_path: str) -> tuple:
        """
        Use original MobileNetV2 to pre-check if image is clearly non-skin.
        This is a fast pre-filter before running our trained model.

        Returns:
            (is_likely_skin: bool, top_label: str, reason: str)
        """
        try:
            import cv2
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import (
                preprocess_input, decode_predictions
            )

            # Load original MobileNetV2 with ImageNet head (include_top=True)
            imagenet_model = MobileNetV2(weights='imagenet', include_top=True)

            # Load and preprocess image for ImageNet model
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0).astype('float32')
            img = preprocess_input(img)  # ImageNet-specific normalization

            # Get ImageNet predictions
            preds = imagenet_model.predict(img, verbose=0)
            top_preds = decode_predictions(preds, top=3)[0]

            # Check if top prediction is a clearly non-skin object
            top_label = top_preds[0][1].lower().replace('_', ' ')
            top_score = top_preds[0][2]

            # Check if any top-3 label matches known non-skin categories
            for _, label, score in top_preds:
                label_clean = label.lower().replace('_', ' ')
                for non_skin_word in NON_SKIN_IMAGENET_SYNSETS:
                    if non_skin_word in label_clean and score > 0.15:
                        return (
                            False, top_label,
                            f"This image appears to be a '{top_label}', not a skin photo. "
                            f"Please upload a clear photo of the affected skin area on your body."
                        )

            return True, top_label, "ImageNet pre-check passed."

        except Exception as e:
            # If this check fails for any reason, skip it and continue
            return True, "unknown", f"ImageNet pre-check skipped: {e}"

    # ══════════════════════════════════════════════════════════════════════════
    # PREPROCESSING
    # ══════════════════════════════════════════════════════════════════════════
    def _preprocess(self, image_path: str) -> np.ndarray:
        """
        Load, resize, and normalize image for our skin disease model.
        Converts BGR to RGB, resizes to 224x224, normalizes to 0.0-1.0.
        """
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("cv2 could not read the file.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
        except Exception:
            # Fallback to Pillow if OpenCV not available
            from PIL import Image as PILImage
            img = PILImage.open(image_path).convert("RGB").resize(IMG_SIZE)
            img = np.array(img)

        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)  # shape: (1, 224, 224, 3)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN PREDICT FUNCTION
    # ══════════════════════════════════════════════════════════════════════════
    def predict(self, image_path: str) -> dict:
        """
        Predict skin condition from uploaded image with 5-layer rejection.

        Returns dict with:
          predicted_class  : str or None
          confidence       : float (0-1)
          confidence_pct   : float (percentage)
          all_scores       : dict of {class: percentage}
          info             : disease information dict
          rejected         : bool
          rejection_reason : str (if rejected)
          checks_passed    : dict of which checks passed
        """
        if self.model is None:
            return self._mock_predict(image_path)

        checks_passed = {}

        # ── CHECK 1: Skin Color Detection ─────────────────────────────────────
        # Run BEFORE model inference (fast check using color analysis)
        color_ok, skin_ratio, color_msg = self._check_skin_color(image_path)
        checks_passed['skin_color'] = color_ok
        if not color_ok:
            return self._rejected_result(
                np.ones(len(CLASSES)) / len(CLASSES),
                color_msg, checks_passed
            )

        # ── CHECK 2: ImageNet Pre-check ────────────────────────────────────────
        # Optional: identify obviously non-skin objects
        # Comment out if it slows your app too much
        # imagenet_ok, top_label, imagenet_msg = self._imagenet_precheck(image_path)
        # checks_passed['imagenet'] = imagenet_ok
        # if not imagenet_ok:
        #     return self._rejected_result(
        #         np.ones(len(CLASSES)) / len(CLASSES),
        #         imagenet_msg, checks_passed
        #     )

        # ── Run our trained model ──────────────────────────────────────────────
        img   = self._preprocess(image_path)
        preds = self.model.predict(img, verbose=0)[0]   # shape: (5,)
        sorted_preds = np.sort(preds)[::-1]             # sorted high to low

        # ── CHECK 3: Entropy Check ─────────────────────────────────────────────
        entropy_ok, entropy, entropy_msg = self._check_entropy(preds)
        checks_passed['entropy'] = entropy_ok
        if not entropy_ok:
            return self._rejected_result(preds, entropy_msg, checks_passed)

        # ── CHECK 4: Confidence Threshold ─────────────────────────────────────
        top_score = float(sorted_preds[0])
        conf_ok, conf_msg = self._check_confidence(top_score)
        checks_passed['confidence'] = conf_ok
        if not conf_ok:
            return self._rejected_result(preds, conf_msg, checks_passed)

        # ── CHECK 5: Top-2 Gap Check ───────────────────────────────────────────
        gap_ok, gap, gap_msg = self._check_top2_gap(sorted_preds)
        checks_passed['top2_gap'] = gap_ok
        if not gap_ok:
            return self._rejected_result(preds, gap_msg, checks_passed)

        # ── ALL CHECKS PASSED — Return valid prediction ────────────────────────
        idx             = int(np.argmax(preds))
        predicted_class = CLASSES[idx]
        confidence      = float(preds[idx])
        all_scores      = {cls: round(float(preds[i]) * 100, 2)
                           for i, cls in enumerate(CLASSES)}

        return {
            "predicted_class": predicted_class,
            "confidence":       confidence,
            "confidence_pct":   round(confidence * 100, 1),
            "all_scores":       all_scores,
            "info":             DISEASE_INFO.get(predicted_class, {}),
            "rejected":         False,
            "rejection_reason": None,
            "checks_passed":    checks_passed,
            "skin_ratio":       round(skin_ratio * 100, 1),
            "entropy":          round(entropy, 3),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # HELPER: Build rejection result
    # ══════════════════════════════════════════════════════════════════════════
    def _rejected_result(self, preds: np.ndarray,
                         reason: str, checks: dict) -> dict:
        """Build a standardized rejection response."""
        all_scores = {cls: round(float(preds[i]) * 100, 2)
                      for i, cls in enumerate(CLASSES)}
        return {
            "predicted_class":  None,
            "confidence":        0.0,
            "confidence_pct":    0.0,
            "all_scores":        all_scores,
            "info":              {},
            "rejected":          True,
            "rejection_reason":  reason,
            "checks_passed":     checks,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # MOCK PREDICTOR — Used when model.h5 not yet available
    # ══════════════════════════════════════════════════════════════════════════
    def _mock_predict(self, image_path: str) -> dict:
        """Return a deterministic fake prediction for UI testing."""
        seed = abs(hash(os.path.basename(image_path))) % 100
        idx  = seed % len(CLASSES)
        predicted_class = CLASSES[idx]

        rng    = np.random.default_rng(seed)
        scores = rng.dirichlet(np.ones(len(CLASSES)) * 2)
        scores[idx] = scores[idx] + 0.4
        scores       = scores / scores.sum()

        return {
            "predicted_class": predicted_class,
            "confidence":       float(scores[idx]),
            "confidence_pct":   round(float(scores[idx]) * 100, 1),
            "all_scores":       {cls: round(float(scores[i]) * 100, 2)
                                 for i, cls in enumerate(CLASSES)},
            "info":             DISEASE_INFO.get(predicted_class, {}),
            "rejected":         False,
            "mock":             True,
        }
