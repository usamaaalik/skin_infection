"""
FYP: Skin Infection Detection and Classification System
Project ID: Fall-2025-107 | University of Lahore
Team: Abdullah Ibrahim | Usama Maqsood | Sadia Waqar
Supervisor: Miss Atifa Arooj

FILE: train_model.py
PURPOSE: Train MobileNetV2 model with NOT_SKIN class for proper rejection.

PERMANENT SOLUTION FOR NON-SKIN REJECTION:
  Added a 6th class: "not_skin"
  This teaches the model what NON-skin images look like.
  When a car/tree/food image is uploaded, model now predicts "not_skin"
  instead of a random disease — giving us a clean and honest rejection.

Dataset structure now needs 6 class folders:
  dataset/
    train/  acne/ eczema/ normal/ psoriasis/ ringworm/ not_skin/
    val/    (same 6 folders)
    test/   (same 6 folders)

Where to get "not_skin" images:
  Download any general image dataset from:
  - kaggle.com/datasets/prasunroy/natural-images (8,000 diverse images)
  - Open Images Dataset (Google)
  - ImageNet sample images
  Take 500-1000 varied images (cars, animals, food, buildings, nature)
  Place in dataset/train/not_skin/ (350), val/not_skin/ (75), test/not_skin/ (75)
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "skin_model.h5")

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 40

# 6 classes now — includes not_skin for rejection
NUM_CLASSES   = 6
CLASSES       = ["acne", "eczema", "normal", "not_skin", "psoriasis", "ringworm"]
# Note: alphabetical order because Keras reads folders alphabetically

LR_PHASE1 = 1e-3
LR_PHASE2 = 5e-5
L2_REG    = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Data generators ──────────────────────────────────────────────────────────
def build_generators():
    """
    Build training, validation, and test data generators.
    Strong augmentation on training data to improve generalization.
    """
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.20,
        height_shift_range=0.20,
        horizontal_flip=True,
        vertical_flip=False,       # skin images don't appear upside-down usually
        zoom_range=0.25,
        brightness_range=[0.7, 1.3],
        shear_range=0.15,
        channel_shift_range=20.0,  # slight color jitter for robustness
        fill_mode="nearest",
    )
    val_test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_data = val_test_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_data = val_test_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    detected_classes = len(train_data.class_indices)
    print(f"\n  Classes detected : {train_data.class_indices}")
    print(f"  NUM_CLASSES      : {detected_classes}")

    return train_data, val_data, test_data, detected_classes


# ── Class weights ─────────────────────────────────────────────────────────────
def get_class_weights(train_data):
    """
    Compute balanced class weights to handle imbalanced class sizes.
    Classes with fewer images get higher weight so model learns them equally.
    """
    labels  = train_data.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    weight_dict = dict(zip(classes, weights))
    print(f"\n  Class weights:")
    for i, cls in enumerate(train_data.class_indices):
        print(f"    {cls:<12}: {weight_dict[i]:.3f}")
    return weight_dict


# ── Model architecture ────────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    """
    Build MobileNetV2 Transfer Learning model with regularized classification head.
    num_classes: 5 (skin only) or 6 (skin + not_skin for rejection training)

    Architecture:
    MobileNetV2 (frozen) → GlobalAveragePooling → BatchNorm →
    Dense(256, L2, ReLU) → Dropout(0.5) →
    Dense(128, L2, ReLU) → Dropout(0.4) →
    Dense(num_classes, softmax)
    """
    reg = regularizers.l2(L2_REG)

    # Load MobileNetV2 pre-trained on ImageNet, without original head
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Frozen in Phase 1

    # Build model using functional API
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))

    # Pass through MobileNetV2 (training=False keeps BatchNorm frozen in phase 1)
    x = base(inputs, training=False)

    # Flatten feature maps to vector
    x = layers.GlobalAveragePooling2D()(x)

    # Normalize layer outputs for stable training
    x = layers.BatchNormalization()(x)

    # First dense block — learn disease features
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)

    # Second dense block — refine features
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.4)(x)

    # Output: one probability per class, softmax sums to 1.0
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="SkinDetection_MobileNetV2")
    return model, base


# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks(phase: int) -> list:
    """
    Training callbacks:
    - ModelCheckpoint: save best model (by val_accuracy)
    - EarlyStopping: stop if no improvement for 7 epochs
    - ReduceLROnPlateau: halve LR when val_loss stagnates
    """
    cb = [
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if phase == 1:
        cb.append(callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1,
        ))
    else:
        cb.append(callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-8, verbose=1,
        ))
    return cb


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_history(h1, h2):
    """Save combined Phase 1 + Phase 2 training curves as PNG."""
    acc   = h1.history["accuracy"]     + h2.history["accuracy"]
    val   = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]         + h2.history["loss"]
    vloss = h1.history["val_loss"]     + h2.history["val_loss"]
    split = len(h1.epoch)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Skin Infection Detection — Training Curves\nFYP Fall-2025-107 | University of Lahore",
                 fontsize=13, fontweight="bold")

    for ax, tr, vl, title in [
        (axes[0], acc, val,    "Accuracy"),
        (axes[1], loss, vloss, "Loss"),
    ]:
        ax.plot(tr, label=f"Train {title}",      color="#2196F3", linewidth=2)
        ax.plot(vl, label=f"Validation {title}", color="#FF9800", linewidth=2)
        ax.axvline(x=split - 1, color="green", linestyle="--",
                   linewidth=1.5, label="Fine-tuning starts")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Training curves saved → {plot_path}")


# ── Main training function ─────────────────────────────────────────────────────
def train():
    print("\n" + "=" * 65)
    print("  FYP: Skin Infection Detection — Training Script")
    print("  Project ID: Fall-2025-107 | University of Lahore")
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  Dataset   : {DATASET_DIR}")
    print(f"  Model     : {MODEL_PATH}")
    print("=" * 65)

    # Check if not_skin folder exists
    not_skin_path = os.path.join(DATASET_DIR, "train", "not_skin")
    if os.path.exists(not_skin_path):
        not_skin_count = len(os.listdir(not_skin_path))
        print(f"\n  not_skin class FOUND: {not_skin_count} images")
        print("  Model will learn to REJECT non-skin images!")
    else:
        print("\n  WARNING: not_skin folder NOT found!")
        print("  Model will NOT be able to reject non-skin images.")
        print("  Add dataset/train/not_skin/ with 350+ diverse non-skin images")
        print("  Source: kaggle.com/datasets/prasunroy/natural-images")

    # Load data
    train_data, val_data, test_data, num_classes = build_generators()

    # Class weights for imbalanced training
    class_weights = get_class_weights(train_data)

    # Build model
    model, base = build_model(num_classes=num_classes)
    model.summary()

    # ── PHASE 1: Train head only (base frozen) ────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PHASE 1: Training head only (base frozen)")
    print(f"  Max epochs: {PHASE1_EPOCHS} | LR: {LR_PHASE1}")
    print(f"{'='*65}\n")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_PHASE1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(
        train_data,
        epochs=PHASE1_EPOCHS,
        validation_data=val_data,
        callbacks=make_callbacks(phase=1),
        class_weight=class_weights,
    )
    p1_best = max(h1.history["val_accuracy"])
    print(f"\n  Phase 1 best val_accuracy: {p1_best * 100:.2f}%")

    # ── PHASE 2: Fine-tune top 50 layers of MobileNetV2 ──────────────────────
    print(f"\n{'='*65}")
    print(f"  PHASE 2: Fine-tuning top 50 MobileNetV2 layers")
    print(f"  Max epochs: {PHASE2_EPOCHS} | LR: {LR_PHASE2}")
    print(f"{'='*65}\n")

    # Unfreeze top layers of base model
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"  Trainable base layers: {trainable} / {len(base.layers)}")

    # Recompile with much lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_PHASE2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(
        train_data,
        epochs=len(h1.epoch) + PHASE2_EPOCHS,
        validation_data=val_data,
        callbacks=make_callbacks(phase=2),
        initial_epoch=len(h1.epoch),
        class_weight=class_weights,
    )
    p2_best = max(h2.history["val_accuracy"])
    print(f"\n  Phase 2 best val_accuracy: {p2_best * 100:.2f}%")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FINAL EVALUATION on test set...")
    print(f"{'='*65}")
    loss, acc = model.evaluate(test_data, verbose=1)

    print(f"\n  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")

    # Save final model
    model.save(MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")

    # Save training plots
    plot_history(h1, h2)

    # Final summary
    overall = max(p1_best, p2_best)
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE!")
    print(f"  Phase 1 Best : {p1_best * 100:.2f}%")
    print(f"  Phase 2 Best : {p2_best * 100:.2f}%")
    print(f"  Test Accuracy: {acc * 100:.2f}%")
    if overall >= 0.85:
        print(f"  STATUS: 85%+ TARGET ACHIEVED!")
    else:
        print(f"  STATUS: Add more images to hit 85%+")
    print("=" * 65)


if __name__ == "__main__":
    train()
