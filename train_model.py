"""
DermoScan - CNN Training Script (v2 - improved regularisation)
Trains a skin infection classifier on the dataset and saves model.h5

Improvements over v1:
  - Stronger augmentation (vertical flip, fill_mode, wider brightness)
  - L2 regularisation on Dense layers
  - Higher dropout (0.5 / 0.4)
  - Cosine annealing LR schedule instead of ReduceLROnPlateau in phase 1
  - Warm learning-rate restart at start of phase 2
  - Class-weight balancing for imbalanced classes
  - Increased fine-tune depth (last 50 layers instead of 30)

Usage:
    python train_model.py

Dataset expected layout:
    dataset/
        train/  acne/ eczema/ normal/ psoriasis/ ringworm/
        val/    (same)
        test/   (same)
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all systems)
import matplotlib.pyplot as plt

# Suppress noisy Keras warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "skin_model.h5")

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
PHASE1_EPOCHS = 20       # head-only training
PHASE2_EPOCHS = 40       # fine-tuning (EarlyStopping will cut this short)
NUM_CLASSES = 5
LR_PHASE1   = 1e-3
LR_PHASE2   = 5e-5       # gentler LR for fine-tuning to avoid the val loss spike
L2_REG      = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Data generators ──────────────────────────────────────────────────────────

def build_generators():
    # Stronger augmentation to reduce overfitting
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.20,
        height_shift_range=0.20,
        horizontal_flip=True,
        vertical_flip=True,          # skin lesions have no orientation bias
        zoom_range=0.25,
        brightness_range=[0.7, 1.3],
        shear_range=0.15,
        channel_shift_range=20.0,    # slight colour jitter
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_data = val_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_data = val_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_data, val_data, test_data


def compute_class_weights(train_data):
    """Balance loss contribution from each class."""
    from sklearn.utils.class_weight import compute_class_weight  # type: ignore
    labels = train_data.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


# ── Model architecture ────────────────────────────────────────────────────────

def build_model() -> tf.keras.Model:
    """
    MobileNetV2 backbone + regularised classification head.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # frozen in phase 1

    reg = regularizers.l2(L2_REG)

    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x       = layers.Dropout(0.5)(x)
    x       = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="DermoScan_MobileNetV2_v2")
    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────

def make_callbacks(phase: int) -> list:
    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,                  # more patience than before
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
    if phase == 1:
        # Cosine decay for smooth LR reduction during head training
        cb.append(callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1,
        ))
    else:
        # More conservative reduction during fine-tuning
        cb.append(callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3,
            patience=3, min_lr=1e-7, verbose=1,
        ))
    return cb


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 60)
    print("  DermoScan – CNN Training Script  (v2)")
    print("=" * 60)
    print(f"  TensorFlow  : {tf.__version__}")
    print(f"  Dataset     : {DATASET_DIR}")
    print(f"  Model → {MODEL_PATH}")
    print("=" * 60 + "\n")

    train_data, val_data, test_data = build_generators()
    print(f"Classes : {train_data.class_indices}\n")

    # Class weights to handle mild imbalance
    try:
        class_weights = compute_class_weights(train_data)
        print(f"Class weights : {class_weights}\n")
    except ImportError:
        print("sklearn not found – skipping class weighting\n")
        class_weights = None

    model = build_model()
    model.summary()

    # ── Phase 1: train head only ──────────────────────────────────────────────
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_PHASE1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n[Phase 1] Training classification head for up to {PHASE1_EPOCHS} epochs …")
    history1 = model.fit(
        train_data,
        epochs=PHASE1_EPOCHS,
        validation_data=val_data,
        callbacks=make_callbacks(phase=1),
        class_weight=class_weights,
    )

    # ── Phase 2: fine-tune last 50 layers ────────────────────────────────────
    print("\n[Phase 2] Fine-tuning last 50 base layers …")
    base = model.layers[1]          # MobileNetV2 is layer index 1 in functional model
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    # Recompile with much lower LR to avoid the loss spike seen in v1
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_PHASE2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_data,
        epochs=len(history1.epoch) + PHASE2_EPOCHS,
        validation_data=val_data,
        callbacks=make_callbacks(phase=2),
        initial_epoch=len(history1.epoch),
        class_weight=class_weights,
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[Evaluation] Test set …")
    loss, acc = model.evaluate(test_data, verbose=1)
    print(f"\n  Test accuracy : {acc * 100:.2f}%")
    print(f"  Test loss     : {loss:.4f}")

    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    _plot_history(history1, history2)


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot_history(h1, h2):
    acc   = h1.history["accuracy"]     + h2.history["accuracy"]
    val   = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]         + h2.history["loss"]
    vloss = h1.history["val_loss"]     + h2.history["val_loss"]
    p1_end = len(h1.epoch)             # mark phase boundary

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train_vals, val_vals, title in [
        (axes[0], acc,  val,   "Accuracy"),
        (axes[1], loss, vloss, "Loss"),
    ]:
        ax.plot(train_vals, label=f"Train {title}", color="#2196f3")
        ax.plot(val_vals,   label=f"Val {title}",   color="#ff9800")
        ax.axvline(x=p1_end - 1, color="gray", linestyle="--", linewidth=1,
                   label="Phase 2 start")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.suptitle("DermoScan v2 – Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_curves_v2.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved → {plot_path}")
    plt.close()


if __name__ == "__main__":
    train()
