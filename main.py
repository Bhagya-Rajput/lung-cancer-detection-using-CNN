"""
Train the Custom CNN Lung Cancer Classifier Locally (GroupNormalization, 160x160).

Run from the repository root:
    python main.py
"""

from pathlib import Path
import json

import cv2
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "Augmented IQ-OTHNCCD lung cancer dataset"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "lung_cancer_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 55
RANDOM_SEED = 42

FOLDER_NAMES = [
    "Benign cases",
    "Malignant cases",
    "Normal cases"
]

DISPLAY_NAMES = [
    "Benign",
    "Malignant",
    "Normal"
]

NUM_CLASSES = len(DISPLAY_NAMES)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}

ORIGINAL_PREFIXES = (
    "bengin case",
    "benign case",
    "malignant case",
    "normal case"
)


# ==========================================================
# 2. LOAD DATASET (LEAK-FREE: ORIGINALS VS AUGMENTED)
# ==========================================================

def load_dataset():
    orig_images, orig_labels = [], []
    aug_images, aug_labels = [], []

    print("\n========== LOADING DATASET (LEAK-FREE) ==========\n")

    for class_index, folder_name in enumerate(FOLDER_NAMES):
        folder = DATASET_DIR / folder_name

        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        files = sorted([
            file for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not files:
            raise ValueError(f"No images found in: {folder}")

        orig_count = 0
        aug_count = 0
        skipped_count = 0

        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)

            if image is None:
                print(f"Unreadable image: {path}")
                skipped_count += 1
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_AREA
            )

            if path.name.lower().startswith(ORIGINAL_PREFIXES):
                orig_images.append(image)
                orig_labels.append(class_index)
                orig_count += 1
            else:
                aug_images.append(image)
                aug_labels.append(class_index)
                aug_count += 1

        print(
            f"{DISPLAY_NAMES[class_index]:10s} -> "
            f"Original: {orig_count:3d} | "
            f"Augmented: {aug_count:4d} | "
            f"Skipped: {skipped_count}"
        )

    if len(orig_images) == 0:
        raise ValueError("No valid original images found!")

    X_orig = np.asarray(orig_images, dtype=np.float32) / 255.0
    y_orig = np.asarray(orig_labels, dtype=np.int32)

    X_aug = np.asarray(aug_images, dtype=np.float32) / 255.0
    y_aug = np.asarray(aug_labels, dtype=np.int32)

    # Hold out 20% of original unaugmented images strictly for validation
    X_orig_train, X_val, y_orig_train, y_val = train_test_split(
        X_orig,
        y_orig,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_orig
    )

    # Combine remaining 80% originals + all pre-augmented images for training
    X_train = np.concatenate([X_orig_train, X_aug], axis=0)
    y_train = np.concatenate([y_orig_train, y_aug], axis=0)

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(X_train))
    return X_train[perm], y_train[perm], X_val, y_val


# ==========================================================
# 3. DATASET INFORMATION & VISUALIZATION
# ==========================================================

def display_dataset_info(X_train, y_train, X_val, y_val):
    print("\n========== DATASET INFORMATION ==========\n")
    print("Training shape (balanced):", X_train.shape)
    print("Validation shape (originals):", X_val.shape)

    print("\nTraining class distribution:")
    for index, name in enumerate(DISPLAY_NAMES):
        print(f"  {name}: {int(np.sum(y_train == index))}")

    print("\nValidation class distribution:")
    for index, name in enumerate(DISPLAY_NAMES):
        print(f"  {name}: {int(np.sum(y_val == index))}")

    print("\nPixel value range:", X_train.min(), "to", X_train.max())


def visualize_dataset(X, y):
    plt.figure(figsize=(12, 9))
    for class_index, class_name in enumerate(DISPLAY_NAMES):
        indices = np.where(y == class_index)[0]
        selected = np.random.choice(
            indices,
            size=min(4, len(indices)),
            replace=False
        )
        for j, image_index in enumerate(selected):
            position = class_index * 4 + j + 1
            plt.subplot(3, 4, position)
            plt.imshow(X[image_index])
            plt.title(class_name)
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# ==========================================================
# 4. DATA AUGMENTATION & CUSTOM CNN (GROUP NORMALIZATION)
# ==========================================================

def build_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02, fill_mode="reflect")
        ],
        name="light_augmentation"
    )


def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = build_augmentation()(inputs)

    # ---------------- BLOCK 1 (160 -> 80) ----------------
    x = layers.Conv2D(32, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.15)(x)

    # ---------------- BLOCK 2 (80 -> 40) ----------------
    x = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)

    # ---------------- BLOCK 3 (40 -> 20) ----------------
    x = layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=16)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # ---------------- BLOCK 4 (20 -> 10) ----------------
    x = layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=16)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    # ---------------- BLOCK 5 (10x10 Refinement) ----------------
    x = layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=16)(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    # ---------------- CLASSIFIER HEAD ----------------
    x = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(0.0005),
        use_bias=False
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.50)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="class_output"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="custom_lung_cnn_gn_160"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# 5. CALLBACKS & EVALUATION
# ==========================================================

def build_callbacks():
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]


def prediction_diagnostics(y_val, predictions, probabilities):
    print("\n========== MODEL DIAGNOSTICS ==========\n")
    print("All probabilities finite:", np.isfinite(probabilities).all())
    print("Probability range:", probabilities.min(), probabilities.max())

    print("\nActual validation distribution:")
    for index, name in enumerate(DISPLAY_NAMES):
        print(f"{name}: {np.sum(y_val == index)}")

    print("\nPredicted validation distribution:")
    for index, name in enumerate(DISPLAY_NAMES):
        print(f"{name}: {np.sum(predictions == index)}")

    print("\nMean probability per class:")
    for index, name in enumerate(DISPLAY_NAMES):
        print(f"{name}: {probabilities[:, index].mean():.4f}")

    print("\nFirst 15 predictions:\n")
    for i in range(min(15, len(probabilities))):
        actual = DISPLAY_NAMES[y_val[i]]
        predicted = DISPLAY_NAMES[predictions[i]]
        print(
            f"Actual: {actual:10s} | "
            f"Predicted: {predicted:10s} | "
            f"Probabilities: {np.round(probabilities[i], 4)}"
        )
    print("\n=======================================\n")


def evaluate_model(model, X_val, y_val):
    print("\n========== MODEL EVALUATION ==========\n")
    probabilities = model.predict(X_val, batch_size=BATCH_SIZE, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    prediction_diagnostics(y_val, predictions, probabilities)

    accuracy = accuracy_score(y_val, predictions)
    print(f"Validation accuracy: {accuracy * 100:.2f}%\n")

    print("Classification Report:\n")
    print(
        classification_report(
            y_val,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=DISPLAY_NAMES,
            zero_division=0
        )
    )

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_val, predictions, labels=list(range(NUM_CLASSES))))


def save_class_names():
    CLASS_NAMES_PATH.write_text(
        json.dumps(DISPLAY_NAMES, indent=2),
        encoding="utf-8"
    )
    print("\nClass names saved to:", CLASS_NAMES_PATH)


# ==========================================================
# 6. MAIN EXECUTION
# ==========================================================

def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_dataset()

    display_dataset_info(X_train, y_train, X_val, y_val)
    visualize_dataset(X_val, y_val)

    # Tuned weights to balance Benign precision and Normal recall
    class_weights = {0: 0.90, 1: 1.00, 2: 1.15}

    model = build_model()
    model.summary()
    callbacks = build_callbacks()

    print("\n========== STARTING TRAINING ==========\n")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    print("\nLoading best saved model...")
    best_model = keras.models.load_model(MODEL_PATH)
    evaluate_model(best_model, X_val, y_val)
    save_class_names()

    print("\n========== TRAINING COMPLETED ==========\n")
    print("Best model saved at:", MODEL_PATH)
    print("Next command: streamlit run app.py")


if __name__ == "__main__":
    main()