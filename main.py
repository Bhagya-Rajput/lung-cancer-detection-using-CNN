"""Train the lung CT classifier. Run from the repository root:
    python main.py
"""
from pathlib import Path
import json

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "IQ-OTHNCCD"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "lung_cancer_model.keras"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25

# Keep this order identical in app.py. Folder spelling follows the dataset.
FOLDER_NAMES = ["Bengin cases", "Malignant cases", "Normal cases"]
DISPLAY_NAMES = ["Benign", "Malignant", "Normal"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_dataset():
    images, labels = [], []
    for class_index, folder_name in enumerate(FOLDER_NAMES):
        folder = DATASET_DIR / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"No supported images found in: {folder}")

        loaded_count = 0
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_AREA)
            images.append(image)
            labels.append(class_index)
            loaded_count += 1
        print(f"{DISPLAY_NAMES[class_index]}: loaded {loaded_count} images")

    if not images:
        raise ValueError("No images could be loaded from the dataset.")

    # The same [0, 1] scale is used by app.py for uploaded images.
    X = np.asarray(images, dtype=np.float32) / 255.0
    y = np.asarray(labels, dtype=np.int32)
    return X, y

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(len(DISPLAY_NAMES), activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    tf.keras.utils.set_random_seed(42)
    X, y = load_dataset()
    print("Dataset shape:", X.shape)
    print("Class counts:", {
        name: int(np.sum(y == index))
        for index, name in enumerate(DISPLAY_NAMES)
    })

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {int(label): float(weight)
                     for label, weight in zip(np.unique(y_train), weights)}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=2, min_lr=1e-6, verbose=1
        )
    ]

    model = build_model()
    model.summary()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True
    )

    # Load the best checkpoint, not simply the final epoch.
    best_model = keras.models.load_model(MODEL_PATH)
    probabilities = best_model.predict(X_val, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    print("\nValidation classification report:")
    print(classification_report(
        y_val, predictions, labels=list(range(len(DISPLAY_NAMES))),
        target_names=DISPLAY_NAMES, zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions))

    (MODEL_DIR / "class_names.json").write_text(
        json.dumps(DISPLAY_NAMES, indent=2), encoding="utf-8"
    )
    print(f"Best model saved to: {MODEL_PATH}")
    print("Now run: streamlit run app.py")

if __name__ == "__main__":
    main()
