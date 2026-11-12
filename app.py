from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "lung_cancer_model.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"
IMG_SIZE = 160

st.set_page_config(page_title="Lung CT Image Classifier", page_icon="🫁")
st.title("Lung CT Image Classifier")
st.warning(
    "Educational project only. This model is not a medical device and must "
    "not be used to diagnose, rule out, or guide treatment for cancer."
)
st.write(
    "Upload a lung CT image similar to the images used in the IQ-OTHNCCD "
    "training dataset."
)

if not MODEL_PATH.is_file():
    st.error(
        "Trained model not found. First place the IQ-OTHNCCD dataset in the "
        "repository folder and run: python main.py"
    )
    st.stop()

@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_classifier()

if CLASS_NAMES_PATH.is_file():
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
else:
    # Must match FOLDER_NAMES order in main.py.
    class_names = ["Benign", "Malignant", "Normal"]

uploaded_file = st.file_uploader(
    "Choose a lung CT image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
        if image.width < 32 or image.height < 32:
            st.error("The uploaded image is too small to analyze.")
            st.stop()

        st.image(image, caption="Uploaded image", use_container_width=True)

        resized = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        probabilities = model.predict(image_array, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])

        st.subheader("Model output")
        st.caption(
            "A low-confidence result is shown as unrecognized. The threshold "
            "below is only a basic safeguard, not reliable out-of-distribution "
            "detection."
        )

        # This provisional threshold must be calibrated using a separate,
        # representative validation set containing non-lung and non-dataset images.
        CONFIDENCE_THRESHOLD = 0.60
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                "Uncertain / image not confidently recognized as one of the "
                "three trained classes. Please upload a suitable lung CT image."
            )
        else:
            st.success(f"Predicted class: {class_names[best_index]}")

        st.write(f"Top-class model score: {confidence:.1%}")
        st.caption(
            "The score is a neural-network softmax score, not the probability "
            "that a patient has cancer."
        )

        st.write("Scores for each trained class")
        for name, score in zip(class_names, probabilities):
            st.write(f"{name}: {float(score):.1%}")
            st.progress(float(score))

    except Exception as error:
        st.error(f"Could not process this file as an image: {error}")
#TO RUN THE APP: c:\Python313\python.exe -m streamlit run app.py