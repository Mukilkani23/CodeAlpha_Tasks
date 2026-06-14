import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import string
import cv2

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/emnist_letters_cnn.h5")

model = load_model()
labels_map = list(string.ascii_uppercase)

st.title("Handwritten Character Recognition")
st.markdown("Draw a letter (A–Z) in the box below")
st.markdown("Draw only Capital letter and click **Predict**.")

# Session state for canvas key (incrementing it forces a fresh canvas)
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# Clear button
if st.button("🔄 Clear & Draw Again"):
    st.session_state.canvas_key += 1

# Drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}"
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
        img = img.convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0

        # EMNIST rotation fix
        img_array = np.fliplr(np.rot90(img_array, k=3))
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        predictions = model.predict(img_array)[0]
        top3_idx = predictions.argsort()[-3:][::-1]

        st.subheader("Prediction Results")
        for i, idx in enumerate(top3_idx):
            label = labels_map[idx]
            confidence = predictions[idx] * 100
            if i == 0:
                st.success(f"**{label}** — {confidence:.1f}% confidence")
            else:
                st.info(f"{label} — {confidence:.1f}%")

        st.image(img.resize((140, 140)), caption="What the model saw", clamp=True)
    else:
        st.warning("Draw something first!")

st.markdown("---")
st.caption("EMNIST Letters CNN | CodeAlpha ML Internship | Mukilkani23")