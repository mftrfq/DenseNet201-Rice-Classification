import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from rembg import remove
from io import BytesIO
import os

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("densenet201_rice_model.h5")

model = load_model()

# Class names and label colors
class_names = ["Ciherang", "IR64", "Mentik Wangi"]
label_colors = {
    "Ciherang": (0, 255, 0),  # Green
    "IR64": (255, 0, 0),      # Blue
    "Mentik Wangi": (255, 255, 0)  # Cyan
}

# Sample images directory
sample_images = {
    "Ciherang": ["samples/ciherang1.jpg", "samples/ciherang2.jpg", "samples/ciherang3.jpg"],
    "IR64": ["samples/ir64_1.jpg", "samples/ir64_2.jpg", "samples/ir64_3.jpg"],
    "Mentik Wangi": ["samples/mentik1.jpg", "samples/mentik2.jpg", "samples/mentik3.jpg"]
}

# Single prediction
def import_and_predict(image_data, model):
    resized_image = image_data.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    predictions = model.predict(img_array, verbose=0)
    return predictions[0]

# Multiple grain detection and classification
def detect_and_classify_grains(image_pil, model):
    ori_img = np.array(image_pil)
    draw_img = ori_img.copy()
    gray = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)

    # Remove background
    with BytesIO() as f:
        image_pil.save(f, format="PNG")
        input_bytes = f.getvalue()
    output_bytes = remove(input_bytes)
    img_no_bg = Image.open(BytesIO(output_bytes)).convert("RGB")
    img_np = np.array(img_no_bg)

    # Thresholding and Connected Component
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    count = 0
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        if area < 300:
            continue

        side = int(max(w, h) * 1.5)
        cx_int, cy_int = int(cx), int(cy)
        x1 = max(0, cx_int - side // 2)
        y1 = max(0, cy_int - side // 2)
        side = min(side, min(img_np.shape[1] - x1, img_np.shape[0] - y1))
        crop = img_np[y1:y1 + side, x1:x1 + side]

        resized = cv2.resize(crop, (224, 224))
        x_input = tf.expand_dims(resized / 255.0, axis=0)

        pred = model.predict(x_input, verbose=0)
        score = tf.nn.softmax(pred[0])
        label = class_names[np.argmax(score)]
        color = label_colors.get(label, (0, 255, 255))

        cv2.rectangle(draw_img, (x1, y1), (x1 + side, y1 + side), color, 2)
        cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=color, thickness=2)
        count += 1

    return draw_img, count

# Show description
@st.cache_data
def display_info(label):
    if label == "Ciherang":
        st.markdown("Ciherang is a popular rice variety... (custom info here)")
    elif label == "IR64":
        st.markdown("IR64 is known for its high yield... (custom info here)")
    elif label == "Mentik Wangi":
        st.markdown("Mentik Wangi is an aromatic rice... (custom info here)")

# UI
st.title("Rice Variety Classification using DenseNet201")

img_source = st.sidebar.radio("Select Image Source", ("Sample image", "Upload image"))
prediction_type = st.sidebar.radio("Prediction Type", ("Single Grain", "Multiple Grain"))

if img_source == "Sample image":
    st.sidebar.header("Select a class")
    selected_class = st.sidebar.selectbox("Rice Variety", list(sample_images.keys()))

    st.header(f"Sample of {selected_class} images")
    columns = st.columns(3)
    selected_image = None
    for i, image_path in enumerate(sample_images[selected_class]):
        with columns[i % 3]:
            image = Image.open(image_path)
            st.image(image, caption=f"Sample {i + 1}", use_container_width=True)
            if st.button(f"Select Sample {i + 1}", key=image_path):
                selected_image = image_path

    if selected_image:
        st.success(f"You selected: {selected_image}")
        image = Image.open(selected_image).convert('RGB')
        st.image(image, caption=selected_image, use_container_width=True)

        if prediction_type == "Single Grain":
            predictions = import_and_predict(image, model)
            confidence = np.max(predictions) * 100
            pred_class = class_names[np.argmax(predictions)]
            label = f"Identified variety : {pred_class.upper()}"

            st.sidebar.header("🔎RESULT")
            st.sidebar.warning(label)
            st.sidebar.info(f"Confidence score : {confidence:.2f}%")

            st.markdown("### 💡Information")
            display_info(pred_class)

        elif prediction_type == "Multiple Grain":
            draw_img, count = detect_and_classify_grains(image, model)
            st.image(draw_img, caption=f"Total Detected: {count}", use_container_width=True)

else:
    file = st.file_uploader("Upload an image file...", type=["jpg", "png"])
    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, use_container_width=True)

        if prediction_type == "Single Grain":
            predictions = import_and_predict(image, model)
            confidence = np.max(predictions) * 100
            pred_class = class_names[np.argmax(predictions)]
            label = f"Identified variety : {pred_class.upper()}"

            st.sidebar.header("🔎RESULT")
            st.sidebar.warning(label)
            st.sidebar.info(f"Confidence score : {confidence:.2f}%")

            st.markdown("### 💡Information")
            display_info(pred_class)

        elif prediction_type == "Multiple Grain":
            draw_img, count = detect_and_classify_grains(image, model)
            st.image(draw_img, caption=f"Total Detected: {count}", use_container_width=True)
    else:
        st.info("Please upload an image to begin.")
