import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
from rembg import remove
from io import BytesIO
import warnings

# Hide deprecation warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Rice Variety Classification",
    page_icon="🌾",
    initial_sidebar_state='auto'
)

# Hide footer & main menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Caching model loading
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Sidebar
with st.sidebar:
    st.title("RICE VARIETY CLASSIFICATION")
    st.subheader("DenseNet-201")
    st.text("Accurate Rice Variety Classifier.")

    model_options = {
        "Transfer Learning E10": r'Models/TL_model_10epoch.keras',
        "Transfer Learning E20": r'Models/TL_model_20epoch.keras',
        "Transfer Learning E30": r'Models/TL_model_30epoch.keras',
        "Non-Transfer Learning E10": r'Models/nonTL_model_10epoch.keras',
        "Non-Transfer Learning E20": r'Models/nonTL_model_20epoch.keras',
        "Non-Transfer Learning E30": r'Models/nonTL_model_30epoch.keras',
    }
    selected_model = st.selectbox("Select Classification Model", list(model_options.keys()))
    model_path = model_options[selected_model]

    try:
        with st.spinner(f'Loading {selected_model}...'):
            model = load_model(model_path)
        st.success(f"{selected_model} selected!")
    except Exception as e:
        st.error(f"{selected_model} failed to load!")

    # Jenis klasifikasi
    classification_type = st.radio("Classification Type", ("Single Grain", "Multiple Grain"))

    # Sumber gambar
    img_source = st.radio("Choose image source", ("Upload image", "Sample image"))

# Kelas dan warna label
class_names = ['ciherang', 'ir64', 'mentik']
label_colors = {
    'ciherang': (255, 0, 0),
    'ir64': (0, 0, 255),
    'mentik': (0, 255, 0),
}

# Informasi varietas
rice_info = {
    "ciherang": "Ciherang adalah varietas unggul yang banyak ditanam di Indonesia...",
    "ir64": "IR64 adalah varietas hasil pemuliaan dengan produktivitas tinggi...",
    "mentik": "Mentik adalah varietas lokal dengan aroma wangi dan tekstur sangat pulen...",
}

# Fungsi prediksi
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image) / 255.0
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Fungsi tampilkan info
def display_info(predicted_class):
    st.warning(f"{predicted_class.upper()} VARIETY")
    st.write(rice_info[predicted_class])

# Fungsi visualisasi
def visualize_predictions(predictions, class_names):
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, predictions[0], color=['blue', 'orange', 'green'])
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")
    st.pyplot(plt)

# Gambar sampel
sample_images = {
    "Ciherang": ['Images/sampel ciherang_1.png', 'Images/sampel ciherang_2.png', 'Images/sampel ciherang_3.png'],
    "IR64": ['Images/sampel ir64_1.png', 'Images/sampel ir64_2.png', 'Images/sampel ir64_3.png'],
    "Mentik": ['Images/sampel mentik_1.png', 'Images/sampel mentik_2.png', 'Images/sampel mentik_3.png']
}

# Header
st.header("🌾RICE VARIETY CLASSIFICATION")

# ------------------ SINGLE GRAIN ------------------
if classification_type == "Single Grain":
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
            try:
                image = Image.open(selected_image).convert('RGB')
                st.image(image, caption=selected_image, use_container_width=True)
                predictions = import_and_predict(image, model)
                confidence = np.max(predictions) * 100
                pred_class = class_names[np.argmax(predictions)]
                label = f"Identified variety : {pred_class.upper()}"
                st.sidebar.header("🔎RESULT")
                st.sidebar.warning(label)
                st.sidebar.info(f"Confidence score : {confidence:.2f}%")
                st.markdown("### 💡Information")
                display_info(pred_class)
            except Exception as e:
                st.error("Error processing the sample image.")
                st.error(str(e))
        else:
            st.info("Select an image for prediction")

    else:
        file = st.file_uploader("Upload an image file...", type=["jpg", "png"])
        if file is not None:
            try:
                image = Image.open(file).convert('RGB')
                st.image(image, use_container_width=True)
                predictions = import_and_predict(image, model)
                confidence = np.max(predictions) * 100
                pred_class = class_names[np.argmax(predictions)]
                label = f"Identified variety : {pred_class.upper()}"
                st.sidebar.header("🔎RESULT")
                st.sidebar.warning(label)
                st.sidebar.info(f"Confidence score : {confidence:.2f}%")
                st.markdown("### 💡Information")
                display_info(pred_class)
            except Exception as e:
                st.error("Error processing the image. Please try again.")
                st.error(str(e))
        else:
            st.text("Please upload an image file")

# ------------------ MULTIPLE GRAIN ------------------
elif classification_type == "Multiple Grain":
    file = st.file_uploader("Upload an image with multiple grains...", type=["jpg", "png"])
    if file:
        try:
            # Background removal
            input_bytes = file.read()
            output_bytes = remove(input_bytes)
            img_no_bg = Image.open(BytesIO(output_bytes)).convert("RGB")
            img_np = np.array(img_no_bg)

            # Grayscale & thresholding
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            draw_img = img_np.copy()
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
                            fontScale=0.7, color=color, thickness=2)
                count += 1

            st.image(draw_img, channels="RGB", caption=f"Total Grains Detected: {count}", use_container_width=True)
            st.success(f"Total Grains Detected: {count}")
        except Exception as e:
            st.error("Failed to process multiple grain image.")
            st.error(str(e))
