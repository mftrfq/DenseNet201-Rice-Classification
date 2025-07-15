import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Hide deprecation warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Rice Variety Classification",
    page_icon="🌾",
    initial_sidebar_state='auto'
)

# Hide Streamlit menu & footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load .tflite model with TFLite Interpreter
@st.cache_resource
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Class labels
class_names = ['ciherang', 'ir64', 'mentik']

# Info dictionary
rice_info = {
    "ciherang": "Ciherang adalah varietas unggul yang banyak ditanam di Indonesia. Beras Ciherang memiliki tekstur pulen dan hasil panen tinggi.🍚",
    "ir64": "IR64 adalah varietas produktif dengan masa panen cepat. Bijinya panjang, ramping, dan cenderung pera.🍚",
    "mentik": "Mentik dikenal dengan aroma wangi dan tekstur sangat pulen. Sering dianggap beras premium.🍚"
}

# Prediction using TFLite interpreter
def import_and_predict(image_data, interpreter):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype(np.float32) / 255.0
    input_data = np.expand_dims(img, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Tampilkan info varietas
def display_info(predicted_class):
    st.warning(f"{predicted_class.upper()} VARIETY")
    st.write(rice_info[predicted_class])

# Visualisasi probabilitas prediksi
def visualize_predictions(predictions, class_names):
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, predictions[0], color=['blue', 'orange', 'green'])
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")
    st.pyplot(plt)

# Sample images
sample_images = {
    "Ciherang": [
        r'Images/sampel ciherang_1.png',
        r'Images/sampel ciherang_2.png',
        r'Images/sampel ciherang_3.png'
    ],
    "IR64": [
        r'Images/sampel ir64_1.png',
        r'Images/sampel ir64_2.png',
        r'Images/sampel ir64_3.png'
    ],
    "Mentik": [
        r'Images/sampel mentik_1.png',
        r'Images/sampel mentik_2.png',
        r'Images/sampel mentik_3.png'
    ]
}

# Model options
model_options = {
    "Model": "Models/model_streamlit_compatible.tflite"
}

# Sidebar
with st.sidebar:
    st.title("RICE VARIETY CLASSIFICATION")
    st.subheader("DenseNet-201 Classifier")
    st.text("Aplikasi klasifikasi varietas beras berbasis citra.")
    
    selected_model = st.selectbox("Pilih Model Klasifikasi", ["-- Pilih Model --"] + list(model_options.keys()))
    img_source = st.radio("Sumber Gambar", ("Upload image", "Sample image"))

    if selected_model != "-- Pilih Model --":
        model_path = model_options[selected_model]
        try:
            with st.spinner(f'Memuat {selected_model}...'):
                model = load_model(model_path)
            st.success(f"{selected_model} berhasil dimuat.")
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            model = None
    else:
        model = None

# Header
st.header("🌾 RICE VARIETY CLASSIFICATION")
st.write(
    "Tahukah anda? biji padi yang kita kenal sebagai beras merupakan sumber karbohidrat utama bagi sebagian besar penduduk dunia. "
    "Beras tidak hanya menjadi makanan pokok yang menyediakan energi, tetapi juga memiliki peran penting dalam budaya, ekonomi, dan ketahanan pangan banyak negara, terutama di Asia."
)

# Main prediction logic
if model:
    if img_source == "Sample image":
        st.sidebar.header("Pilih Kelas Sampel")
        selected_class = st.sidebar.selectbox("Varietas Beras", list(sample_images.keys()))
        
        st.header(f"Contoh Gambar {selected_class}")
        columns = st.columns(3)
        selected_image = None
        for i, image_path in enumerate(sample_images[selected_class]):
            with columns[i % 3]:
                image = Image.open(image_path)
                st.image(image, caption=f"Sampel {i+1}", use_container_width=True)
                if st.button(f"Pilih Sampel {i+1}", key=image_path):
                    selected_image = image_path

        if selected_image:
            st.success(f"Anda memilih: {selected_image}")
            image = Image.open(selected_image).convert('RGB')
            st.image(image, caption="Gambar terpilih", use_container_width=True)

            predictions = import_and_predict(image, model)
            confidence = np.max(predictions) * 100
            pred_class = class_names[np.argmax(predictions)]

            st.sidebar.header("🔎 HASIL PREDIKSI")
            st.sidebar.warning(f"Varietas: {pred_class.upper()}")
            st.sidebar.info(f"Skor Keyakinan: {confidence:.2f}%")

            st.markdown("### 💡 Informasi Varietas")
            display_info(pred_class)
            visualize_predictions(predictions, class_names)
        else:
            st.info("Silakan pilih salah satu gambar sampel untuk klasifikasi.")

    else:
        file = st.file_uploader("Upload gambar beras (jpg/png)...", type=["jpg", "png"])
        if file:
            try:
                image = Image.open(file).convert('RGB')
                st.image(image, use_container_width=True)

                predictions = import_and_predict(image, model)
                confidence = np.max(predictions) * 100
                pred_class = class_names[np.argmax(predictions)]

                st.sidebar.header("🔎 HASIL PREDIKSI")
                st.sidebar.warning(f"Varietas: {pred_class.upper()}")
                st.sidebar.info(f"Skor Keyakinan: {confidence:.2f}%")

                st.markdown("### 💡 Informasi Varietas")
                display_info(pred_class)
                visualize_predictions(predictions, class_names)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
        else:
            st.info("Silakan unggah gambar untuk melakukan klasifikasi.")
else:
    st.warning("Silakan pilih model terlebih dahulu dari sidebar.")
