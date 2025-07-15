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

# Sidebar content
with st.sidebar:
    st.title("RICE VARIETY CLASSIFICATION")
    st.subheader("DenseNet-201")
    st.text("Accurate Rice Variety Classifier. It helps users to easily classify rice based on images.")

    # Dropdown model
    model_options = {
    "Transfer Learning E10": "Models/TL_model_10epoch.keras",
    "Transfer Learning E20": "Models/TL_model_20epoch.keras",
    "Transfer Learning E30": "Models/TL_model_30epoch.keras",
    "Non-Transfer Learning E10": "Models/nonTL_model_10epoch.keras",
    "Non-Transfer Learning E20": "Models/nonTL_model_20epoch.keras",
    "Non-Transfer Learning E30": "Models/nonTL_model_30epoch.keras",
    }

    selected_model = st.selectbox("Select Classification Model", list(model_options.keys()))
    model_path = model_options[selected_model]
    st.write("Checking model file at:", model_path)
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
    else:
        st.success("Model file found.")
        
    # try:
    #     with st.spinner(f'Loading {selected_model}...'):
    #         model = load_model(model_path)
    #     st.success(f"{selected_model} selected!")
    # except Exception as e:
    #     st.error(f"{selected_model} failed to load!")

    # Image suorce
    img_source = st.radio("Choose image source", ("Upload image", "Sample image"))

# Headline
st.header("🌾RICE VARIETY CLASSIFICATION")
st.write("Tahukah anda? biji padi yang kita kenal sebagai beras merupakan sumber karbohidrat utama bagi sebagian besar penduduk dunia. " \
"Beras tidak hanya menjadi makanan pokok yang menyediakan energi, tetapi juga memiliki peran penting dalam budaya, " \
"ekonomi, dan ketahanan pangan banyak negara, terutama di Asia.")

# Class name
class_names = ['ciherang', 'ir64', 'mentik']

# Varietu info
rice_info = {
    "ciherang": "Ciherang adalah varietas unggul yang banyak ditanam di Indonesia. "
                "Varietas ini dikenal karena hasil panennya yang tinggi dan daya adaptasinya yang baik "
                "terhadap berbagai kondisi lingkungan. Beras Ciherang memiliki tekstur pulen yang disukai "
                "banyak masyarakat, serta aroma yang tidak terlalu kuat, menjadikannya pilihan populer untuk konsumsi sehari-hari.🍚",
    "ir64": "IR64 adalah varietas hasil pemuliaan yang memiliki produktivitas tinggi "
            "dan masa panen yang relatif singkat. Varietas ini terkenal dengan biji-bijinya yang panjang dan ramping, "
            "serta teksturnya yang cenderung lebih pera (tidak terlalu lengket) setelah dimasak.🍚",
    "mentik": "Mentik adalah varietas lokal yang memiliki ciri khas aroma wangi dan tekstur yang sangat pulen. "
              "Beras Mentik sering dianggap sebagai beras premium karena kualitasnya yang tinggi dan rasa khasnya yang unik. "
              "Varietas ini umumnya ditanam di daerah tertentu dengan iklim yang sesuai, dan sering digunakan dalam hidangan "
              "tradisional atau acara khusus.🍚"
}

# Pred funct
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image) / 255.0 
    img_reshape = img[np.newaxis, ...] 
    prediction = model.predict(img_reshape)
    return prediction

# Show info funct
def display_info(predicted_class):
    st.warning(f"{predicted_class.upper()} VARIETY")
    st.write(rice_info[predicted_class])

# Vis result
def visualize_predictions(predictions, class_names):
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, predictions[0], color=['blue', 'orange', 'green'])
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")
    st.pyplot(plt)

# Image sample
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

# Process condition 
if img_source == "Sample image":
    st.sidebar.header("Select a class")
    selected_class = st.sidebar.selectbox("Rice Variety", list(sample_images.keys()))

    # Preview
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
    if file is None:
        st.text("Please upload an image file")
    else:
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
            st.error("Error processing the image. Please try again with a valid image file.")
            st.error(str(e))
