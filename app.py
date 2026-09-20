import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.models import load_model


st.set_page_config(
    page_title="FASHION MNIST  Classifier",
    page_icon=":tada:",
    layout="wide"
    )

# =================================
# Load data
# ================================= 
df_cnn = pd.read_csv("data/cnn_history.csv")
df_vgg = pd.read_csv("data/vgg16_history.csv")

# =================================
# Class names
# =================================
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# =================================
# Title
# =================================
st.title("FASHION MNIST Classifier")
st.write(
    "Завантажте зображення та оберіть модель "
    "для класифікації."
)

# ===============================
# Load models
# ===============================
@st.cache_resource
def load_cnn_model():
    cnn_model= load_model("models/cnn_model.keras")
    vgg16_model = load_model("models/vgg16_model.keras")
    return cnn_model, vgg16_model

cnn_model, vgg16_model = load_cnn_model()

# =================================
# Model selection
# =================================
selected_model = st.sidebar.selectbox(
    "Оберіть модель",
    ["CNN", "VGG16"]
)
if selected_model == 'CNN':
    history = df_cnn
    model = cnn_model
else:
    history = df_vgg
    model = vgg16_model

# =================================
# Training history
# =================================

st.header(f"Історія навчання — {selected_model}")

col1, col2 = st.columns(2)

# Accuracy
with col1:
    st.subheader("Точність")
    fig1, ax1 = plt.subplots()
    ax1.plot(history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training and Validation Accuracy over Epochs')
    ax1.legend()
    st.pyplot(fig1)

# Loss
with col2:
    st.subheader("Втрати")
    fig2, ax2 = plt.subplots()
    ax2.plot(history['loss'], label='Training Loss', color='blue')
    ax2.plot(history['val_loss'], label='Validation Loss', color='orange')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss over Epochs')
    ax2.legend()
    st.pyplot(fig2)

# =================================
# Image upload
# =================================

st.header("Завантажте зображення для класифікації")

upload_file = st.file_uploader(
    "Оберіть зображення...", 
    type=["jpg", "png", "jpeg"]
    )

if upload_file is not None:
    image = Image.open(upload_file)

    # Display the uploaded image
    st.image(image, 
             caption='Завантажене зображення', 
             width=300
             )

    # =============================
    # Preprocessing
    # ============================

    input_shape = model.input_shape

    st.write(f"Розмір вхідного шару моделі: {input_shape}")

    height, width = input_shape[1], input_shape[2]
    channels = input_shape[3]

    if channels == 1:
        # grayscale
        img = image.convert("L")
        img = img.resize((width, height))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
    else:
        # RGB
        img = image.convert("RGB")
        img = img.resize((width, height))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    # add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    st.write(f"Розмірність зображення після обробки: {img_array.shape}")

    # =============================
    # Classification
    # ============================

    if st.button("Класифікувати"):
        with st.spinner("Виконується класифікація..."):

            predictions = model.predict(
                img_array,
                verbose=0
            )[0]
        predict_index = np.argmax(predictions)
        predicted_class_name = class_names[predict_index]
        confidence = predictions[predict_index]

        # =============================
        # Result
        # ============================

        st.success(
            f"Передбачений клас: {predicted_class_name} з ймовірністю {confidence:.2f}"
        )

        st.metric(
            label="Ймовірність",
            value=f"{confidence:.2f}"
        )

        # =============================
        # All probabilities
        # ============================

        results = pd.DataFrame({
            "Клас": class_names,
            "Ймовірність": predictions
        })

        results["Ймовірність (%)"] = (results["Ймовірність"] * 100).round(2)

        results = results.sort_values(by="Ймовірність",
                                       ascending=False)
        st.subheader("Ймовірності для всіх класів")
        st.dataframe(results[["Клас", "Ймовірність (%)"]],
                    use_container_width=True,
                    hide_index=True
                     )
        st.bar_chart(results.set_index("Клас")["Ймовірність (%)"])








