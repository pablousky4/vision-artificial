import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# ------------------------------------------------------------
# Configuración de la app
# ------------------------------------------------------------
st.set_page_config(page_title="CNN Visual Cortex", layout="wide")
st.title("🧠 CNN Visual Cortex - CIFAR-10 Demo")

# Clases CIFAR-10
class_names = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

# ------------------------------------------------------------
# Cargar o definir modelo CNN
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.load_weights("cnn_cifar10.weights.h5")  # Asegúrate de tener este archivo guardado
    return model

model = load_model()

# ------------------------------------------------------------
# Sección 1: Clasificación
# ------------------------------------------------------------
st.header("🎯 Clasificación de Imágenes (Input → Output)")
uploaded_file = st.file_uploader("Sube una imagen (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen original", use_container_width=True)
    
    # Preprocesamiento
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predicción
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions)]

    st.subheader(f"🔍 Predicción: {predicted_class}")
    st.bar_chart(predictions[0])

    # Visualizar mapas de características
    st.subheader("🔬 Visualización de las capas (Conv / Pool / Dense)")

    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_batch)

    for layer_name, layer_activation in zip([l.name for l in model.layers if 'conv' in l.name or 'pool' in l.name], activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        display_grid = np.zeros((size, size * min(8, n_features)))

        for i in range(min(8, n_features)):
            x = layer_activation[0, :, :, i]
            x -= x.mean()
            x /= (x.std() + 1e-5)
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x

        st.write(f"**Capa {layer_name}**")
        st.image(display_grid, clamp=True, use_container_width=True)

# ------------------------------------------------------------
# Sección 2: Reconstrucción (Output → Imagen)
# ------------------------------------------------------------
st.header("🔁 Reconstrucción de Imagen desde el Output (Clase)")

selected_class = st.selectbox("Selecciona una clase:", class_names)
if st.button("Mostrar imagen representativa"):
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    idxs = np.where(y_test.flatten() == class_names.index(selected_class))[0]
    if len(idxs) > 0:
        idx = np.random.choice(idxs)
        img = x_test[idx]
        st.image(img, caption=f"Ejemplo de clase: {selected_class}", use_container_width=True)
    else:
        st.warning("No se encontró imagen para esta clase.")
