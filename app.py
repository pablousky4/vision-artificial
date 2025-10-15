import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.keras.datasets import cifar10

# -------------------------------
# 🔹 Configuración de la app
# -------------------------------
st.set_page_config(page_title="CNN Visualizer - CIFAR10", layout="wide")
st.title("🧠 Visualización de Red Neuronal Convolucional (CNN) - CIFAR10")

# -------------------------------
# 🔹 Crear el modelo funcional
# -------------------------------
@st.cache_resource
def load_model():
    inputs = tf.keras.Input(shape=(32, 32, 3), name="input")

    # Bloques de la CNN
    x = layers.Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    outputs = layers.Dense(10, activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_cifar10")

    # Compilación
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Cargar pesos (si existen)
    try:
        model.load_weights("cnn_cifar10.weights.h5")
        st.success("✅ Pesos cargados correctamente")
    except Exception as e:
        st.warning(f"No se pudieron cargar los pesos: {e}")

    return model


model = load_model()

# Clases CIFAR-10
class_names = [
    'avión', 'auto', 'pájaro', 'gato', 'ciervo',
    'perro', 'rana', 'caballo', 'barco', 'camión'
]

# -------------------------------
# 🔹 Subir imagen
# -------------------------------
st.header("📸 Subir imagen para clasificación")
uploaded_file = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    img_array = np.array(image) / 255.0
    input_data = np.expand_dims(img_array, axis=0)

    st.image(image, caption="🖼️ Imagen subida", width=200)

    # Predicción
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    st.subheader(f"🎯 Predicción: {class_names[predicted_class].capitalize()}")

    # -------------------------------
    # 🔹 Botón para descargar el output (.npy)
    # -------------------------------
    st.markdown("#### 💾 Descargar el vector de salida (`output`) del modelo")
    output_bytes = io.BytesIO()
    np.save(output_bytes, prediction)
    st.download_button(
        label="⬇️ Descargar output (.npy)",
        data=output_bytes.getvalue(),
        file_name="cnn_output.npy",
        mime="application/octet-stream"
    )

    # -------------------------------
    # 🔹 Visualización paso a paso
    # -------------------------------
    st.header("🔍 Proceso de la red (Conv → Pool → Flatten → Dense → Output)")

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(input_data)

    for layer_name, activation in zip([l.name for l in model.layers], activations):
        st.markdown(f"#### 🔸 Capa: `{layer_name}` — Salida: {activation.shape}")

        if activation.ndim == 4:  # Capas Conv o Pool
            num_features = activation.shape[-1]
            n = min(4, num_features)
            fig, axarr = plt.subplots(1, n, figsize=(10, 3))
            for i in range(n):
                axarr[i].imshow(activation[0, :, :, i], cmap='viridis')
                axarr[i].axis('off')
            st.pyplot(fig)
        elif activation.ndim == 2:
            st.write(activation[0])

# -------------------------------
# 🔹 Reconstrucción inversa (demo)
# -------------------------------
st.header("🔁 Reconstrucción inversa (experimental)")

st.markdown("""
Sube un archivo `.npy` con un vector de salida del modelo (10 valores),
para ver una reconstrucción simbólica.  
⚠️ Esto es **una simulación educativa**, ya que una CNN no es invertible.
""")

inverse_file = st.file_uploader("Sube archivo de salida (.npy)", type=["npy"], key="inverse")

if inverse_file is not None:
    try:
        output_array = np.load(inverse_file)
        st.write("✅ Archivo cargado correctamente")
        st.write("🔢 Vector de salida:", output_array)

        reconstructed = np.random.rand(32, 32, 3)
        st.image(reconstructed, caption="Reconstrucción simulada", width=200)
        st.info("La reconstrucción real requeriría un modelo generativo (autoencoder, GAN, etc.).")
    except Exception as e:
        st.error(f"Error al reconstruir: {e}")

# -------------------------------
# 🔹 Muestras descargables CIFAR-10
# -------------------------------
st.header("🧩 Descarga imágenes de ejemplo (CIFAR-10)")

(x_train, y_train), _ = cifar10.load_data()

# Selector de clase
selected_class = st.selectbox("Elige una clase para descargar una imagen de ejemplo:", class_names)

# Mostrar y permitir descarga
if selected_class:
    class_idx = class_names.index(selected_class)
    idx = np.where(y_train.flatten() == class_idx)[0][0]
    sample_img = x_train[idx]
    st.image(sample_img, caption=f"Ejemplo de: {selected_class}", width=200)

    # Convertir a bytes para descargar
    img_pil = Image.fromarray(sample_img)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="PNG")

    st.download_button(
        label=f"⬇️ Descargar imagen de {selected_class}",
        data=img_bytes.getvalue(),
        file_name=f"{selected_class}.png",
        mime="image/png"
    )

# -------------------------------
# 🔹 Pie
# -------------------------------
st.markdown("---")
st.caption("🧠 App desarrollada con TensorFlow + Streamlit | Pxblo")
