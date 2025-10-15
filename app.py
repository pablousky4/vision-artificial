import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# -------------------------------
# 🔹 Configuración de la app
# -------------------------------
st.set_page_config(page_title="CNN Visualizer - CIFAR10", layout="wide")
st.title("🧠 Visualización de Red Neuronal Convolucional (CNN) - CIFAR10")

# -------------------------------
# 🔹 Función para crear el modelo
# -------------------------------
@st.cache_resource
def load_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),  # ✅ Define Input explícitamente
        layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Flatten(name='flatten'),
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dense(10, activation='softmax', name='output')
    ])

    # Compilación
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Carga de pesos entrenados
    try:
        model.load_weights("cnn_cifar10.weights.h5")  # ✅ Nombre correcto
        st.success("✅ Pesos cargados correctamente")
    except Exception as e:
        st.warning(f"No se pudo cargar el archivo de pesos: {e}")
    
    return model


model = load_model()

# Si no está construido aún (seguridad extra)
if not model.built:
    model.build((None, 32, 32, 3))

# Lista de clases CIFAR-10
class_names = [
    'avión', 'auto', 'pájaro', 'gato', 'ciervo',
    'perro', 'rana', 'caballo', 'barco', 'camión'
]

# -------------------------------
# 🔹 Subida de imagen
# -------------------------------
st.header("📸 Subir imagen para clasificación")

uploaded_file = st.file_uploader("Sube una imagen (formato JPG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer y preprocesar imagen
    image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    img_array = np.array(image) / 255.0
    input_data = np.expand_dims(img_array, axis=0)

    # Mostrar imagen original
    st.image(image, caption="🖼️ Imagen subida", width=200)

    # Predicción
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    st.subheader(f"🎯 Predicción: {class_names[predicted_class].capitalize()}")

    # -------------------------------
    # 🔹 Visualizar paso a paso
    # -------------------------------
    st.header("🔍 Proceso de la red (Conv → Pool → Flatten → Dense → Output)")

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(input_data)

    for layer_name, activation in zip([l.name for l in model.layers], activations):
        st.markdown(f"#### 🔸 Capa: `{layer_name}` — Salida: {activation.shape}")

        # Mostrar solo las primeras 4 activaciones si es conv/pool
        if activation.ndim == 4:
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
# 🔹 Reconstrucción inversa
# -------------------------------
st.header("🔁 Reconstrucción inversa (experimental)")

st.markdown("""
Intenta subir un archivo de salida (por ejemplo, los valores del `output layer`)
para ver una reconstrucción aproximada de la imagen original.  
⚠️ Esto es **experimental** y no garantiza una reconstrucción fiel.
""")

inverse_file = st.file_uploader("Sube archivo de salida (.npy)", type=["npy"], key="inverse")

if inverse_file is not None:
    try:
        output_array = np.load(inverse_file)
        st.write("✅ Archivo cargado correctamente")

        # Reconstrucción inversa básica (placeholder)
        # ⚠️ Esto es simbólico, ya que la CNN no es invertible
        reconstructed = np.random.rand(32, 32, 3)  # Simulación
        st.image(reconstructed, caption="Reconstrucción simulada", width=200)
        st.info("La reconstrucción inversa completa requeriría un modelo generativo (p. ej., un autoencoder).")
    except Exception as e:
        st.error(f"Error al reconstruir: {e}")

# -------------------------------
# 🔹 Pie de página
# -------------------------------
st.markdown("---")
st.caption("🧠 App desarrollada con TensorFlow + Streamlit | CIFAR-10 CNN Demo")
