# vision-artificial
https://github.com/pablousky4/vision-artificial
# link-streamlit
https://vision-artificial-wrgpeu9vq9hjwznvbc6ymc.streamlit.app/

# 🧩 Visualizador de CNN - CIFAR10 con Streamlit

Una aplicación interactiva creada con **Streamlit** y **TensorFlow** que permite visualizar el proceso interno de una **Red Neuronal Convolucional (CNN)** entrenada sobre el dataset **CIFAR-10**.  
Puedes subir una imagen, observar cómo pasa por cada capa (Conv2D, Pooling, Flatten, Dense, Output), descargar el vector de salida del modelo y hasta realizar una reconstrucción inversa simbólica.

---

## 🚀 Características principales

- 🖼️ **Subida de imágenes** (JPG/PNG)
- 🔍 **Visualización paso a paso** de las activaciones de cada capa
- 🎯 **Predicción de la clase** con etiquetas de CIFAR-10
- 💾 **Descarga del output del modelo** en formato `.npy`
- 🔁 **Reconstrucción inversa simulada**
- 🧩 **Galería de ejemplos CIFAR-10** descargables desde la propia app

---

## ⚙️ Requisitos

Asegúrate de tener instaladas las siguientes dependencias:

pip install streamlit tensorflow matplotlib pillow numpy

---
🏗️ Estructura del proyecto
<pre> ```
📁 vision-artificial/
│
├── cnn_cifrar10.ipynb      #Jupyter notebook para crear el modelo
├── app.py                  # Aplicación principal de Streamlit
├── cnn_cifar10.weights.h5  # Pesos del modelo (opcional)
├── README.md               # Este archivo
└── requirements.txt        # Dependencias (opcional)
``` </pre>
---

Ejecución

Clona o descarga el repositorio:

git clone https://github.com/tuusuario/vision-artificial.git
cd vision-artificial


Instala los requisitos:

pip install -r requirements.txt


Ejecuta la aplicación:

streamlit run app.py


Abre tu navegador en http://localhost:8501

---

🧠 Dataset utilizado

El modelo utiliza CIFAR-10, un conjunto de datos de visión por computadora ampliamente usado para entrenamiento y pruebas de redes neuronales.

Clases:

Avión ✈️

Auto 🚗

Pájaro 🐦

Gato 🐱

Ciervo 🦌

Perro 🐶

Rana 🐸

Caballo 🐴

Barco 🚤

Camión 🚚

---

🎓 Uso educativo

Esta app fue diseñada para fines educativos, ideal para visualizar cómo una CNN procesa la información en tareas de clasificación de imágenes.

---

La reconstrucción inversa es una simulación, ya que las CNN no son directamente invertibles sin modelos generativos (como Autoencoders o GANs).

📁 Ejemplo visual
<img src="https://upload.wikimedia.org/wikipedia/commons/6/61/CIFAR10.png" width="400"/>

---

👨‍💻 Autor

Desarrollado por: [Pxblo]

---

🧾 Licencia

Este proyecto está bajo la licencia MIT, lo que significa que puedes modificarlo y distribuirlo libremente con atribución.