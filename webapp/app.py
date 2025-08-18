import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# --- Lade- und Inferenzlogik ---

# Diese Funktion lädt das Modell aus dem Upload des Nutzers
# @st.cache_resource sorgt dafür, dass das Modell im Speicher bleibt, solange die App läuft


@st.cache_resource
def load_model_from_upload(uploaded_file):
    """Lädt ein ONNX-Modell aus einer hochgeladenen Datei."""
    try:
        model_bytes = uploaded_file.getvalue()
        session = ort.InferenceSession(model_bytes)
        return session
    except Exception as e:
        st.error(f"Fehler beim Laden des ONNX-Modells: {e}")
        return None


def preprocess_image(image):
    """Bereitet ein einzelnes Bild für die Inferenz vor."""
    img_size = 256
    image = image.resize((img_size, img_size))
    input_data = np.array(image, dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def run_inference(session, image_data):
    """Führt die Inferenz aus."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_data})
    return outputs[0], outputs[1]  # anomaly_map, anomaly_score

# --- Streamlit UI ---


st.set_page_config(layout="wide", page_title="Anomalie-Detektor")
st.title("Industrielle Anomalie-Erkennung")
st.write("Dies ist ein Werkzeug zur visuellen Qualitätskontrolle. Bitte laden Sie zuerst ein ONNX-Modell und dann die zu prüfenden Bilder hoch.")

# Schritt 1: Modell-Upload
st.header("Schritt 1: ONNX-Modell hochladen")
uploaded_model = st.file_uploader("ONNX-Modell auswählen", type=["onnx"])

session = None
if uploaded_model is not None:
    session = load_model_from_upload(uploaded_model)
    if session:
        st.success(
            f"Modell '{uploaded_model.name}' erfolgreich geladen und einsatzbereit.")

st.divider()

# Schritt 2: Bild-Upload (wird erst angezeigt, wenn ein Modell geladen ist)
if session:
    st.header("Schritt 2: Bilder zur Analyse hochladen")
    uploaded_files = st.file_uploader(
        "Bilder auswählen",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Analyse starten", use_container_width=True):
            for uploaded_file in uploaded_files:
                original_image = Image.open(uploaded_file).convert("RGB")

                input_data = preprocess_image(original_image)
                anomaly_map, anomaly_score = run_inference(session, input_data)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_image, caption="Originalbild",
                             use_column_width=True)

                with col2:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(original_image)
                    heatmap = ax.imshow(np.squeeze(anomaly_map), cmap='jet', alpha=0.5)
                    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title("Anomalie-Heatmap")
                    ax.axis('off')
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    st.pyplot(fig, use_container_width=True)
                    st.caption("Anomalie-Heatmap")

                score_value = anomaly_score[0]
                st.info(f"Anomalie-Score: {score_value:.4f}")
                st.divider()
else:
    st.info("Bitte laden Sie ein ONNX-Modell hoch, um fortzufahren.")
