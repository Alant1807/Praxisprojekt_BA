import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import time  # Hinzugefügt
import cv2

# --- SEITENKONFIGURATION ---
st.set_page_config(layout="wide", page_title="Anomalie-Detektor")

# --- LADE- UND INFERENZLOGIK ---

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
    """Führt die Inferenz aus und misst die Zeit."""
    input_name = session.get_inputs()[0].name

    start_time = time.perf_counter()
    outputs = session.run(None, {input_name: image_data})
    end_time = time.perf_counter()

    inference_time = end_time - start_time
    # Gibt jetzt auch die Inferenzzeit zurück
    return outputs[0], outputs[1], inference_time

# --- UI-KOMPONENTEN ---

def create_heatmap(original_image, anomaly_map):
    """Erstellt und überlagert die Heatmap auf dem Originalbild."""
    raw_map = np.squeeze(anomaly_map)
    min_val, max_val = raw_map.min(), raw_map.max()

    normalized_map = (raw_map - min_val) / (max_val -
                                            min_val) if max_val > min_val else np.zeros_like(raw_map)
    heatmap_uint8 = (normalized_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    original_image_cv = cv2.cvtColor(
        np.array(original_image), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(
        heatmap_colored, (original_image_cv.shape[1], original_image_cv.shape[0]))

    superimposed_img = cv2.addWeighted(
        heatmap_resized, 0.5, original_image_cv, 0.5, 0)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# --- HAUPT-UI ---

st.title("🕵️ Industrielle Anomalie-Erkennung")
st.write("Ein Werkzeug zur visuellen Qualitätskontrolle. Konfigurieren Sie die Analyse in der Seitenleiste.")

# --- SEITENLEISTE FÜR EINSTELLUNGEN ---
with st.sidebar:
    st.header("⚙️ Einstellungen")

    # Schritt 1: Modell-Upload
    st.subheader("1. ONNX-Modell hochladen")
    uploaded_model = st.file_uploader(
        "ONNX-Modell auswählen", type=["onnx"], label_visibility="collapsed")

    session = None
    if uploaded_model:
        session = load_model_from_upload(uploaded_model)
        if session:
            st.success(f"Modell '{uploaded_model.name}' geladen.")

    # Schritt 2: Bild-Upload (deaktiviert, bis Modell geladen ist)
    st.subheader("2. Bilder hochladen")
    uploaded_files = st.file_uploader(
        "Bilder auswählen (Strg+A für alle)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        disabled=not session,
        label_visibility="collapsed"
    )

    # Analyse-Button
    st.divider()
    analyze_button = st.button(
        "Analyse starten", use_container_width=True, disabled=not uploaded_files)

# --- ERGEBNISANZEIGE ---

if not session:
    st.info("Bitte laden Sie in der Seitenleiste ein ONNX-Modell hoch, um zu beginnen.")
elif not uploaded_files:
    st.info("Bitte laden Sie Bilder zur Analyse hoch.")

if analyze_button and uploaded_files and session:
    results = []
    total_inference_time = 0.0  # Hinzugefügt
    with st.spinner('Analysiere Bilder...'):
        for uploaded_file in uploaded_files:
            original_image = Image.open(uploaded_file).convert("RGB")
            input_data = preprocess_image(original_image)

            # Die Inferenzzeit wird hier ebenfalls empfangen
            anomaly_map, anomaly_score, inference_time = run_inference(
                session, input_data)
            total_inference_time += inference_time  # Hinzugefügt

            heatmap_image = create_heatmap(original_image, anomaly_map)

            results.append({
                "name": uploaded_file.name,
                "image": original_image,
                "heatmap": heatmap_image,
                "score": anomaly_score[0],
                "time": inference_time  # Hinzugefügt
            })

    # --- ZUSAMMENFASSUNG ---
    st.header("📄 Analyseergebnisse")

    # Hinzugefügte Metriken für die Zeit
    avg_time = total_inference_time / len(results) if results else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Bilder insgesamt", f"{len(results)}")
    col2.metric("Gesamte Inferenzzeit", f"{total_inference_time:.2f} s")
    col3.metric("Ø Zeit pro Bild", f"{avg_time:.4f} s")

    # --- EINZELERGEBNISSE ---
    for res in results:
        # Titel des Expanders zeigt jetzt auch die Zeit an
        expander_title = f"{res['name']} (Score: {res['score']:.4f} | Zeit: {res['time']:.4f}s)"
        with st.expander(expander_title, expanded=False):
            col1, col2 = st.columns(2)
            col1.image(res["image"], caption="Originalbild",
                       use_container_width=True)
            col2.image(res["heatmap"], caption="Anomalie-Heatmap",
                       use_container_width=True)
            # Info-Box zeigt jetzt auch die Zeit an
            st.info(
                f"**Anomalie-Score:** {res['score']:.4f}\n\n**Inferenzzeit:** {res['time']:.4f} Sekunden")
