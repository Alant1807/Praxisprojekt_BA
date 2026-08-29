import streamlit as st
import onnxruntime as ort
import numpy as np
import time
import cv2
import json
import os
import psutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any, Optional

st.set_page_config(
    layout="wide", page_title="Anomalie-Detektor", page_icon=""
)


def create_optimized_session_options() -> ort.SessionOptions:
    """Konfiguriert die ONNX Runtime für maximale Geschwindigkeit.

    Returns:
        ort.SessionOptions: Die optimierten Einstellungen.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    cpu_count = os.cpu_count() or 4
    so.intra_op_num_threads = cpu_count
    so.inter_op_num_threads = 1

    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    so.enable_mem_reuse = True

    try:
        so.add_session_config_entry("session.use_env_allocators", "1")
        so.add_session_config_entry("session.intra_op.allow_spinning", "1")
    except Exception:
        pass

    return so


def get_optimized_providers() -> list:
    """Sucht nach verfügbaren Hardware-Beschleunigern (z.B. CUDA/GPU).

    Returns:
        list: Liste der verfügbaren ONNX Execution Provider.
    """
    available_providers = ort.get_available_providers()
    providers = []

    if "TensorrtExecutionProvider" in available_providers:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_cache",
                },
            )
        )

    if "CUDAExecutionProvider" in available_providers:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                    "cudnn_conv_use_max_workspace": "1",
                },
            )
        )

    if "OpenVINOExecutionProvider" in available_providers:
        providers.append(
            (
                "OpenVINOExecutionProvider",
                {
                    "device_type": "CPU_FP32",
                    "enable_vpu_fast_compile": False,
                    "num_of_threads": os.cpu_count(),
                },
            )
        )

    if "DmlExecutionProvider" in available_providers:
        providers.append(
            (
                "DmlExecutionProvider",
                {
                    "device_id": 0,
                },
            )
        )

    providers.append(
        (
            "CPUExecutionProvider",
            {
                "arena_extend_strategy": "kSameAsRequested",
            },
        )
    )

    return providers


def validate_model_compatibility(session: ort.InferenceSession) -> Dict[str, Any]:
    """Prüft, ob das ONNX-Modell die erwarteten Ein- und Ausgänge hat.

    Args:
        session (ort.InferenceSession): Die geladene ONNX-Sitzung.

    Returns:
        dict: Informationen zur Kompatibilität und eventuelle Warnungen.
    """
    info = {
        "compatible": True,
        "warnings": [],
        "input_name": None,
        "input_shape": None,
        "img_size": None,
        "output_names": [],
    }

    inputs = session.get_inputs()
    if len(inputs) != 1:
        info["warnings"].append(f"Erwartet 1 Input, gefunden: {len(inputs)}")

    input_info = inputs[0]
    info["input_name"] = input_info.name
    info["input_shape"] = input_info.shape

    expected_input_name = "input_image"
    if input_info.name != expected_input_name:
        info["warnings"].append(
            f"Input-Name '{input_info.name}' (erwartet: '{expected_input_name}')"
        )

    shape = input_info.shape
    if len(shape) == 4:
        height = shape[1] if isinstance(shape[1], int) else 256
        width = shape[2] if isinstance(shape[2], int) else 256
        channels = shape[3] if isinstance(shape[3], int) else 3

        if height != width:
            info["warnings"].append(f"Nicht-quadratisches Input: {height}x{width}")

        if channels != 3:
            info["warnings"].append(f"Erwartet 3 Kanäle, gefunden: {channels}")
            info["compatible"] = False

        info["img_size"] = height
    else:
        info["warnings"].append(f"Unerwartete Input-Shape: {shape}")
        info["compatible"] = False
        info["img_size"] = 256

    if input_info.type != "tensor(uint8)":
        info["warnings"].append(
            f"Input-Typ '{input_info.type}' (erwartet: 'tensor(uint8)')"
        )

    outputs = session.get_outputs()
    info["output_names"] = [o.name for o in outputs]

    expected_outputs = ["anomaly_map", "anomaly_score"]
    for expected in expected_outputs:
        if expected not in info["output_names"]:
            info["warnings"].append(f"Output '{expected}' nicht gefunden")

    if len(outputs) < 2:
        info["compatible"] = False
        info["warnings"].append(
            f"Erwartet mindestens 2 Outputs, gefunden: {len(outputs)}"
        )

    return info


@st.cache_resource
def load_model_from_upload(uploaded_file):
    """Lädt das ONNX-Modell und führt einen kurzen Warmup-Lauf durch.

    Args:
        uploaded_file: Die in Streamlit hochgeladene Modelldatei.

    Returns:
        tuple: (Die geladene InferenceSession, Bildgröße) oder (None, None) bei Fehler.
    """
    try:
        # Alte Sitzungsergebnisse zurücksetzen
        for key in [
            "latency_results",
            "analysis_results",
            "analysis_metrics",
            "analysis_threshold",
        ]:
            if key in st.session_state:
                del st.session_state[key]
                
        model_bytes = uploaded_file.getvalue()

        so = create_optimized_session_options()
        providers = get_optimized_providers()

        load_start = time.perf_counter()
        session = ort.InferenceSession(
            model_bytes, sess_options=so, providers=providers
        )
        load_time = time.perf_counter() - load_start

        model_info = validate_model_compatibility(session)

        if not model_info["compatible"]:
            st.error("Modell ist nicht kompatibel!")
            for warning in model_info["warnings"]:
                st.warning(f"{warning}")
            return None, None

        for warning in model_info["warnings"]:
            st.warning(f"{warning}")

        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / (1024 * 1024)

        img_size = model_info["img_size"]
        input_name = model_info["input_name"]

        actual_provider = session.get_providers()[0]
        model_size_mb = len(model_bytes) / (1024 * 1024)

        # Warmup mit Dummy-Daten durchführen, um Laufzeit-Allokationen vorzubereiten
        dummy_input = np.random.randint(
            0, 256, (1, img_size, img_size, 3), dtype=np.uint8
        )
        for _ in range(5):
            _ = session.run(None, {input_name: dummy_input})

        st.session_state.load_time = load_time
        st.session_state.mem_usage = mem_usage
        st.session_state.provider = actual_provider
        st.session_state.input_name = input_name
        st.session_state.output_names = model_info["output_names"]
        st.session_state.model_size_mb = model_size_mb

        return session, img_size

    except Exception as e:
        st.error(f"Fehler beim Laden des ONNX-Modells: {e}")
        return None, None


def measure_realtime_latency(
    session: ort.InferenceSession,
    img_size: int,
    num_runs: int = 50,
    warmup_runs: int = 10,
    input_name: str = None,
) -> Dict[str, float]:
    """Misst die durchschnittliche Inferenzzeit des Modells.

    Args:
        session (ort.InferenceSession): Das ONNX-Modell.
        img_size (int): Bildgröße.
        num_runs (int, optional): Anzahl der gemessenen Durchläufe.
        warmup_runs (int, optional): Anzahl der ungemessenen Aufwärm-Durchläufe.
        input_name (str, optional): Name des Eingabe-Knotens.

    Returns:
        dict: Ergebnisse der Latenzmessung.
    """
    if input_name is None:
        input_name = session.get_inputs()[0].name

    dummy_input = np.random.randint(
        0, 256, (1, img_size, img_size, 3), dtype=np.uint8
    )

    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    return {
        "median_ms": float(np.median(latencies)),
        "std_ms": float(latencies.std()),
        "num_runs": num_runs,
    }


def load_and_preprocess_single(file_obj, img_size: int) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Lädt ein Bild aus dem Arbeitsspeicher und ändert die Größe via OpenCV.

    Args:
        file_obj: Das hochgeladene Bildobjekt aus Streamlit.
        img_size (int): Die Zielauflösung.

    Returns:
        tuple: (Dateiobjekt, Originalbild, verkleinertes Bild).
    """
    try:
        bytes_data = np.frombuffer(file_obj.getvalue(), np.uint8)
        
        img_bgr = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Bild konnte nicht dekodiert werden.")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        return file_obj, img_rgb, resized
    except Exception as e:
        return file_obj, None, None


def run_inference_batch_optimized(
    session: ort.InferenceSession,
    images_data: np.ndarray,
    input_name: str = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Führt das ONNX-Modell für einen Batch von Bildern aus.

    Args:
        session (ort.InferenceSession): Das ONNX-Modell.
        images_data (np.ndarray): Die vorverarbeiteten Bilder.
        input_name (str, optional): Name des Eingabe-Knotens.

    Returns:
        tuple: (Anomaly Maps, Anomaly Scores, Inferenzzeit in Sekunden).
    """
    if input_name is None:
        input_name = session.get_inputs()[0].name

    start_time = time.perf_counter()
    outputs = session.run(None, {input_name: images_data})
    inference_time = time.perf_counter() - start_time

    anomaly_map = outputs[0]
    anomaly_score = outputs[1]

    if anomaly_score.ndim > 1:
        anomaly_score = anomaly_score.squeeze(-1)

    return anomaly_map, anomaly_score, inference_time


def create_heatmap_optimized(
    original_image: np.ndarray, anomaly_map: np.ndarray
) -> np.ndarray:
    """Legt die berechnete Anomalie-Heatmap über das Originalbild.

    Args:
        original_image (np.ndarray): Das ursprüngliche Bild.
        anomaly_map (np.ndarray): Die vom Modell berechnete Anomalie-Map.

    Returns:
        np.ndarray: Das fusionierte Bild (Heatmap + Original).
    """
    raw_map = np.squeeze(anomaly_map)
    while raw_map.ndim > 2:
        raw_map = np.squeeze(raw_map, axis=0)

    raw_map = cv2.GaussianBlur(raw_map, (5, 5), sigmaX=1.0, sigmaY=1.0)

    min_val, max_val = raw_map.min(), raw_map.max()
    range_val = max_val - min_val

    if range_val > 1e-8:
        normalized_map = (raw_map - min_val) / range_val
    else:
        normalized_map = np.zeros_like(raw_map)

    heatmap_uint8 = (normalized_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # RGB-Format für korrekte Farben sicherstellen
    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    target_size = (original_bgr.shape[1], original_bgr.shape[0])
    heatmap_resized = cv2.resize(
        heatmap_colored, target_size, interpolation=cv2.INTER_LINEAR
    )

    superimposed = cv2.addWeighted(heatmap_resized, 0.5, original_bgr, 0.5, 0)

    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)


def process_single_image_optimized(args: Tuple) -> Dict[str, Any]:
    """Bereitet die Analyseergebnisse für ein einzelnes Bild auf.

    Args:
        args (tuple): Alle Daten zum Bild (File, Original, Map, Score, Zeit).

    Returns:
        dict: Fertiges Ergebnis-Dictionary für die GUI-Anzeige.
    """
    (
        uploaded_file,
        original_image,
        anomaly_map,
        anomaly_score,
        inference_time,
    ) = args
    heatmap_image = create_heatmap_optimized(original_image, anomaly_map)

    return {
        "name": uploaded_file.name,
        "image": original_image,
        "heatmap": heatmap_image,
        "score": float(anomaly_score),
        "time": inference_time,
    }


def process_images_parallel_optimized(
    uploaded_files: List,
    session: ort.InferenceSession,
    img_size: int,
    batch_size: int = 8,
) -> Tuple[List, float, float, float]:
    """Verarbeitet mehrere Bilder parallel (Laden & Heatmap) und führt Inference batchweise aus.

    Args:
        uploaded_files (list): Liste der hochgeladenen Bilder.
        session (ort.InferenceSession): Das ONNX-Modell.
        img_size (int): Bildauflösung.
        batch_size (int, optional): Anzahl Bilder pro Inferenzschritt.

    Returns:
        tuple: (Ergebnis-Liste, Zeit Laden, Zeit Inferenz, Zeit Postprocessing).
    """
    results = []
    total_inference_time = 0.0
    total_pre_time = 0.0
    total_post_time = 0.0

    input_name = st.session_state.get("input_name", session.get_inputs()[0].name)
    num_batches = (len(uploaded_files) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(uploaded_files))
        batch_files = uploaded_files[start_idx:end_idx]

        if not batch_files:
            continue

        pre_start = time.perf_counter()
        
        valid_files = []
        original_images = []
        input_batch_list = []
        
        max_workers = min(8, len(batch_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            load_results = executor.map(lambda f: load_and_preprocess_single(f, img_size), batch_files)
            
            for f, img_rgb, img_resized in load_results:
                if img_rgb is not None:
                    valid_files.append(f)
                    original_images.append(img_rgb)
                    input_batch_list.append(img_resized)
                else:
                    st.warning(f"Bild '{f.name}' konnte nicht geladen werden.")

        if not valid_files:
            continue
            
        input_batch = np.stack(input_batch_list, axis=0)
        total_pre_time += time.perf_counter() - pre_start

        anomaly_maps, anomaly_scores, inference_time = run_inference_batch_optimized(
            session, input_batch, input_name
        )
        total_inference_time += inference_time

        post_start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            args_list = [
                (
                    valid_files[i],
                    original_images[i],
                    anomaly_maps[i : i + 1],
                    anomaly_scores[i],
                    inference_time / len(valid_files),
                )
                for i in range(len(valid_files))
            ]
            batch_results = list(executor.map(process_single_image_optimized, args_list))

        total_post_time += time.perf_counter() - post_start
        results.extend(batch_results)

    return results, total_pre_time, total_inference_time, total_post_time


def load_config_from_json(uploaded_json) -> Optional[float]:
    """Liest den optimalen Schwellenwert aus der hochgeladenen JSON-Datei.

    Args:
        uploaded_json: Das hochgeladene JSON-Dateiobjekt.

    Returns:
        float: Der extrahierte Schwellenwert oder None bei Fehler.
    """
    try:
        uploaded_json.seek(0)
        summary_data = json.load(uploaded_json)

        perf = summary_data.get("performance_metrics", {})
        threshold = perf.get("quantile_threshold") or perf.get(
            "optimal_threshold_youden_j"
        )

        if threshold is None:
            st.warning(
                "Optimaler Schwellenwert nicht gefunden.\nStandardwert 0.5 wird verwendet."
            )
            return 0.5
        return threshold
    except Exception as e:
        st.error(f"Fehler beim Lesen der JSON-Konfiguration: {e}")
        return None


# =============================================================================
# Streamlit Benutzeroberfläche
# =============================================================================

st.title("Industrielle Anomalie-Erkennung")

with st.sidebar:
    st.header("Einstellungen")

    st.subheader("ONNX-Modell hochladen")
    uploaded_model = st.file_uploader(
        "Wählen Sie ein Modell (.onnx oder .ort) aus.",
        type=["onnx", "ort"],
        label_visibility="collapsed",
    )

    session, img_size = (None, None)
    if uploaded_model:
        session, img_size = load_model_from_upload(uploaded_model)
        if session:
            st.success(f"Modell '{uploaded_model.name}' geladen")
            st.caption(f"Input-Größe: {img_size}x{img_size}")

            col1, col2 = st.columns(2)
            if "load_time" in st.session_state:
                col1.metric("Ladezeit", f"{st.session_state.load_time:.2f}s")

            if "mem_usage" in st.session_state:
                col2.metric("RAM", f"{st.session_state.mem_usage:.0f}MB")

            if "provider" in st.session_state:
                provider_name = st.session_state.provider.replace(
                    "ExecutionProvider", ""
                )
                provider_emoji = (
                    ""
                    if "CUDA" in st.session_state.provider
                    or "Tensorrt" in st.session_state.provider
                    else ""
                )
                st.metric(f"{provider_emoji} Provider".strip(), provider_name)

            if "model_size_mb" in st.session_state:
                st.caption(
                    f"Modellgröße: {st.session_state.model_size_mb:.1f} MB"
                )

            if "output_names" in st.session_state:
                st.caption(
                    f"Outputs: {', '.join(st.session_state.output_names)}"
                )

            if st.button("Echtzeit-Latenz messen", use_container_width=True):
                with st.spinner(
                    "Messe Latenz (50 Durchläufe, Batch-Size 1)..."
                ):
                    input_name = st.session_state.get(
                        "input_name", session.get_inputs()[0].name
                    )
                    latency = measure_realtime_latency(
                        session,
                        img_size,
                        num_runs=50,
                        warmup_runs=10,
                        input_name=input_name,
                    )
                    st.session_state.latency_results = latency

            if "latency_results" in st.session_state:
                lat = st.session_state.latency_results
                st.subheader("Echtzeit-Latenz (Batch-Size 1)")
                st.metric("Median", f"{lat['median_ms']:.1f} ms")

                st.caption(
                    f"Gemessen über {lat['num_runs']} Durchläufe (10 Warm-Up)"
                )

                latency_text = (
                    f"Modell: {uploaded_model.name}\n"
                    f"Modellgröße: {st.session_state.get('model_size_mb', 0):.1f} MB\n"
                    f"Provider: {st.session_state.get('provider', 'N/A')}\n"
                    f"Input-Größe: {img_size}x{img_size}\n"
                    f"\nLatenz-Ergebnisse (Batch-Size 1):\n"
                    f"  Median:     {lat['median_ms']:.2f} ms\n"
                    f"  Durchläufe: {lat['num_runs']}\n"
                )
                st.download_button(
                    "Latenz exportieren",
                    latency_text,
                    "latenz_benchmark.txt",
                    use_container_width=True,
                )

    st.divider()

    st.subheader("Konfigurationsdatei")
    uploaded_summary = st.file_uploader(
        "Laden Sie die 'inference_summary.json' Datei hoch.",
        type="json",
        disabled=not session,
        label_visibility="collapsed",
    )

    OPTIMAL_THRESHOLD = None
    if uploaded_summary:
        OPTIMAL_THRESHOLD = load_config_from_json(uploaded_summary)
        if OPTIMAL_THRESHOLD is not None:
            st.success(f"Schwellenwert: **{OPTIMAL_THRESHOLD:.4f}**")

            OPTIMAL_THRESHOLD = st.slider(
                "Schwellenwert anpassen",
                min_value=0.0,
                max_value=max(1.0, float(OPTIMAL_THRESHOLD) * 5),
                value=float(OPTIMAL_THRESHOLD),
                step=0.0001,
                format="%.4f",
                help="Höher = weniger Anomalien erkannt.",
            )

    st.divider()

    st.subheader("Bilder hochladen")
    uploaded_files = st.file_uploader(
        "Wählen Sie Bilder zur Analyse aus.",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        disabled=not session or OPTIMAL_THRESHOLD is None,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} Bild(er) hochgeladen")

    st.divider()

    with st.expander("Performance-Einstellungen", expanded=False):
        batch_size = st.slider(
            "Batch-Größe",
            min_value=1,
            max_value=32,
            value=8,
            step=1,
            help="Größere Batches = schneller, aber mehr RAM",
        )

        st.caption("Empfehlung: Batch-Größe 8-16 für optimale Performance")

        st.markdown("---")
        st.markdown("**Verfügbare Provider:**")
        for provider in ort.get_available_providers():
            emoji = (
                "Gültig"
                if provider
                in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
                else "nicht gültig"
            )
            st.caption(f"{emoji} {provider.replace('ExecutionProvider', '')}")

    analyze_button = st.button(
        "Analyse starten",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )


if not session:
    st.info(
        "Bitte laden Sie in der Seitenleiste ein ONNX-Modell hoch, um zu beginnen."
    )
elif not OPTIMAL_THRESHOLD:
    st.info(
        "Bitte laden Sie nun die zugehörige 'inference_summary.json'-Datei hoch."
    )
elif not uploaded_files:
    st.info(
        "Modell und Konfiguration sind geladen. Bitte laden Sie nun Bilder zur Analyse hoch."
    )

if (
    analyze_button
    and uploaded_files
    and session
    and OPTIMAL_THRESHOLD is not None
    and img_size is not None
):

    progress_bar = st.progress(0, text="Initialisiere Analyse...")

    try:
        progress_bar.progress(10, text="Verarbeite Bilder...")

        results, total_pre_time, total_inference_time, total_post_time = (
            process_images_parallel_optimized(
                uploaded_files, session, img_size, batch_size
            )
        )

        progress_bar.progress(100, text="Analyse abgeschlossen!")
        time.sleep(0.3)
        progress_bar.empty()

        num_images = len(results)
        avg_pre_time = total_pre_time / num_images if num_images else 0
        avg_inf_time = total_inference_time / num_images if num_images else 0
        avg_post_time = total_post_time / num_images if num_images else 0
        avg_total_time = avg_pre_time + avg_inf_time + avg_post_time

        st.session_state.analysis_results = results
        st.session_state.analysis_threshold = OPTIMAL_THRESHOLD
        st.session_state.analysis_metrics = {
            "num_images": num_images,
            "avg_pre_time": avg_pre_time,
            "avg_inf_time": avg_inf_time,
            "avg_post_time": avg_post_time,
            "avg_total_time": avg_total_time,
            "total_time": total_pre_time
            + total_inference_time
            + total_post_time,
        }

    except Exception as e:
        progress_bar.empty()
        st.error(f"Fehler bei der Analyse: {e}")
        st.exception(e)

if (
    "analysis_results" in st.session_state
    and st.session_state.analysis_results
):
    results = st.session_state.analysis_results
    metrics = st.session_state.analysis_metrics

    DISPLAY_THRESHOLD = (
        OPTIMAL_THRESHOLD
        if OPTIMAL_THRESHOLD is not None
        else st.session_state.analysis_threshold
    )

    num_anomalies = sum(1 for r in results if r["score"] > DISPLAY_THRESHOLD)
    num_ok = metrics["num_images"] - num_anomalies

    st.header("Analyseergebnisse")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bilder", metrics["num_images"])
    col2.metric("Anomalien", num_anomalies)
    col3.metric("OK", num_ok)
    rate = (
        (num_anomalies / metrics["num_images"] * 100)
        if metrics["num_images"] > 0
        else 0
    )
    col4.metric("Rate", f"{rate:.1f}%")

    st.divider()

    st.subheader("Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pre", f"{metrics['avg_pre_time'] * 1000:.1f}ms")
    col2.metric("Inferenz", f"{metrics['avg_inf_time'] * 1000:.1f}ms")
    col3.metric("Post", f"{metrics['avg_post_time'] * 1000:.1f}ms")
    col4.metric("Gesamt", f"{metrics['avg_total_time'] * 1000:.1f}ms")

    col1, col2 = st.columns(2)
    throughput = (
        1 / metrics["avg_total_time"] if metrics["avg_total_time"] > 0 else 0
    )
    col1.metric("Throughput", f"{throughput:.1f} FPS")
    col2.metric("Total", f"{metrics['total_time']:.2f}s")

    st.divider()

    st.subheader("Ergebnisse filtern")
    filter_option = st.radio(
        "Anzeigen:",
        ["Alle Bilder", "Nur Anomalien", "Nur OK"],
        horizontal=True,
        label_visibility="collapsed",
    )

    filtered_results = results
    if filter_option == "Nur Anomalien":
        filtered_results = [
            r for r in results if r["score"] > DISPLAY_THRESHOLD
        ]
    elif filter_option == "Nur OK":
        filtered_results = [
            r for r in results if r["score"] <= DISPLAY_THRESHOLD
        ]

    if not filtered_results:
        st.info(f"Keine Ergebnisse für Filter '{filter_option}'")
    else:
        st.caption(f"Zeige {len(filtered_results)} von {len(results)} Bildern")

        for res in filtered_results:
            is_anomaly = res["score"] > DISPLAY_THRESHOLD
            status = "Anomalie" if is_anomaly else "OK"

            expander_title = (
                f"{status} | {res['name']} | Score: {res['score']:.4f}"
            )

            with st.expander(expander_title, expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        res["image"],
                        caption="Original",
                        use_container_width=True,
                    )

                with col2:
                    st.image(
                        res["heatmap"],
                        caption="Heatmap",
                        use_container_width=True,
                    )

                st.divider()
                if is_anomaly:
                    st.error(
                        f"""
                        **Anomalie erkannt** Score: `{res['score']:.4f}` | Schwelle: `{DISPLAY_THRESHOLD:.4f}`
                    """
                    )
                else:
                    st.success(
                        f"""
                    **In Ordnung** Score: `{res['score']:.4f}` | Schwelle: `{DISPLAY_THRESHOLD:.4f}`
                    """
                    )

    st.divider()

    results_df = pd.DataFrame(
        [
            {
                "Dateiname": r["name"],
                "Score": r["score"],
                "Status": (
                    "Anomalie" if r["score"] > DISPLAY_THRESHOLD else "OK"
                ),
                "Schwellenwert": DISPLAY_THRESHOLD,
                "Latenz (ms)": r["time"] * 1000,
            }
            for r in results
        ]
    )

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Ergebnisse als CSV exportieren",
        csv_data,
        "anomalie_ergebnisse.csv",
        "text/csv",
        use_container_width=True,
    )

st.divider()
st.caption("Powered by ONNX Runtime | Built with Streamlit")