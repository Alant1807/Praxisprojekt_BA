import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import roc_curve
import yaml
from fvcore.nn import FlopCountAnalysis
from thop import profile
import timm
import pprint


def cache_teacher_features(
    model,
    dataloader,
    device,
    format,
    use_amp,
    use_non_blocking,
    cache_on_cpu=False,
):
    """Berechnet die Features des eingefrorenen Teachers einmalig vor.

    Da der Teacher nicht trainiert wird, bleiben seine Ausgaben immer gleich.
    Das Zwischenspeichern (Caching) spart enorm viel Rechenzeit pro Epoche.

    Args:
        model (nn.Module): Das Gesamtmodell (inkl. Teacher).
        dataloader (DataLoader): Daten für das Caching.
        device (torch.device): Das verwendete Gerät (CPU/GPU).
        format (torch.memory_format): Das Speicherformat (z.B. Channels Last).
        use_amp (bool): Ob Mixed Precision genutzt werden soll.
        use_non_blocking (bool): Ob asynchrone Datenübertragung genutzt wird.
        cache_on_cpu (bool, optional): Ob der Cache im Arbeitsspeicher (RAM) statt VRAM liegen soll.

    Returns:
        dict: Ein Wörterbuch mit den gesammelten Features und Meta-Daten.
    """
    # Teacher darf unter keinen Umständen trainiert werden
    model.teacher_model.eval()
    if model.stem_model:
        model.stem_model.eval()

    cache_device = torch.device("cpu") if cache_on_cpu else device

    if cache_on_cpu and device.type == "cuda":
        print("Lege Cache für Teacher-Features an... (CPU-Caching aktiviert - spart GPU Memory)")
    else:
        print("Lege Cache für Teacher-Features an... (Index-basiert, VRAM)")

    all_features_per_layer = []
    path_to_index = {}
    current_index = 0

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        for images, _, image_paths, _ in tqdm(dataloader, desc="Caching Teacher Features"):
            img_t = images.to(device, memory_format=format, non_blocking=use_non_blocking)

            stem_output = model.stem_model(img_t) if model.stem_model else img_t
            teacher_features = model.teacher_model(stem_output)
            teacher_maps = list(teacher_features.values())

            if not all_features_per_layer:
                all_features_per_layer = [[] for _ in range(len(teacher_maps))]

            for layer_idx, t_map in enumerate(teacher_maps):
                if cache_on_cpu:
                    all_features_per_layer[layer_idx].append(t_map.detach().clone().cpu())
                else:
                    all_features_per_layer[layer_idx].append(t_map.detach().clone())

            for path in image_paths:
                path_to_index[path] = current_index
                current_index += 1

    # Listen zu großen Tensoren zusammenfassen, um späteres Auslesen extrem schnell zu machen
    print("  Stacke Features zu zusammenhängenden Tensoren...")
    stacked_features = []
    total_bytes = 0

    for layer_idx, layer_features in enumerate(all_features_per_layer):
        stacked = torch.cat(layer_features, dim=0)
        stacked_features.append(stacked)
        total_bytes += stacked.numel() * stacked.element_size()

        # Temporären Speicher sofort freigeben
        all_features_per_layer[layer_idx] = None

    cache_size_mb = total_bytes / (1024 * 1024)
    print(f"Cache erstellt: {len(path_to_index)} Bilder, {len(stacked_features)} Layer, ~{cache_size_mb:.1f} MB ({cache_device})")

    return {
        "features": stacked_features,
        "path_to_index": path_to_index,
        "device": cache_device,
        "is_indexed": True,
    }


def get_teacher_features_cache(teacher_cache, image_paths, device=None, non_blocking=False):
    """Holt die vorberechneten Features für ein bestimmtes Bild aus dem Cache.

    Args:
        teacher_cache (dict): Der erstellte Cache aus `cache_teacher_features`.
        image_paths (list): Pfade der aktuell benötigten Bilder.
        device (torch.device, optional): Ziel-Gerät für den Tensor.
        non_blocking (bool, optional): Ob der Transfer asynchron passieren soll.

    Returns:
        list: Liste von Feature-Tensoren für die angefragten Bilder.
    """
    if isinstance(teacher_cache, dict) and teacher_cache.get("is_indexed", False):
        features = _get_features_indexed(teacher_cache, image_paths)

        if device is not None:
            cache_device = teacher_cache.get("device", torch.device("cpu"))
            # Auf die Grafikkarte schieben, falls der Cache im RAM lag
            if cache_device != device:
                features = [f.to(device, non_blocking=non_blocking) for f in features]

        return features


def _get_features_indexed(cache, image_paths):
    """Extrahiert die Features blitzschnell anhand der Bildpfade."""
    indices = [cache["path_to_index"][path] for path in image_paths]
    result = []
    for layer_features in cache["features"]:
        batch_features = layer_features[indices]
        result.append(batch_features)
    return result


def get_optimal_threshold(y_true, y_scores):
    """Findet den optimalen Schwellenwert zur Trennung von "Normal" und "Anomalie".

    Args:
        y_true (list/array): Die tatsächlichen Klassen-Labels.
        y_scores (list/array): Die vom Modell berechneten Anomalie-Scores.

    Returns:
        float: Der beste Schwellenwert.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Youden-Index: Sucht den Punkt mit der besten Balance aus Erkennung und Fehlalarmen
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def get_optimal_num_workers(device="cpu"):
    """Bestimmt eine sinnvolle Anzahl an Hintergrundprozessen zum Datenladen.

    Args:
        device (str, optional): Das verwendete Gerät ("cpu" oder "cuda").

    Returns:
        int: Die optimale Anzahl an Workern.
    """
    num_cpus = os.cpu_count() or 4

    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        # Bei GPUs viele Worker nutzen, um die GPU konstant mit Daten zu füttern
        return min(6, num_cpus)
    else:
        # Bei reiner CPU-Nutzung weniger Worker nutzen, damit Rechenkerne fürs Training frei bleiben
        return max(1, num_cpus // 2 - 1)


def get_optimal_batch_size(device="cpu", img_size=256, model_name="resnet18"):
    """Schätzt eine sichere Batch-Größe basierend auf dem System-Speicher (vermeidet Abstürze).

    Args:
        device (str, optional): "cpu" oder "cuda".
        img_size (int, optional): Die Bildauflösung.
        model_name (str, optional): Der Modellname (aktuell nur als Platzhalter).

    Returns:
        int: Empfohlene Batch-Größe.
    """
    is_gpu = device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda")

    if is_gpu and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if vram_gb >= 16:
            base_batch = 128
        elif vram_gb >= 8:
            base_batch = 64
        elif vram_gb >= 4:
            base_batch = 32
        else:
            base_batch = 16

        # Anpassung bei stark abweichenden Bildgrößen
        if img_size > 256:
            base_batch = base_batch // 2
        elif img_size < 224:
            base_batch = int(base_batch * 1.5)

        return base_batch
    else:
        # CPU-Strategie
        num_cpus = os.cpu_count() or 4
        if num_cpus >= 8:
            base_batch = 32
        elif num_cpus >= 4:
            base_batch = 16
        else:
            base_batch = 4

        if img_size > 256:
            base_batch = max(2, base_batch // 2)

        return base_batch


def print_system_info():
    """Gibt eine kurze Hardware- und Software-Übersicht in der Konsole aus."""
    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CPU Cores: {os.cpu_count()}")

    print(f"\nCPU Optimizations:")
    print(f"  MKL available: {torch.backends.mkl.is_available() if hasattr(torch.backends, 'mkl') else 'N/A'}")
    print(f"  MKL-DNN available: {torch.backends.mkldnn.is_available() if hasattr(torch.backends, 'mkldnn') else 'N/A'}")
    print(f"  OpenMP threads: {torch.get_num_threads()}")

    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    else:
        print(f"\nNo CUDA GPU available")

    print("=" * 50 + "\n")


def load_config(config_path):
    """Lädt eine YAML-Konfigurationsdatei sicher ein.

    Args:
        config_path (str): Der Pfad zur Datei.

    Returns:
        dict: Die Konfigurationsdaten oder None bei einem Fehler.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Die Konfigurationsdatei wurde nicht gefunden: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as exc:
        print(f"Fehler beim Parsen der YAML-Datei: {exc}")
        return None


def setup_cpu_optimizations():
    """Optimiert PyTorch für die Ausführung auf der CPU.

    Verhindert "Oversubscription" (zu viele Threads bremsen sich gegenseitig aus).

    Returns:
        dict: Aktive CPU-Konfigurationen.
    """
    num_physical_cores = os.cpu_count() // 2 if os.cpu_count() > 1 else 1

    try:
        torch.backends.mkldnn.enabled = True
    except AttributeError:
        pass

    torch.set_num_threads(num_physical_cores)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    return {
        "num_threads": num_physical_cores,
        "mkldnn_enabled": (
            torch.backends.mkldnn.is_available() if hasattr(torch.backends, "mkldnn") else False
        ),
    }


def setup_gpu_optimizations(device, config):
    """Konfiguriert CUDA/GPU Spezifika (wie Tensor Cores und Benchmark-Modus).

    Args:
        device (torch.device): Das verwendete Gerät.
        config (dict): Das Konfigurations-Wörterbuch.

    Returns:
        dict: Aktive GPU-Konfigurationen.
    """
    if device.type != "cuda":
        return {"cudnn_benchmark": False, "tf32_enabled": False}

    if config["training"]["cudnn_benchmark"]:
        torch.cuda.empty_cache()

        # Erlaubt TF32 auf neueren GPUs für höhere Geschwindigkeit bei gleicher Qualität
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Findet den schnellsten Algorithmus, falls die Bildgröße immer gleich bleibt
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return {
        "cudnn_benchmark": True if config["training"]["cudnn_benchmark"] else False,
        "tf32_enabled": True if config["training"]["cudnn_benchmark"] else False,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "BF16": True if torch.cuda.is_bf16_supported() else False,
    }


def adjust_yaml_configs(root_directory: str):
    """Aktualisiert massenhaft YAML-Konfigurationen in einem Verzeichnis.

    Args:
        root_directory (str): Der Hauptordner mit den YAML-Dateien.
    """
    if not os.path.isdir(root_directory):
        print(f"Fehler: Das Verzeichnis '{root_directory}' wurde nicht gefunden.")
        return

    print(f"Starte die Suche nach .yaml-Dateien in '{root_directory}'...\n")

    for subdir, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith((".yaml", ".yml")):
                file_path = os.path.join(subdir, filename)
                print(f"Verarbeite Datei: {file_path}")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f)

                    if not config_data:
                        print("Leer oder ungültig. Überspringe.")
                        continue

                    # Beispielhafte Änderung (hartcodiert)
                    config_data["scheduler"]["params"]["max_lr"] = 0.034385

                    with open(file_path, "w", encoding="utf-8") as f:
                        yaml.dump(config_data, f, sort_keys=False, indent=4)
                    print(f"Datei '{filename}' erfolgreich aktualisiert.")

                except Exception as e:
                    print(f"Fehler beim Verarbeiten der Datei: {e}")

    print("\n🎉 Skript beendet.")


def check_model_complexity(model_list):
    """Gibt die theoretische Größe und Rechenlast (FLOPs/MACs) von Modellen aus.

    Args:
        model_list (list): Liste mit Namen von timm-Modellen.
    """
    def human_readable(n):
        if n >= 1e9:
            return f"{n/1e9:.3f} B"
        if n >= 1e6:
            return f"{n/1e6:.3f} M"
        if n >= 1e3:
            return f"{n/1e3:.3f} K"
        return str(n)

    feste_input_groesse = (3, 256, 256) 

    for model_name in model_list:
        print("-" * 50)
        print(f"Lade Modell: {model_name}...")
        model = timm.create_model(model_name, pretrained=False).eval()

        n_params = sum(p.numel() for p in model.parameters())

        if feste_input_groesse:
            input_size = feste_input_groesse
        else:
            input_size = (getattr(model, "pretrained_cfg", {}) or {}).get("input_size", (3, 256, 256))

        dummy = torch.randn(1, *input_size)

        print(f"Input-Größe: {input_size}")
        print(f"Parameter:   {n_params:,} ({human_readable(n_params)})")

        flops = None
        macs = None

        try:
            # Versuche fvcore (berechnet FLOPs, MACs = FLOPs / 2)
            macs = FlopCountAnalysis(model, dummy).total()
            flops = macs * 2.0
            print(f"MACs:        {macs:,.0f} ({human_readable(macs)})")
            print(f"FLOPs:       {flops:,.0f} ({human_readable(flops)}) [via fvcore]")
        except Exception:
            try:
                # Fallback auf thop
                macs, _ = profile(model, inputs=(dummy,), verbose=False)
                flops = macs * 2.0
                print(f"MACs:        {macs:,.0f} ({human_readable(macs)})")
                print(f"FLOPs:       {flops:,.0f} ({human_readable(flops)}) [via thop]")
            except Exception as e:
                print("FLOPs/MACs konnten nicht berechnet werden.")
                print(f"Details: {e}")


def inspect_backbone(arch: str, img_size: int = 256):
    """Zeigt eine tabellarische Übersicht aller Schichten eines Modells.

    Args:
        arch (str): Modellname.
        img_size (int, optional): Auflösung des Eingabebildes.

    Returns:
        list: Infos zu den Layern (Name, Auflösung, Kanäle).
    """
    model = timm.create_model(arch, pretrained=False, features_only=True)
    dummy = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        features = model(dummy)

    info = model.feature_info
    print(f"\n{'=' * 60}")
    print(f"  {arch}  (Input: {img_size}×{img_size})")
    print(f"{'=' * 60}")
    print(f"  {'Index':<8} {'Layer':<25} {'Auflösung':<14} {'Kanäle':<10}")
    print(f"  {'-' * 55}")

    for i, (feat, fi) in enumerate(zip(features, info)):
        h, w = feat.shape[2], feat.shape[3]
        layer_name = fi.get("module", f"feature_{i}")
        channels = fi.get("num_chs", feat.shape[1])
        print(f"  {i:<8} {layer_name:<25} {h}×{w:<10} {channels:<10}")

    del model
    return [
        (fi.get("module", f"feature_{i}"), features[i].shape[2], fi.get("num_chs", features[i].shape[1]))
        for i, fi in enumerate(info)
    ]


def suggest_matching(teacher_arch: str, student_arch: str, img_size: int = 256):
    """Sucht Schichten in Teacher und Student, deren Bildauflösungen zusammenpassen.

    Hilft beim Konfigurieren von asymmetrischen STFPM-Modellen.

    Args:
        teacher_arch (str): Modellname des Teachers.
        student_arch (str): Modellname des Students.
        img_size (int, optional): Standardauflösung.
    """
    print(f"\n{'#' * 60}")
    print(f"  Layer-Matching: {teacher_arch} → {student_arch}")
    print(f"{'#' * 60}")

    t_info = inspect_backbone(teacher_arch, img_size)
    s_info = inspect_backbone(student_arch, img_size)

    t_by_res = {res: (name, ch) for name, res, ch in t_info}
    s_by_res = {res: (name, ch) for name, res, ch in s_info}

    common_res = sorted(set(t_by_res.keys()) & set(s_by_res.keys()), reverse=True)

    print(f"\n{'=' * 60}")
    print(f"  Vorgeschlagene Zuordnung (nach Auflösung)")
    print(f"{'=' * 60}")

    if not common_res:
        print("  Keine übereinstimmenden Auflösungen gefunden!")
        return

    print(f"  {'Auflösung':<12} {'Teacher':<22} {'Ch':<6} {'Student':<22} {'Ch':<6}")
    print(f"  {'-' * 66}")

    for res in common_res:
        t_name, t_ch = t_by_res[res]
        s_name, s_ch = s_by_res[res]
        match = "✓" if t_ch == s_ch else f"→ 1×1 Conv({s_ch}→{t_ch})"
        print(f"  {res}×{res:<8} {t_name:<22} {t_ch:<6} {s_name:<22} {s_ch:<6} {match}")


def get_all_model_feature_infos(model_name):
    """Gibt die Feature-Infos eines Modells in der Konsole aus."""
    model = timm.create_model(model_name, pretrained=False, features_only=True)
    print("Die optimalen Layer für die Feature-Extraktion sind:")
    pprint.pprint(model.feature_info.info)


def get_all_timm_backbones():
    """Listet alle in der `timm` Bibliothek verfügbaren Modell-Architekturen auf."""
    all_models = timm.list_models()
    print(f"Insgesamt {len(all_models)} Modelle verfügbar:\n")
    for model in sorted(all_models):
        print(model)


def analyze_real_stfpm_complexity(stfpm_model, sample_input, device, teacher_name, student_name):
    """Überprüft die exakte Rechenlast (GMACs/GFLOPs) mit einem echten Bild-Tensor.

    Args:
        stfpm_model (nn.Module): Das konfigurierte Gesamtmodell.
        sample_input (torch.Tensor): Ein Beispiel-Bild aus dem DataLoader.
        device (torch.device): CPU oder GPU.
        teacher_name (str): Architektur-Name des Lehrers.
        student_name (str): Architektur-Name des Schülers.
    """
    def get_complexity(module, dummy_input):
        if module is None or isinstance(module, torch.nn.Identity):
            return 0, 0
            
        n_params = sum(p.numel() for p in getattr(module, "parameters", lambda: [])())
        macs = 0
        
        try:
            from fvcore.nn import FlopCountAnalysis
            macs = FlopCountAnalysis(module, dummy_input).total()
        except Exception:
            try:
                from thop import profile
                macs, _ = profile(module, inputs=(dummy_input,), verbose=False)
            except Exception:
                pass
                
        return n_params, macs

    input_shape = tuple(sample_input.shape)
    print(f"\n=== STFPM Komplexitäts-Analyse (Echter Input: {input_shape}) ===")
    print(f"  Teacher Backbone: {teacher_name}")
    print(f"  Student Backbone: {student_name}\n")
    
    stfpm_model.eval().to(device)
    
    stem = stfpm_model.stem_model if hasattr(stfpm_model, 'stem_model') else torch.nn.Identity()
    stem_params, stem_macs = get_complexity(stem, sample_input)
    
    with torch.no_grad():
        stem_output = stem(sample_input)
        
    t_params, t_macs = get_complexity(stfpm_model.teacher_model, stem_output)
    s_params, s_macs = get_complexity(stfpm_model.student_model, stem_output)
    
    # Projektionsköpfe zählen zur Komplexität des Students (nur asymmetrischer Modus)
    if hasattr(stfpm_model, 'is_asymmetric') and stfpm_model.is_asymmetric:
        with torch.no_grad():
            student_features = stfpm_model.student_model(stem_output)
            if isinstance(student_features, dict):
                student_features = list(student_features.values())
                
        for i, s_map in enumerate(student_features):
            p_params, p_macs = get_complexity(stfpm_model.projection_heads[i], s_map)
            s_params += p_params
            s_macs += p_macs

    def calc_metrics(params, macs):
        flops = macs * 2.0
        return params / 1e6, macs / 1e9, flops / 1e9, (params * 4) / (1024 * 1024)

    stem_mparams, stem_gmacs, stem_gflops, stem_size = calc_metrics(stem_params, stem_macs)
    t_mparams, t_gmacs, t_gflops, t_size = calc_metrics(t_params, t_macs)
    s_mparams, s_gmacs, s_gflops, s_size = calc_metrics(s_params, s_macs)

    print(f"  {'Komponente':<18} | {'MParams':<10} | {'GMACs':<10} | {'GFLOPs':<10} | {'Größe (MB)':<10}")
    print(f"  {'-'*68}")
    if stem_params > 0:
        print(f"  {'Stem Layer':<18} | {stem_mparams:<10.2f} | {stem_gmacs:<10.2f} | {stem_gflops:<10.2f} | {stem_size:<10.2f}")
    print(f"  {'Teacher':<18} | {t_mparams:<10.2f} | {t_gmacs:<10.2f} | {t_gflops:<10.2f} | {t_size:<10.2f}")
    print(f"  {'Student (+Proj.)':<18} | {s_mparams:<10.2f} | {s_gmacs:<10.2f} | {s_gflops:<10.2f} | {s_size:<10.2f}")
    print(f"  {'-'*68}")
    
    ratio_p = (s_params / t_params) * 100 if t_params > 0 else 0
    ratio_m = (s_macs / t_macs) * 100 if t_macs > 0 else 0
    
    print(f"\n  Verhältnis Student zu Teacher (Ressourcen-Kosten):")
    print(f"  Parameter & Größe: {ratio_p:.1f}% des Teachers")
    if t_macs > 0:
        print(f"  Rechenlast:        {ratio_m:.1f}% des Teachers (GMACs/GFLOPs)")
        
    inf_mparams = stem_mparams + s_mparams
    inf_gmacs = stem_gmacs + s_gmacs
    inf_gflops = stem_gflops + s_gflops
    inf_size = stem_size + s_size
    
    print(f"\n  Reale Inferenzlast im Einsatz (nur Stem + Student):")
    print(f"  Total MParams: {inf_mparams:.2f} M")
    print(f"  Total GMACs:   {inf_gmacs:.2f} G")
    print(f"  Total GFLOPs:  {inf_gflops:.2f} G")
    print(f"  Total Größe:   {inf_size:.2f} MB")
    print("="*70)