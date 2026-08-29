# Vergleich von Optimierungsverfahren neuronaler Netze für Edge-Computing in der Anomalieerkennung

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat&logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![License](https://img.shields.io/badge/License-Academic-green?style=flat)
![Status](https://img.shields.io/badge/Status-Maintained-blue?style=flat)

Optimierte Implementierung des **Student-Teacher Feature Pyramid Matching (STFPM)** Algorithmus für Anomalieerkennung in industriellen Anwendungen. Entwickelt im Rahmen einer Bachelorarbeit an der FH Aachen.

**Fokus:** Edge-Computing-Deployment ohne teure GPU-Infrastruktur – optimiert für KMUs.

Dieses Repository enthält eine für Edge-Computing optimierte Implementierung des STFPM-Ansatzes, basierend auf dem Paper [Student-Teacher Feature Pyramid Matching for Anomaly Detection](https://arxiv.org/pdf/2103.04257).

## Funktionsweise (Architektur)

![STFPM Architektur Übersicht](assets/stfpm.png)

## Features

- **Symmetrisches & Asymmetrisches Training** – Teacher-Student Architekturen mit verschiedenen Backbones (ResNet18, MobileNetV3, EfficientNet)
- **Performance-Optimierungen**
- **Flexible Konfiguration** – YAML-basierte Konfiguration für alle Hyperparameter
- **Offline Data Caching** - Vorverarbeitung von Bilddaten in optimierte uint8-Tensoren zur Eliminierung von CPU-Bottlenecks und massiver Beschleunigung des Dataloadings.
- **Ermittlung optimaler Lernrate** - Automatisierte Ermittlung der Lernrate
- **Umfangreiches Benchmarking** – Automatische Metriken und Timing-Analyse
- **Pareto-Front** - Erzeugung von Pareto-Fronts zur Unterstützung der Modellwahl
- **ONNX Export** – Deployment-ready Modelle für CPU/GPU Inferenz inklusive End-to-End Pre- und Postprocessing im Graphen.
- **Streamlit Webapp** – Interaktive Anomalieerkennung mit Drag-and-Drop, Batch-Verarbeitung und Heatmap-Visualisierung.

## Installation

### Voraussetzungen

- Python 3.10+
- CUDA 11.8+ (optional, für GPU-Training)

### Setup

```bash
# Repository klonen
git clone https://github.com/Alant1807/Praxisprojekt_BA.git
cd stfpm-optimization

# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Quick Start

Alle folgenden Schritte können direkt im beigefügten Jupyter Notebook main.ipynb ausgeführt werden, oder Sie erstellen eigene Python-Skripte basierend auf diesen Beispielen.

### 1. Optional: Daten cachen (Offline Preprocessing)
Dieser Schritt ist **optional**, wird jedoch für **maximale Trainingsgeschwindigkeit** empfohlen. Er konvertiert die Bilder vorab in optimierte `uint8`-Tensoren, um CPU-Engpässe zu eliminieren.
> **Hinweis:** Wird dieser Schritt übersprungen, lädt das System die Bilder während des Trainings live von der Festplatte (Standard-Verhalten), was je nach Hardware langsamer sein kann. Der Cacheordner befindet sich immer als Unterordner im Hauptordner, wo sich die Bilder befinden aus Kompatibilitätsgründen
```python
from Scripts.dataset_gkd import create_tensor_cache

# Erstellt automatisch einen Unterordner mit Suffix 'cached_uint8' 
# (z.B. /tmp/labeled_and_sliced/cached_uint8)
create_tensor_cache(
    data_path='/tmp/labeled_and_sliced/',  # Pfad zu den Rohbildern
    img_size=256
)
```

### 2. Konfiguration generieren

```python
from Scripts.config_generator import *

# Konfigurationen erstellen
generate_gkdConfigs(mode='cpu')
generate_gkdConfigs(mode='baseline')
generate_gkdConfigs(mode='gpu')
```
**Hinweis zur Anpassung:** Die generierten YAML-Dateien dienen als Startpunkt.

Sie haben zwei Möglichkeiten zur Anpassung:

1. **Manuell:** Öffnen Sie die erstellten ```.yaml```-Dateien (im Ordner ```Configs_...```) und ändern Sie Werte wie ```batch_size```, ```learning_rate``` oder Pfade direkt.
2. **Global:** Passen Sie die Standard-Parameter direkt im ```Skript Scripts/config_generator.py``` an, bevor Sie die Generator-Funktion ausführen.

### 3. Optimale Lernrate ermitteln

```python
from Scripts.lr_finder import *

# Erzeuge Runs zur Ermittlung optimierter Lernraten
# Optional gecachte Daten nutzen, wenn verfügbar
# Bei Bedarf Skript erstellen, der mehrere YAML in einem Ordner iteriert
run_lr_range_test(config_path: 'Configs_GKD\resnet18\STFPM_Config_resnet18_GKD.yaml', use_cached_tensors: bool = True, start_lr: float = 1e-7, end_lr: float = 1.0, num_epochs: int = 5, smoothing_beta: float = 0.9)
```

### 4. Training starten

```python
from Scripts.pipeline_runner import Training_GKD

# Training mit einem/mehreren Modellarchitekturen als Backbone
# Optional gecachte Daten nutzen, wenn verfügbar
Training_GKD(
    config_dir='Configs_GKD',
    output_dir='Training_GKD_CPU',
    allowed_models=['resnet18', 'mobilenetv4_conv_small'],
    used_cached_inputs: bool = True
)

# Training mit einem/mehreren spezifischen Yaml Konfigurationen
# Optional gecachte Daten nutzen, wenn verfügbar 
Training_GKD(
    config_dir='Configs_GKD',
    output_dir='Training_GKD_CPU',
    allowed_models=['resnet18'],
    allowed_configs=['STFPM_Config_resnet18_GKD_baseline.yaml', 'STFPM_Config_resnet18_GKD_optimized.yaml'],
    used_cached_inputs: bool = False
)
```

### 5. Inferenz durchführen

```python
from Scripts.pipeline_runner import inference_model_GKD

# Evaluierung auf Testdaten
inference_model_GKD(
    training_dir='Training_GKD_CPU',
    output_dir='Inference_GKD_CPU',
    allowed_models: list: None
    used_cached_inputs: bool = False
    load_to_ram: bool = False
    detailed_profiling; bool = False
)

```

### 6. ONNX Export

```python
from Scripts.run_onnx_export import export_all_models_to_onnx

# Export für Webapp-Deployment
export_all_models_to_onnx(
    training_dir='Training_GKD_CPU',
    inference_dir='Inference_GKD_CPU',
    output_base_dir='exported_onnx_models'
)
```

### 7. Pareto-Front

```python
from Scripts.plot_pareto import *

# Erzeuge Pareto-Front Plots
# immer das Artefakt-Bundle nehmen als base_path (Nach ONNX-Export erzeugt)
execute_pareto_plot(base_path='exported_onnx_models', output_path: str)
```

## Projektstruktur

```
stfpm-optimization/
├── Scripts/
│   ├── model.py              # STFPM Modell (Teacher-Student Architektur)
│   ├── trainer.py            # Training-Loop mit Optimierungen
│   ├── inference.py          # Evaluierung & Metriken
│   ├── loss.py               # Loss-Funktionen (MSE, Cosine Similarity)
│   ├── dataset.py            # MVTec Dataset Loader
│   ├── dataset_gkd.py        # GKD Dataset Loader
│   ├── generate_configs.py   # YAML-Konfigurationsgenerator
│   ├── execute.py            # Orchestrierung (Training, Inference)
│   ├── export_onnx_packages.py  # ONNX Export Pipeline
│   ├── Onnx_Class.py         # ONNX Exporter Klasse
│   ├── plots.py              # Visualisierung (Heatmaps, Plots)
│   ├── results_manager.py    # Ergebnis-Aggregation
│   ├── utils.py              # Hilfsfunktionen
│   ├── lr_finder.py          # Learning Rate Finder
│   ├── create_pareto_plot.py # Pareto-Analyse
│   └── upload_summaries.py   # Ergebnis-Upload
├── webapp
|   ├── app.py                # Streamlit Webapp
|   └── main.py               # Webapp Launcher (für .exe)
├── main.ipynb                # Jupyter Notebook (interaktive Nutzung)
├── Configs_GKD/              # Generierte Konfigurationen
├── lr_finder_plots_GKD/      # Plots und Rohdaten zu ermittelten optimalen Lernraten
├── Training_GKD_CPU/         # Trainings-Ergebnisse
├── Inference_GKD_CPU/        # Inferenz-Ergebnisse
├── exported_onnx_models/     # ONNX-Pakete für Deployment
└── Pareto-Front_plots/       # Speichert Plots zu den Pareto-Fronts

```

## Konfiguration

Alle Parameter werden über YAML-Dateien gesteuert. Beispiel:

```yaml
dataset:
    name: GKD
    base_path: GKD_Bilder
    img_size: 256
    enable_timing: false
dataloader:
    batch_size: 32
    num_workers: 6
    pin_memory: false
    persistent_workers: true
    prefetch_factor: 2
epochs: 100
optimizer:
    active: AdamW
    configs:
        SGD:
            lr: 0.4
            momentum: 0.9
            weight_decay: 0.0001
            nesterov: true
        AdamW:
            lr: 0.001
            weight_decay: 0.01
training:
    cache_teacher_features: true
    fast_zero_grad: true
    async_host_to_device: false
    force_cpu_caching: true
    cudnn_benchmark: false
    detailed_timing: false
    profiling: false
    use_qat: false
    qat_backend: fbgemm
scheduler:
    type: OneCycleLR
    params:
        max_lr: 0.001
        epochs: 100
        pct_start: 0.3
        anneal_strategy: cos
model_settings:
    channels_last: true
    amp_mixed_precision: false
    is_asymmetric: false
    shared_stem: true
    partial_share_depth: 1
    use_original_paper: false
loss:
    params:
        epsilon: 1.0e-08
        reduction: sum
        use_original_paper: false
model:
    architecture: mobilenetv4_conv_small_050
    layers:
    - blocks.0
    - blocks.1
    - blocks.2
    - blocks.4


```

### Optimierungs-Modi

| Modus | Beschreibung |
|-------|--------------|
| `gpu` | Alle GPU-Optimierungen aktiv (AMP, Channels Last, CUDA) |
| `cpu` | CPU-optimiert (kein AMP, angepasstes Threading) |
| `baseline` | Keine Optimierungen (für Vergleichsmessungen) |

## Webapp

Die Streamlit-Webapp ermöglicht interaktive Anomalieerkennung:

**Features:**
- Drag & Drop Upload von ONNX-Modellpaketen
- Batch-Verarbeitung mehrerer Bilder
- Echtzeit-Heatmap-Visualisierung
- Anpassbarer Threshold (Youden Index als Startpunkt)
- Filterung nach Anomalie-Status
- Export der Ergebnisse

#### Webapp als Executable erstellen

Um die Anwendung als eigenständige `.exe` Datei zu exportieren, werden spezielle Hooks für Streamlit, OpenCV und ONNX benötigt. Führen Sie folgenden Befehl im `webapp`-Verzeichnis aus:

```bash
# Wechsele ins webapp Verzeichnis
cd webapp

# Befehl um .exe zu erzeugen
pyinstaller --noconfirm --onefile --console --name "AnomalieDetektor" --add-data "app.py;." --copy-metadata streamlit --collect-all streamlit --collect-all cv2 --collect-all onnxruntime --collect-all PIL main.py
```
Hinweise:

Der Vorgang kann einige Minuten dauern.

Die fertige Datei befindet sich im Ordner dist/.

Der Parameter --add-data "app.py;." nutzt Windows-Syntax (Semikolon). Unter Linux/Mac müsste ein Doppelpunkt (:) verwendet werden.

#### Webapp direkt starten

```bash
streamlit run webapp/app.py
```

## Ergebnisse

### Trainings-Output

Nach dem Training werden folgende Dateien erstellt:

```
Training_GKD_CPU/
└── GKD/
    └── <model>/
        └── <uuid>/
            ├── logs/
                └── <model>_GKD_<uuid>_results.csv  # Alle Metriken als CSV gespeichert
            ├── plots/
                ├── auroc.png                       # Zeigt den AUROC-Score Verlauf über mehrere Epochen
                └── loss.png                        # Zeigt den Loss Verlauf über alle Epochen
            ├── weights/                            # Speichert die Gewichte
            ├── config.yaml                         # angewendete YAML
            └── summary_metrics.json                # Speichert alle wichtigen Trainingsmetriken
        └── <model>_model_variant_summary.csv       # Speichert zu jedem Trainingsrun für die selbe Modellarchitektur die Metriken
```

### Inferenz-Output

```
Inference_GKD_CPU/
└── GKD/
    └── <model>/
        └── <uuid>/
            ├── config.yaml                               # angewendete YAML 
            └── inference_summary.json                    # Speichert alle wichtigen Metriken zur Inferenz
        └── <model>_model_variant_inference_summary.csv   # Speichert zu jedem Inferenzrun für die selbe Modellarchitketur die Metriken
```

### ONNX-Paket

```
exported_onnx_models/
└── GKD/
    └── <model>/
        ├── training_result/                              # Alle Artefakte zum Trainingsrun
        ├── inference.json                                # Metriken zur Inferenz
        ├── model.onnx                                    # Exportiertes ONNX Modell
        └── <model>_<uuid>_onnx_model_config.yaml         # Konfiguration zum ONNX Modell als YAML
    └── <model>_model_variant_inference_summary.csv       # Alle Metriken pro Modell 
```

Diese Implementierung ist Teil der Bachelorarbeit:

> **"Vergleich von Methoden zur Optimierung neuronaler Netze für Edge Computing in der industriellen Anomalieerkennung basierend auf Student-Teacher Feature Pyramid Matching"**

FH Aachen University of Applied Sciences, 2025/2026

## Zitation

```bibtex
@inproceedings{wang2021student_teacher,
    title={Student-Teacher Feature Pyramid Matching for Anomaly Detection},
    author={Wang, Guodong and Han, Shumin and Ding, Errui and Huang, Di},
    booktitle={The British Machine Vision Conference (BMVC)},
    year={2021}
}
```

## Lizenz

Dieses Projekt wurde im Rahmen einer akademischen Arbeit erstellt.

## Kontakt

Bei Fragen zur Implementierung oder Nutzung: alan.tofeq@alumni.fh-aachen.de
