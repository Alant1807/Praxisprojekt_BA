"""
Dieses Skript führt eine Post-Training Static Quantization auf einem STFPM-Modell durch.
Der Prozess umfasst das Laden eines vortrainierten FP32-Modells, die Kalibrierung mit
einem repräsentativen Datensatz und die Konvertierung in ein INT8-Modell zur
Optimierung der Inferenzgeschwindigkeit auf CPUs.
"""

# --- Importe ---
# Standardbibliotheken
from pathlib import Path
import json

# Externe Bibliotheken
import torch
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qconfig_mapping, quantize_fx
import yaml

# Lokale Module (benutzerdefinierte Klassen)
from Scripts.model import STFPM
from Scripts.model2 import *
from Scripts.dataset import MVTecDataset

# --- Konstanten ---
# Definiert die Anzahl der Daten-Batches für die Kalibrierung.
# Eine höhere Anzahl kann die Genauigkeit verbessern, erhöht aber die Dauer des Quantisierungsprozesses.
CALIBRATION_BATCHES = 200
# Legt das Zielgerät fest. Quantisierung wird typischerweise für die CPU-Inferenz optimiert.
DEVICE = torch.device('cpu')


def quantize_model(model_weights_path: str, config: dict, summary_metric: dict):
    """
    Orchestriert den gesamten Quantisierungsprozess für das Studenten-Modell.

    Args:
        model_weights_path (str): Pfad zu den gespeicherten Gewichten des FP32-Studenten-Modells.
        config (dict): Konfigurationswörterbuch, das Modell- und Datensatzparameter enthält.
        summary_metric (dict): Metriken aus dem Training, die zusammen mit dem Modell gespeichert werden.
    """
    print("Lade FP32-Modelle für die Quantisierung...")
    # Initialisiere das gesamte STFPM-Modell, um Zugriff auf den `stem_model` (in FP32)
    # und den `student_model` (zu quantisieren) zu erhalten.
    model_to_quantize = STFPM_QuantizedModels(
        architecture=config['model']['architecture'],
        layers=config['model']['layers'],
        quantize=False
    ).to(DEVICE).eval()

    # Trenne den Stem (Feature Extractor) und den Studenten (Anomaly Detector).
    # Der Stem bleibt in FP32, um die Genauigkeit zu erhalten, nur der Student wird quantisiert.
    stem_model = model_to_quantize.stem_model
    student_to_quantize = model_to_quantize.student_model

    # Lade die vortrainierten Gewichte in das Studenten-Modell.
    student_to_quantize.load_state_dict(
        torch.load(model_weights_path, map_location=DEVICE)
    )
    student_to_quantize.to(DEVICE).eval()

    # Wähle die Quantisierungskonfiguration. 'fbgemm' ist der empfohlene Backend für x86-CPUs.
    qconfig_mapping = get_default_qconfig_mapping('fbgemm')

    # Erstelle einen Beispiel-Input, damit FX Tracing den Graphen des Modells analysieren kann.
    # Die Dimensionen müssen mit der erwarteten Eingabe des Modells übereinstimmen.
    example_inputs = (stem_model(torch.randn(
        1, 3, config['dataset']['img_size'], config['dataset']['img_size']
    )),)

    print("Bereite das Studenten-Modell für die Quantisierung vor (FX Graph Transformation)...")
    # `prepare_fx` fügt "Observer"-Module in den Modellgraphen ein.
    # Diese Observer sammeln während der Kalibrierung Statistiken (z.B. min/max-Werte) über die Aktivierungen.
    model_prepared = quantize_fx.prepare_fx(
        student_to_quantize, qconfig_mapping, example_inputs)

    # Richte den DataLoader für die Kalibrierungsdaten ein.
    calibration_loader = setup_calibration_loader(config)

    # Führe die Kalibrierung durch, um die Observer mit Daten zu füttern.
    calibrate_model(model_prepared, calibration_loader, stem_model)

    print("Konvertiere das vorbereitete Modell zum endgültigen quantisierten Modell...")
    # `convert_fx` verwendet die gesammelten Statistiken, um die Gewichte und Aktivierungen
    # von FP32 in INT8 zu konvertieren und die Observer zu entfernen.
    quantized_student_model = quantize_fx.convert_fx(model_prepared)

    # Speichere das quantisierte Modell und die zugehörigen Artefakte.
    save_artifacts(quantized_student_model, config, summary_metric)


def setup_calibration_loader(config: dict) -> DataLoader:
    """
    Erstellt und konfiguriert einen DataLoader für die Kalibrierungsphase.

    Args:
        config (dict): Das Konfigurationswörterbuch mit Datensatzinformationen.

    Returns:
        DataLoader: Ein DataLoader, der "gute" (anomaliefreie) Trainingsbilder liefert.
    """
    print("Erstelle den Kalibrierungs-Daten-Loader...")
    train_set = MVTecDataset(
        img_size=config['dataset']['img_size'],
        base_path=config['dataset']['base_path'],
        cls=config['dataset']['class'],
        mode='train',
        # Zur Kalibrierung werden nur "gute" Bilder verwendet, da sie die
        # typische Datenverteilung im Normalbetrieb repräsentieren.
        subfolders=['good']
    )
    return DataLoader(train_set, batch_size=16, shuffle=True)


def calibrate_model(model: torch.nn.Module, calibration_loader: DataLoader, stem_model: torch.nn.Module):
    """
    Führt den Kalibrierungsprozess durch, indem Daten durch das vorbereitete Modell geleitet werden.

    Args:
        model (torch.nn.Module): Das mit Observern vorbereitete Modell.
        calibration_loader (DataLoader): Der DataLoader mit Kalibrierungsdaten.
        stem_model (torch.nn.Module): Das FP32-Stem-Modell zur Merkmalsextraktion.
    """
    print("Starte die Kalibrierung des Modells...")
    model.eval()
    stem_model.eval()

    # `torch.no_grad()` deaktiviert die Gradientenberechnung, was Speicher und Rechenzeit spart,
    # da für die Kalibrierung kein Training (Backpropagation) erforderlich ist.
    with torch.no_grad():
        for i, (images, _, _, _) in enumerate(calibration_loader):
            if i >= CALIBRATION_BATCHES:
                break
            model(stem_model(images.to(DEVICE)))

    print("Kalibrierung abgeschlossen.")


def save_artifacts(model: torch.nn.Module, config: dict, summary_metric: dict):
    """
    Speichert das quantisierte Modell, die Konfiguration und die Trainingsmetriken in einem strukturierten Verzeichnis.

    Args:
        model (torch.nn.Module): Das fertig quantisierte Modell.
        config (dict): Die Konfigurationsparameter des Laufs.
        summary_metric (dict): Die zugehörigen Trainings-Metriken.
    """
    # Erzeuge einen eindeutigen Pfad basierend auf Datensatz, Modellarchitektur und Trainings-ID.
    run_id = summary_metric.get('training_id', 'quantized_run')
    base_path = Path('quantized_models') / \
        f"{config['dataset']['name']}_{config['dataset']['class']}" / \
        config['model']['architecture'] / run_id

    # Erstelle die Verzeichnisstruktur, falls sie noch nicht existiert.
    base_path.mkdir(parents=True, exist_ok=True)

    # Speichere die Konfigurationsdatei (YAML).
    config_path = base_path / \
        f"STFPM_Config_{config['model']['architecture']}.yaml"
    with config_path.open('w') as f:
        yaml.dump(config, f)

    # Speichere die Metriken (JSON).
    summary_path = base_path / 'summary_metric.json'
    with summary_path.open('w') as f:
        json.dump(summary_metric, f, indent=4)

    # Speichere die Gewichte des quantisierten Modells.
    model_path = base_path / 'quantized_model.pth'
    torch.save(model.state_dict(), model_path)
    print(
        f"Quantisiertes Modell und Artefakte erfolgreich gespeichert unter: {base_path}")
