# Importieren der notwendigen Bibliotheken und Module
from torch.ao.quantization import get_default_qconfig_mapping, quantize_fx
from pathlib import Path
import torch
import yaml
import json
from torch.utils.data import DataLoader
# Importieren der benutzerdefinierten Klassen aus den Skript-Dateien
from Scripts.model2 import *
from Scripts.dataset import MVTecDataset

# --- KONSTANTEN ---
# Definiert die Anzahl der Daten-Batches, die für die Kalibrierung verwendet werden.
CALIBRATION_BATCHES = 20
# Legt das Gerät fest, auf dem die Berechnungen ausgeführt werden (hier CPU, da Quantisierung oft für CPU-Inferenz optimiert ist).
DEVICE = torch.device('cpu')


def quantize_model(model_weights_path, config, summary_metric):
    print("Lade Modelle fuer die Quantisierung...")
    model_to_quantize = STFPM(
        architecture=config['model']['architecture'],
        layers=config['model']['layers'],
        quantize=False
    ).to(DEVICE).eval()
    stem_model = model_to_quantize.stem_model
    student_to_quantize = model_to_quantize.student_model
    student_to_quantize.load_state_dict(
        torch.load(model_weights_path, map_location=DEVICE)
    )
    student_to_quantize.to(DEVICE).eval()
    qconfig_mapping = get_default_qconfig_mapping('fbgemm')
    example_inputs = (stem_model(torch.randn(
        1, 3, config['dataset']['img_size'], config['dataset']['img_size']
    )),)
    print("Bereite das Studenten-Modell fuer die Quantisierung vor...")
    model_prepared = quantize_fx.prepare_fx(
        student_to_quantize, qconfig_mapping, example_inputs)
    calibration_loader = setup_calibration_loader(config)
    calibrate_model(model_prepared, calibration_loader, stem_model)
    print("Konvertiere das zum quantisierten Modell...")
    quantized_student_model = quantize_fx.convert_fx(model_prepared)
    save_artifacts(quantized_student_model, config, summary_metric)


def setup_calibration_loader(config):
    print("Erstelle den Kalibrierungs-Daten-Loader...")
    train_set = MVTecDataset(
        img_size=config['dataset']['img_size'],
        base_path=config['dataset']['base_path'],
        cls=config['dataset']['class'],
        mode='train',
        # Nur "gute" Bilder werden zur Kalibrierung verwendet
        subfolders=['good']
    )
    return DataLoader(train_set, batch_size=16, shuffle=True)


def calibrate_model(model, calibration_loader, stem_model):
    print("Kalibriere das Modell...")
    model.eval()
    stem_model.eval()
    with torch.no_grad():  # Gradientenberechnung ist hier nicht nötig
        for i, (images, _, _, _) in enumerate(calibration_loader):
            if i >= CALIBRATION_BATCHES:
                break
            # Zuerst werden die Bilder durch den FP32-Stem geleitet.
            stem_output = stem_model(images.to(DEVICE))
            # Der Output des Stems wird dann in das vorbereitete Modell gegeben, um die Observer zu füttern.
            model(stem_output)
    print("Kalibrierung abgeschlossen.")


def save_artifacts(model, config, summary_metric):
    """
    Speichert das quantisierte Modell, die zugehörige Konfiguration und die Trainings-Metriken.

    Args:
        model (torch.nn.Module): Das quantisierte Modell.
        config (dict): Die Konfigurationsparameter.
        summary_metric (dict): Die Trainings-Metriken.
    """
    # Erzeuge einen eindeutigen Pfad basierend auf den Konfigurationsdetails.
    run_id = summary_metric.get('training_id', 'quantized_run')
    base_path = Path('quantized_models') / \
        f"{config['dataset']['name']}_{config['dataset']['class']}" / \
        config['model']['architecture'] / run_id

    # Erstelle die Verzeichnisse, falls sie nicht existieren.
    base_path.mkdir(parents=True, exist_ok=True)

    # Speichere die Konfigurations-Datei (YAML).
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
    print(f"Quantisiertes Modell und Artefakte gespeichert unter: {base_path}")
