import torch
from torch import nn
# Aktualisierte Imports für die neue API
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping

import yaml
import os
import json

from Scripts.model import STFPM
from torch.utils.data import DataLoader
from Scripts.dataset import MVTecDataset


def quantize_model(best_student_weight_path, config, summary_metric):
    """
    Führt die statische Quantisierung im FX Graph Mode durch.
    """
    print("--- Starte Quantisierungsprozess im FX Graph Mode ---")
    torch.backends.quantized.engine = "fbgemm"
    cpu_device = torch.device("cpu")

    # 1. Modell laden
    print("Lade Modell...")
    model = STFPM(
        architecture=config['model']['architecture'],
        layers=config['model']['layers']
    )
    # Lade die Gewichte direkt in das student_model
    model.student_model.load_state_dict(torch.load(
        best_student_weight_path, map_location="cpu"))
    model.to(cpu_device).eval()

    # Wir quantisieren nur das student_model
    student_model = model.student_model
    student_model.eval()

    # 2. Quantisierung vorbereiten (prepare_fx)
    print("Bereite das Modell für die Quantisierung vor (prepare_fx)...")
    # Verwende die neue get_default_qconfig_mapping Funktion
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")

    # Beispiel-Input für den Tracer
    example_inputs = (torch.randn(1, 3, 256, 256),)

    prepared_model = prepare_fx(student_model, qconfig_mapping, example_inputs)

    # 3. Kalibrierung
    print("Starte Kalibrierung...")
    calibrate_model(prepared_model, model.stem_model, config, cpu_device)
    print("Kalibrierung abgeschlossen.")

    # 4. Modell konvertieren (convert_fx)
    print("Konvertiere Modell...")
    quantized_model = convert_fx(prepared_model)
    print("Modell erfolgreich quantisiert.")

    # Speichern des quantisierten Modells
    save_quantized_model(quantized_model, config, summary_metric)


def calibrate_model(model, stem_model, config, device):
    """
    Kalibriert das vorbereitete FX-Modell.
    """
    model.eval()
    stem_model.eval()

    train_set = MVTecDataset(
        img_size=config['dataset']['img_size'],
        base_path=config['dataset']['base_path'],
        cls=config['dataset']['class'],
        mode='train',
        subfolders=['good']
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config['dataloader']['batch_size'],
        shuffle=True
    )
    with torch.no_grad():
        for i, (images, _, _, _) in enumerate(train_loader):
            if i >= 10:
                break
            images = images.to(device)
            stem_output = stem_model(images)
            model(stem_output)

# Die save_quantized_model Funktion bleibt unverändert


def save_quantized_model(model, config, summary_metric):
    save_path = os.path.join(
        'quantized_models',
        f"{config['model']['architecture']}",
        f"{summary_metric.get('training_id', 'quantized_run')}"
    )
    os.makedirs(save_path, exist_ok=True)

    yaml_path = os.path.join(
        save_path, f"STFPM_Config_{config['model']['architecture']}.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    summary_metric_path = os.path.join(save_path, 'summary_metric.json')
    with open(summary_metric_path, 'w') as f:
        json.dump(summary_metric, f, indent=4)

    torch.save(model.state_dict(), os.path.join(
        save_path, 'quantized_model.pth'))

    print(f"Quantisiertes Modell gespeichert unter: {save_path}")
