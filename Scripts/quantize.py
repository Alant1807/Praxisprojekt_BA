from torch.ao.quantization import get_default_qconfig_mapping, quantize_fx
from pathlib import Path
import torch
import copy
import os
import yaml
import json
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import DataLoader
from Scripts.model2 import *
from Scripts.dataset import MVTecDataset

# --- KONSTANTEN ---
CALIBRATION_BATCHES = 20
DEVICE = torch.device('cpu')
# -----------------


def quantize_model(model_weights_path, config, summary_metric):
    """
    Hauptfunktion zur Quantisierung eines STFPM-Modells.

    Lädt ein vortrainiertes Modell, führt eine Post-Training Static Quantization
    mit Kalibrierung durch und speichert das Ergebnis.
    """
    print("Loading model for quantization...")
    model_to_quantize = STFPM(
        architecture=config['model']['architecture'],
        layers=config['model']['layers'],
        quantize=True  
    ).to(DEVICE)

    model_to_quantize.student_model.load_state_dict(
        torch.load(model_weights_path, map_location=DEVICE)
    )
    model_to_quantize.eval()

    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    example_inputs = (torch.randn(1, 3, 256, 256),)

    print("Preparing model for quantization...")
    model_prepared = quantize_fx.prepare_fx(
        model_to_quantize, qconfig_mapping, example_inputs
    )

    calibration_loader = setup_calibration_loader(config)
    calibrate_model(model_prepared, calibration_loader,
                    model_to_quantize.stem_model)

    print("Converting to quantized model...")
    quantized_model = quantize_fx.convert_fx(model_prepared)

    save_artifacts(quantized_model, config, summary_metric)


def setup_calibration_loader(config):
    """Bereitet den DataLoader für die Kalibrierung vor."""
    print("Setting up calibration data loader...")
    train_set = MVTecDataset(
        img_size=config['dataset']['img_size'],
        base_path=config['dataset']['base_path'],
        cls=config['dataset']['class'],
        mode='train',
        subfolders=['good']
    )
    return DataLoader(train_set, batch_size=16, shuffle=True)


def calibrate_model(model, calibration_loader, stem_model):
    """Führt die Kalibrierung des vorbereiteten Modells durch."""
    print("Calibrating the model...")
    model.eval()
    stem_model.eval()
    with torch.no_grad():
        for i, (images, _, _, _) in enumerate(calibration_loader):
            if i >= CALIBRATION_BATCHES:
                break
            stem_output = stem_model(images.to(DEVICE))
            model(stem_output)
    print("Calibration completed.")


def save_artifacts(model, config, summary_metric):
    """Speichert das quantisierte Modell, die Konfiguration und die Metriken."""
    run_id = summary_metric.get('training_id', 'quantized_run')
    base_path = Path('quantized_models') / \
        f"{config['dataset']['name']}_{config['dataset']['class']}" / \
        config['model']['architecture'] / run_id
    base_path.mkdir(parents=True, exist_ok=True)

    config_path = base_path / \
        f"STFPM_Config_{config['model']['architecture']}.yaml"
    with config_path.open('w') as f:
        yaml.dump(config, f)

    summary_path = base_path / 'summary_metric.json'
    with summary_path.open('w') as f:
        json.dump(summary_metric, f, indent=4)

    model_path = base_path / 'quantized_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Quantized model and artifacts saved to: {base_path}")
