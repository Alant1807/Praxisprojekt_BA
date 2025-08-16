import copy
import yaml
import os
import json

import torch
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping
from Scripts.model import STFPM
from Scripts.dataset import MVTecDataset
from torch.utils.data import DataLoader


def quantize_model(model_weights_path, config, summary_metric):
    """
    Führt die Quantisierung im FX Graph Mode durch.
    """
    print("--- Starte Quantisierungsprozess im FX Graph Mode (CPU) ---")

    print("Lade FP32-Modell...")
    cpu_device = torch.device("cpu")
    stfpm_model = STFPM(
        architecture=config['model']['architecture'],
        layers=config['model']['layers']
    )
    stfpm_model.student_model.load_state_dict(
        torch.load(model_weights_path, map_location=cpu_device)
    )
    stfpm_model.eval()
    student_model = stfpm_model.student_model.to(cpu_device)

    print("Bereite das Modell für die Quantisierung vor (prepare_fx)...")
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    example_inputs = (torch.randn(1, 3, 256, 256),)

    prepared_model = prepare_fx(student_model, qconfig_mapping, example_inputs)

    # 3. Kalibrierung
    print("Starte Kalibrierung...")
    train_set = MVTecDataset(
        img_size=config['dataset']['img_size'],
        base_path=config['dataset']['base_path'],
        cls=config['dataset']['class'],
        mode='train',
        subfolders=['good']
    )
    calibration_loader = DataLoader(
        train_set,
        batch_size=16, 
        shuffle=True
    )
    stfpm_model.stem_model.eval()
    with torch.no_grad():
        for i, (images, _, _, _) in enumerate(calibration_loader):
            if i >= 20:  
                break
            stem_output = stfpm_model.stem_model(images.to(cpu_device))
            prepared_model(stem_output)

    print("Kalibrierung abgeschlossen.")

    print("Konvertiere Modell...")
    quantized_model = convert_fx(prepared_model)

    print("Modell erfolgreich quantisiert.")

    save_quantized_model(quantized_model, config, summary_metric)

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
