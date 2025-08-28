import torch
import torchvision.models as models
import copy
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.export import export
from Scripts.model2 import *
import json
import yaml
import os
from pathlib import Path

import torch._inductor.config as config
config.cpp_wrapper = True

DEVICE = torch.device('cpu')

def quantize_model_pt2e(best_weights_path, config, summary_metric):
    model = STFPM(
        architecture=config['model']['architecture'],
        layers=config['model']['layers'],
        quantize=False
    ).to(DEVICE).eval()
    stem_model = model.stem_model
    student_to_quantize = model.student_model
    student_to_quantize.load_state_dict(
        torch.load(best_weights_path, map_location=DEVICE)
    )
    student_to_quantize.to(DEVICE).eval()
    example_inputs = (stem_model(torch.randn(
        1, 3, 256, 256).contiguous(memory_format=torch.channels_last)),)
    with torch.no_grad():
        exported_model = export(
            student_to_quantize,
            example_inputs
        )
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)
    converted_model = convert_pt2e(prepared_model)
    with torch.no_grad():
        optimized_model = torch.compile(converted_model)
        # Running some benchmark
        optimized_model(*example_inputs)

    save_artifacts(optimized_model, config, summary_metric)


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
