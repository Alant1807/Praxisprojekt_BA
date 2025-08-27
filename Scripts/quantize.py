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
# -----------------


def quantize_model(model_weights_path, config, summary_metric):
    """
    Hauptfunktion zur Quantisierung des Studenten-Modells.
    Führt eine "Post-Training Static Quantization" durch.

    Args:
        model_weights_path (str): Pfad zu den trainierten Gewichten des Studenten-Modells.
        config (dict): Konfigurationsparameter für das Modell und die Quantisierung.
        summary_metric (str): Metrik zur Bewertung der Quantisierung.

    Returns:
        quantized_model (torch.nn.Module): Das quantisierte Studenten-Modell.
    """
    device = torch.device('cpu')
    print("Lade Modelle für die Quantisierung...")

    # Schritt 1: Lade ein temporäres FP32 (Gleitkomma) STFPM-Modell.
    # Dies ist notwendig, um an den originalen, nicht-quantisierten "stem_model" zu gelangen.
    # Der Stem bleibt während der Inferenz in FP32, um die Eingabebilder zu verarbeiten.
    with torch.no_grad():
        temp_float_model = STFPM(
            architecture=config['model']['architecture'],
            layers=config['model']['layers'],
            quantize=False,  # Wichtig: Explizit die Nicht-quantisierte Version laden
        ).to(device).eval()
        float_stem_model = temp_float_model.stem_model

    # Schritt 2: Erstelle die Architektur des Studenten-Modells, das quantisiert werden soll.
    # Es wird die `FeatureExtractor`-Klasse verwendet, die das eigentliche Backbone-Netzwerk enthält.
    student_to_quantize = FeatureExtractor(
        backbone=config['model']['architecture'],
        pretrained=False,  # Die Gewichte werden separat geladen
        layers=config['model']['layers'],
        quantize=True,  # Wichtig: Die quantisierbare Architektur-Variante wird hier erstellt
        requires_grad=True
    )

    # Schritt 3: "Modell-Chirurgie" (Model Surgery)
    # Da der 'stem' separat ausgeführt wird, müssen die entsprechenden Eingangs-Layer
    # im Studenten-Modell durch einen 'Identity'-Layer ersetzt werden.
    # Ein Identity-Layer gibt seine Eingabe unverändert weiter.
    if 'mobilenet' in config['model']['architecture']:
        student_to_quantize.model.features[0] = nn.Identity()
    elif 'resnet' in config['model']['architecture'] or 'shufflenet' in config['model']['architecture']:
        setattr(student_to_quantize.model, 'conv1', nn.Identity())
        setattr(student_to_quantize.model, 'bn1', nn.Identity())
        setattr(student_to_quantize.model, 'relu', nn.Identity())
        setattr(student_to_quantize.model, 'maxpool', nn.Identity())

    # Schritt 4: Lade die trainierten Gewichte des Studenten-Modells.
    student_to_quantize.load_state_dict(
        torch.load(model_weights_path, map_location=device)
    )
    student_to_quantize.to(device).eval()

    # Schritt 5: Vorbereitung der Quantisierung
    # Wähle die Quantisierungs-Konfiguration. "fbgemm" ist der Standard-Backend für x86-CPUs.
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")

    # Erstelle einen Beispiel-Input, den `prepare_fx` benötigt, um den Graphen des Modells zu "tracen" (verfolgen).
    # Der Dummy-Input muss die gleiche Form haben wie die Daten, die das zu quantisierende Modell erwartet.
    # Hier ist das der Output des `float_stem_model`.
    dummy_input = torch.randn(
        1, 3, config['dataset']['img_size'], config['dataset']['img_size'])
    example_inputs = (float_stem_model(dummy_input),)

    print("Bereite das Studenten-Modell für die Quantisierung vor...")
    # `prepare_fx` fügt "Observer" in das Modell ein. Diese Observer sammeln während der Kalibrierung
    # Statistiken (min/max Werte) über die Aktivierungen, um die optimalen Quantisierungsparameter zu bestimmen.
    model_prepared = quantize_fx.prepare_fx(
        student_to_quantize, qconfig_mapping, example_inputs
    )

    # Schritt 6: Kalibrierung
    # Erstelle den DataLoader mit repräsentativen Daten (hier "gute" Bilder).
    calibration_loader = setup_calibration_loader(config)
    # Führe die Kalibrierung durch, indem die Daten durch das vorbereitete Modell geleitet werden.
    calibrate_model(model_prepared, calibration_loader, float_stem_model)

    # Schritt 7: Konvertierung
    print("Konvertiere zum quantisierten Modell...")
    # `convert_fx` entfernt die Observer und ersetzt die Gleitkomma-Module (z.B. Conv2d)
    # durch ihre quantisierten Äquivalente (z.B. QuantizedConv2d).
    quantized_student_model = quantize_fx.convert_fx(model_prepared)

    # Schritt 8: Speichern
    # Speichere das fertig quantisierte Modell und die dazugehörigen Artefakte.
    save_artifacts(quantized_student_model, config, summary_metric)


def setup_calibration_loader(config):
    """
    Bereitet den DataLoader für die Kalibrierung vor.
    Die Kalibrierung sollte mit Daten erfolgen, die die typische Verteilung der Eingabedaten widerspiegeln.

    Args:
        config (dict): Konfigurationsparameter für den Datensatz und die Kalibrierung.

    Returns:
        DataLoader: Ein DataLoader mit den Kalibrierungsdaten.
    """
    print("Erstelle den Kalibrierungs-Daten-Lader...")
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
    """
    Führt die Kalibrierung des vorbereiteten Modells durch.
    Dabei werden Daten durch das Netzwerk geschickt, damit die Observer Statistiken sammeln können.

    Args:
        model (torch.nn.Module): Das vorbereitete Modell, das kalibriert werden soll.
        calibration_loader (DataLoader): DataLoader mit den Kalibrierungsdaten.
        stem_model (torch.nn.Module): Das FP32-Stem-Modell.
    """
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
