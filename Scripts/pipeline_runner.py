import os
import sys
import yaml
import gc
import threading
import multiprocessing as mp
import json
import traceback
from queue import Empty as QueueEmpty
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from Scripts.dataset_mvtec import *
from Scripts.dataset_gkd import *
from Scripts.stfpm_arch import *
from Scripts.trainer import *
from Scripts.results_aggregator import *
from Scripts.utils import *
import queue


def create_training_optimized_dataloader(dataset, config, is_train=True):
    """Erstellt einen auf Geschwindigkeit optimierten DataLoader für das Training.

    Args:
        dataset (Dataset): Der PyTorch Datensatz.
        config (dict): Das Konfigurations-Wörterbuch.
        is_train (bool, optional): Ob der Lader für Training (True) oder Validierung (False) ist. Standard ist True.

    Returns:
        DataLoader: Der fertige PyTorch DataLoader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_gpu = device.type == "cuda"

    config_batch_size = config.get("dataloader", {}).get("batch_size")

    # Berechne optimale Batch-Größe, falls "auto" gewählt wurde
    if config_batch_size is None or config_batch_size == "auto":
        img_size = config.get("dataset", {}).get("img_size", 256)
        model_name = config.get("model", {}).get("architecture", "resnet18")
        batch_size = get_optimal_batch_size(device, img_size, model_name)
        print(f"Auto Batch-Size: {batch_size}")
    else:
        batch_size = config_batch_size

    config_num_workers = config.get("dataloader", {}).get("num_workers")

    if config_num_workers is None or config_num_workers == "auto":
        num_workers = get_optimal_num_workers(device)
        print(f"Auto Workers: {num_workers}")
    else:
        num_workers = min(config_num_workers, os.cpu_count())

    # pin_memory beschleunigt den Datentransfer vom RAM zur Grafikkarte
    pin_memory = config.get("dataloader", {}).get("pin_memory", True) and is_gpu
    persistent_workers = is_train and num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )


def create_optimized_inference_dataloader(dataset, config, load_to_ram=False):
    """Erstellt einen DataLoader speziell für die Inferenz.

    Args:
        dataset (Dataset): Der PyTorch Datensatz.
        config (dict): Das Konfigurations-Wörterbuch.
        load_to_ram (bool, optional): Ob der Datensatz komplett im RAM liegt. Standard ist False.

    Returns:
        DataLoader: Der fertige DataLoader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_gpu = device.type == "cuda"

    config_batch_size = config.get("dataloader", {}).get("batch_size")
    if config_batch_size is None or config_batch_size == "auto":
        img_size = config.get("dataset", {}).get("img_size", 256)
        model_name = config.get("model", {}).get("architecture", "resnet18")
        batch_size = get_optimal_batch_size(device, img_size, model_name)
    else:
        batch_size = config_batch_size

    # Wenn alles im RAM ist, brauchen wir keine Hintergrund-Worker für das Laden
    if load_to_ram:
        print("RAM Caching aktiv: Setze num_workers für Inferenz auf 0.")
        num_workers = 0
    else:
        config_num_workers = config.get("dataloader", {}).get("num_workers")
        if config_num_workers is None or config_num_workers == "auto":
            num_workers = get_optimal_num_workers(device)
        else:
            num_workers = min(config_num_workers, os.cpu_count())

    pin_memory = is_gpu

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,  
        prefetch_factor=2 if num_workers > 0 else None,
    )


# Markiert das Ende des Text-Streams vom Hintergrundprozess
_STREAM_SENTINEL = "___END_OF_OUTPUT___" 


class OutputRedirector:
    """Leitet Textausgaben (print) in eine Warteschlange um."""
    
    def __init__(self, message_queue):
        """
        Args:
            message_queue (queue.Queue): Die Warteschlange für die Textausgabe.
        """
        self._queue = message_queue

    def write(self, data):
        if data: 
            self._queue.put(data)

    def flush(self):
        pass  

    def isatty(self):
        return False

    def writable(self):
        return True


def process_wrapper(target_function, arguments, output_queue):
    """Verpackt die auszuführende Funktion, um ihre Ausgabe umzuleiten.

    Args:
        target_function (callable): Die auszuführende Trainings-/Inferenzfunktion.
        arguments (tuple): Argumente für die Funktion.
        output_queue (queue.Queue): Warteschlange für Textausgaben.
    """
    sys.stdout = OutputRedirector(output_queue)
    sys.stderr = OutputRedirector(output_queue)
    
    try:
        target_function(*arguments)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # Signalisiert dem Hauptprogramm, dass der Prozess fertig ist
        output_queue.put(_STREAM_SENTINEL)


def run_training_in_subprocess(target_function, arguments):
    """Führt eine Funktion in einem komplett eigenen Prozess aus, um RAM-Lecks zu vermeiden.

    Args:
        target_function (callable): Die Ziel-Funktion.
        arguments (tuple): Argumente für die Funktion.
    """
    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    
    process = ctx.Process(
        target=process_wrapper, 
        args=(target_function, arguments, output_queue)
    )
    process.start()

    def print_outputs_from_queue():
        """Liest Text aus der Warteschlange und druckt ihn in Echtzeit."""
        while True: 
            try:
                message = output_queue.get(timeout=0.5)
                
                if message == _STREAM_SENTINEL:
                    break
                    
                print(message, end="", flush=True) 
                
            except queue.Empty:
                if not process.is_alive():
                    # Restliche Nachrichten auslesen, falls der Prozess abgestürzt ist
                    while not output_queue.empty():
                        leftover = output_queue.get()
                        if leftover != _STREAM_SENTINEL:
                            print(leftover, end="", flush=True) 
                    break 

    printer_thread = threading.Thread(target=print_outputs_from_queue, daemon=True)
    printer_thread.start()

    process.join()
    printer_thread.join(timeout=5.0)

    if process.exitcode != 0:
        print(f"\n  WARNING: Sub-process for training ended with error code {process.exitcode}.")


def _run_single_stfpm_training(
    full_config_path, output_dir, use_cached_inputs
):
    """Führt ein einzelnes STFPM-Training aus.

    Args:
        full_config_path (str): Pfad zur YAML-Datei.
        output_dir (str): Zielordner für Modelle und Logs.
        use_cached_inputs (bool): Ob vorverarbeitete Tensoren genutzt werden.
    """
    print(f"\n--- Verarbeite Konfiguration: {full_config_path} ---")
    config = load_config(full_config_path)
    if config is None:
        print(f"Fehler beim Laden der Konfiguration {full_config_path}. Ueberspringe.")
        return

    if not use_cached_inputs:
        full_train_set = GKDDataset(
            img_size=config["dataset"]["img_size"],
            data_path=config["dataset"]["base_path"],
            mode="train",
        )
        test_set = GKDDataset(
            img_size=config["dataset"]["img_size"],
            data_path=config["dataset"]["base_path"],
            mode="test",
        )
    else:
        full_train_set = GKDDatasetCached(
            data_path=config["dataset"]["base_path"],
            mode="train",
            enable_timing=config["dataset"]["enable_timing"],
        )
        test_set = GKDDatasetCached(
            data_path=config["dataset"]["base_path"],
            mode="test",
            enable_timing=config["dataset"]["enable_timing"],
        )

    # 20% der fehlerfreien Trainingsbilder werden nur für die Schwellenwert-Kalibrierung (Threshold) genutzt
    val_size = int(0.2 * len(full_train_set))
    train_size = len(full_train_set) - val_size
    training_set, _ = random_split(
        full_train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Trainings-Split: {train_size} Bilder fuer Training, {val_size} fuer Validierung zurueckgehalten.")

    print("  Erstelle optimierte DataLoader...")
    train_loader = create_training_optimized_dataloader(
        training_set, config, is_train=True
    )
    test_loader = create_training_optimized_dataloader(
        test_set, config, is_train=False
    )

    is_asymmetric = config.get("model_settings", {}).get("is_asymmetric", False)
    model_args = {"is_asymmetric": is_asymmetric}

    if is_asymmetric:
        model_args.update(
            {
                "teacher_architecture": config["teacher_model"]["architecture"],
                "teacher_layers": config["teacher_model"]["layer"],
                "student_architecture": config["student_model"]["architecture"],
                "student_layers": config["student_model"]["layer"],
                "projection_head_type": config["model_settings"].get("projection_head_type", "simple"),
            }
        )
    else:
        model_args.update(
            {
                "architecture": config["model"]["architecture"],
                "layers": config["model"]["layers"],
                "extract_stem": config["model_settings"]["shared_stem"],
                "partial_share_depth": config["model_settings"].get("partial_share_depth", 0),
            }
        )

    print("Lade STFPM Modell für Feature-basierte Distillation...")
    model = STFPM(**model_args)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        test_loader=test_loader,
        run_evaluation=False,
        train_folder_dir=output_dir,
    )
    
    config_file = os.path.basename(full_config_path)
    print(f"Starte Training für Konfiguration: {config_file}...")
    trainer.train(
        set_profiler=config["training"]["profiling"],
        detailed_timing=config["training"]["detailed_timing"],
    )
    print(f"Training für Konfiguration {config_file} abgeschlossen.")


def Training_GKD(
    base_config_path,
    output_dir=None,
    allowed_models=None,
    allowed_configs=None,
    use_cached_inputs=False,
):
    """Hauptschleife für das GKD-Training aller Konfigurationen.

    Args:
        base_config_path (str): Ordner mit den YAML-Konfigurationen.
        output_dir (str, optional): Zielordner für die Modelle.
        allowed_models (list, optional): Welche Modelle trainiert werden sollen.
        allowed_configs (list, optional): Welche spezifischen Configs erlaubt sind.
        use_cached_inputs (bool, optional): Ob der Tensoren-Cache genutzt wird.
    """
    print_system_info()
    trained_model_variant_paths = set()

    for model_dir_name in sorted(os.listdir(base_config_path)):
        model_dir_path = os.path.join(base_config_path, model_dir_name)

        if not os.path.isdir(model_dir_path):
            continue

        if allowed_models and model_dir_name not in allowed_models:
            continue

        for config_file in sorted(os.listdir(model_dir_path)):
            if not config_file.endswith(".yaml"):
                continue

            if allowed_configs and not any(c in config_file for c in allowed_configs):
                continue

            full_config_path = os.path.join(model_dir_path, config_file)

            run_training_in_subprocess(
                _run_single_stfpm_training,
                (full_config_path, output_dir, use_cached_inputs),
            )

            # Pfad zur Variante speichern, um später die Auswertungen zusammenzufassen
            config = load_config(full_config_path)
            if config is None:
                continue

            is_asymmetric = config.get("model_settings", {}).get("is_asymmetric", False)
            if is_asymmetric:
                model_folder = (
                    f"teacher-{config['teacher_model']['architecture']}_"
                    f"student-{config['student_model']['architecture']}"
                )
            else:
                model_folder = f"{config['model']['architecture']}"

            dataset_identifier = (
                f"{config['dataset']['name']}_{config['dataset']['class']}"
                if config["dataset"]["name"] == "MVTecAD"
                else config["dataset"]["name"]
            )

            model_variant_path = os.path.join(output_dir, dataset_identifier, model_folder)
            trained_model_variant_paths.add(model_variant_path)

    print("\n--- Alle Trainingsläufe abgeschlossen. Starte Aggregation der Zusammenfassungen. ---")
    if not trained_model_variant_paths:
        print("Keine Modellvarianten zum Aggregieren gefunden.")
    else:
        for variant_path in sorted(list(trained_model_variant_paths)):
            print(f"Aggregiere Ergebnisse für Modellvariante: {variant_path}")
            try:
                create_model_variant_summary(variant_path)
            except Exception as e:
                print(f"Fehler beim Erstellen der Zusammenfassung für '{variant_path}': {e}")

    print("\n--- Skript beendet. ---")


def training_selected_class_MVTeC(
    base_config_path,
    selected_classes=None,
    output_dir=None,
    allowed_models=None,
    allowed_configs=None,
    used_cached_inputs=False,
    pretrained_weights=None,
):
    """Trainiert spezifische MVTecAD Klassen.

    (Die Argumente entsprechen weitgehend der GKD-Variante).
    """
    print_system_info()
    trained_model_variant_paths = set()

    for selected_class in selected_classes:
        print(f"--- Starte Trainingsläufe für ausgewählte Klasse: '{selected_class}' ---")

        for model_dir_name in sorted(os.listdir(base_config_path)):
            model_dir_path = os.path.join(base_config_path, model_dir_name)

            if not os.path.isdir(model_dir_path):
                continue

            if allowed_models and model_dir_name not in allowed_models:
                continue

            class_dir_path = os.path.join(model_dir_path, selected_class)
            if not os.path.isdir(class_dir_path):
                continue 

            for config_file in sorted(os.listdir(class_dir_path)):
                if not config_file.endswith(".yaml"):
                    continue

                if allowed_configs and not any(c in config_file for c in allowed_configs):
                    continue

                full_config_path = os.path.join(class_dir_path, config_file)
                print(f"\n--- Verarbeite Konfiguration: {full_config_path} ---")

                config = load_config(full_config_path)
                if config is None:
                    print(f"Fehler beim Laden von {full_config_path}. Überspringe.")
                    continue

                if not used_cached_inputs:
                    training_set = MVTecDataset(
                        img_size=config["dataset"]["img_size"],
                        base_path=config["dataset"]["base_path"],
                        cls=config["dataset"]["class"], 
                        mode="train",
                    )
                    test_set = MVTecDataset(
                        img_size=config["dataset"]["img_size"],
                        base_path=config["dataset"]["base_path"],
                        cls=config["dataset"]["class"],
                        mode="test",
                    )
                else:
                    training_set = MVTecDatasetCached(
                        base_path=config["dataset"]["base_path"],
                        cls=config["dataset"]["class"],
                        mode="train",
                        enable_timing=config["dataset"]["enable_timing"],
                    )
                    test_set = MVTecDatasetCached(
                        base_path=config["dataset"]["base_path"],
                        cls=config["dataset"]["class"],
                        mode="test",
                        enable_timing=config["dataset"]["enable_timing"],
                    )

                print("  Erstelle optimierte DataLoader...")
                train_loader = create_training_optimized_dataloader(training_set, config, is_train=True)
                test_loader = create_training_optimized_dataloader(test_set, config, is_train=False)

                is_asymmetric = config.get("model_settings", {}).get("is_asymmetric", False)
                model_args = {"is_asymmetric": is_asymmetric}

                if is_asymmetric:
                    model_args.update(
                        {
                            "teacher_architecture": config["teacher_model"]["architecture"],
                            "teacher_layers": config["teacher_model"]["layer"],
                            "student_architecture": config["student_model"]["architecture"],
                            "student_layers": config["student_model"]["layer"],
                            "projection_head_type": config["model_settings"]["projection_head_type"],
                        }
                    )
                else:
                    model_args.update(
                        {
                            "architecture": config["model"]["architecture"],
                            "layers": config["model"]["layers"],
                            "extract_stem": config["model_settings"]["shared_stem"],
                            "partial_share_depth": config["model_settings"].get("partial_share_depth", 0),
                        }
                    )

                model = STFPM(**model_args)

                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    config=config,
                    test_loader=test_loader,
                    run_evaluation=False,
                    train_folder_dir=output_dir,
                )
                
                print(f"Starte Training für Konfiguration: {config_file}...")
                trainer.train(
                    set_profiler=config.get("training", {}).get("profiling", False),
                    detailed_timing=config.get("training", {}).get("detailed_timing", False),
                )
                print(f"Training für Konfiguration {config_file} abgeschlossen.")

                if is_asymmetric:
                    model_folder = (
                        f"teacher-{config['teacher_model']['architecture']}_"
                        f"student-{config['student_model']['architecture']}"
                    )
                else:
                    model_folder = f"{config['model']['architecture']}"

                dataset_identifier = f"{config['dataset']['name']}_{config['dataset']['class']}"

                variant_path_for_aggregation = os.path.join(output_dir, dataset_identifier, model_folder)
                trained_model_variant_paths.add(variant_path_for_aggregation)

    print("\n--- Alle Trainingsläufe für die Klasse abgeschlossen. Starte Aggregation. ---")
    if not trained_model_variant_paths:
        print("Keine Modellvarianten zum Aggregieren gefunden.")
    else:
        for variant_path in sorted(list(trained_model_variant_paths)):
            print(f"Aggregiere Ergebnisse für Modellvariante: {variant_path}")
            try:
                create_model_variant_summary(variant_path)
            except Exception as e:
                print(f"Fehler beim Erstellen der Zusammenfassung für '{variant_path}': {e}")
    print("\n--- Skript beendet. ---")


def training_all_classes_MVTeC(
    base_config_path,
    output_dir=None,
    allowed_models=None,
    allowed_configs=None,
    use_cached_inputs=False,
):
    """Trainiert alle MVTecAD Klassen. Funktioniert identisch zu training_selected_class_MVTeC."""
    print_system_info()
    trained_model_variant_paths = set()
    print(f"--- Starte Trainingsläufe für ALLE Klassen in: '{base_config_path}' ---")

    for model_dir_name in sorted(os.listdir(base_config_path)):
        model_dir_path = os.path.join(base_config_path, model_dir_name)
        if not os.path.isdir(model_dir_path):
            continue

        if allowed_models and model_dir_name not in allowed_models:
            continue

        for class_name in os.listdir(model_dir_path):
            class_dir_path = os.path.join(model_dir_path, class_name)
            if not os.path.isdir(class_dir_path):
                continue

            for config_file in os.listdir(class_dir_path):
                if not config_file.endswith(".yaml"):
                    continue

                if allowed_configs and config_file not in allowed_configs:
                    continue

                full_config_path = os.path.join(class_dir_path, config_file)
                print(f"\n--- Verarbeite Konfiguration: {full_config_path} ---")

                config = load_config(full_config_path)
                if config is None:
                    print(f"Fehler beim Laden von {full_config_path}. Überspringe.")
                    continue

                if not use_cached_inputs:
                    training_set = MVTecDataset(
                        img_size=config["dataset"]["img_size"],
                        base_path=config["dataset"]["base_path"],
                        cls=config["dataset"]["class"],
                        mode="train",
                    )
                    test_set = MVTecDataset(
                        img_size=config["dataset"]["img_size"],
                        base_path=config["dataset"]["base_path"],
                        cls=config["dataset"]["class"],
                        mode="test",
                    )
                else:
                    training_set = MVTecDatasetCached(
                        base_path=config["dataset"]["base_path"],
                        cls=config["dataset"]["class"],
                        mode="train",
                        enable_timing=config["dataset"]["enable_timing"],
                    )
                    test_set = MVTecDatasetCached(
                        base_path=config["dataset"]["base_path"],
                        cls=config["dataset"]["class"],
                        mode="test",
                        enable_timing=config["dataset"]["enable_timing"],
                    )

                print("  Erstelle optimierte DataLoader...")
                train_loader = create_training_optimized_dataloader(training_set, config, is_train=True)
                test_loader = create_training_optimized_dataloader(test_set, config, is_train=False)

                is_asymmetric = config.get("model_settings", {}).get("is_asymmetric", False)
                model_args = {"is_asymmetric": is_asymmetric}

                if is_asymmetric:
                    model_args.update(
                        {
                            "teacher_architecture": config["teacher_model"]["architecture"],
                            "teacher_layers": config["teacher_model"]["layer"],
                            "student_architecture": config["student_model"]["architecture"],
                            "student_layers": config["student_model"]["layer"],
                            "projection_head_type": config["model_settings"]["projection_head_type"],
                        }
                    )
                else:
                    model_args.update(
                        {
                            "architecture": config["model"]["architecture"],
                            "layers": config["model"]["layers"],
                            "extract_stem": config["model_settings"]["shared_stem"],
                            "partial_share_depth": config["model_settings"].get("partial_share_depth", 0),
                        }
                    )

                print("Verwende STFPM...")
                model = STFPM(**model_args)

                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    config=config,
                    test_loader=test_loader,
                    run_evaluation=False,
                    train_folder_dir=output_dir,
                )
                print(f"Starte Training für Konfiguration: {config_file}...")
                trainer.train(
                    set_profiler=config.get("training", {}).get("profiling", False),
                    detailed_timing=config.get("training", {}).get("detailed_timing", False),
                )
                print(f"Training für Konfiguration {config_file} abgeschlossen.")

                if is_asymmetric:
                    model_folder = (
                        f"teacher-{config['teacher_model']['architecture']}_"
                        f"student-{config['student_model']['architecture']}"
                    )
                else:
                    model_folder = f"{config['model']['architecture']}"
                dataset_identifier = f"{config['dataset']['name']}_{config['dataset']['class']}"

                variant_path_for_aggregation = os.path.join(output_dir, dataset_identifier, model_folder)
                trained_model_variant_paths.add(variant_path_for_aggregation)

    print("\n--- Alle Trainingsläufe abgeschlossen. Starte Aggregation. ---")
    if not trained_model_variant_paths:
        print("Keine Modellvarianten zum Aggregieren gefunden.")
    else:
        for variant_path in sorted(list(trained_model_variant_paths)):
            print(f"Aggregiere Ergebnisse für Modellvariante: {variant_path}")
            try:
                create_model_variant_summary(variant_path)
            except Exception as e:
                print(f"Fehler beim Erstellen der Zusammenfassung für '{variant_path}': {e}")
    print("\n--- Skript beendet. ---")


# =============================================================================
# INFERENZ-FUNKTIONEN
# =============================================================================


def inference_model_Mvtec(
    training_run_folder,
    inference_output_dir,
    allowed_models=None,
    use_cached_inputs=False,
    detailed_profiling=False,
):
    """Sucht nach trainierten MVTecAD Modellen und führt die Auswertung (Inferenz) durch.

    Args:
        training_run_folder (str): Quellordner mit den Trainingsergebnissen.
        inference_output_dir (str): Zielordner für die Inferenz-Ergebnisse.
        allowed_models (list, optional): Filter für spezifische Modelle.
        use_cached_inputs (bool, optional): Tensoren-Cache verwenden?
        detailed_profiling (bool, optional): PyTorch-Profiler aktivieren?
    """
    print_system_info()

    inference_model_variant_paths = set()
    print(f"--- Starte Inferenz für alle Modelle in: '{training_run_folder}' ---")

    for dirpath, dirnames, filenames in os.walk(training_run_folder):
        if ".ipynb_checkpoints" in dirpath or "summary_metrics.json" not in filenames:
            continue

        json_path = os.path.join(dirpath, "summary_metrics.json")
        yaml_path = next((os.path.join(dirpath, f) for f in filenames if f.endswith(".yaml")), None)

        if not yaml_path:
            print(f"Skipping directory {dirpath}: Keine .yaml Konfigurationsdatei gefunden.")
            continue

        try:
            with open(json_path, "r") as f:
                summary_data = json.load(f)
            with open(yaml_path, "r") as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            print(f"Fehler beim Lesen von JSON/YAML in {dirpath}: {e}. Überspringe.")
            continue

        training_id = summary_data.get("training_id")
        if not training_id:
            print(f"Skipping directory {dirpath}: Keine training_id in summary_metrics.json gefunden.")
            continue

        config_model_type = config_data.get("model_settings", {}).get("model_type", "stfpm")
        if config_model_type != "stfpm":
            print(f"Überspringe (falscher Modelltyp '{config_model_type}', erwartet 'stfpm'): {dirpath}")
            continue

        is_asymmetric = config_data.get("model_settings", {}).get("is_asymmetric", False)
        dataset_identifier = f"{config_data['dataset']['name']}_{config_data['dataset']['class']}"

        if is_asymmetric:
            model_folder = (
                 f"teacher-{config_data['teacher_model']['architecture']}_"
                f"student-{config_data['student_model']['architecture']}"
            )
        else:
            model_folder = config_data["model"]["architecture"]

        if allowed_models and model_folder not in allowed_models:
            continue

        expected_inference_summary_path = os.path.join(
            inference_output_dir, dataset_identifier, model_folder, training_id, "inference_summary.json",
        )

        if os.path.exists(expected_inference_summary_path):
            print(f"Inferenz für Trainingslauf {training_id} bereits vorhanden. Überspringe.")
            inference_model_variant_paths.add(os.path.dirname(os.path.dirname(expected_inference_summary_path)))
            continue

        if not use_cached_inputs:
            test_set = MVTecDataset(
                img_size=config_data["dataset"]["img_size"],
                base_path=config_data["dataset"]["base_path"],
                cls=config_data["dataset"]["class"],
                mode="test",
            )
            train_set = MVTecDataset(
                img_size=config_data["dataset"]["img_size"],
                base_path=config_data["dataset"]["base_path"],
                cls=config_data["dataset"]["class"],
                mode="train",
            )
        else:
            test_set = MVTecDatasetCached(
                base_path=config_data["dataset"]["base_path"],
                cls=config_data["dataset"]["class"],
                mode="test",
                enable_timing=config_data["dataset"]["enable_timing"],
            )
            train_set = MVTecDatasetCached(
                base_path=config_data["dataset"]["base_path"],
                cls=config_data["dataset"]["class"],
                mode="train",
                enable_timing=config_data["dataset"]["enable_timing"],
            )

        test_loader = create_optimized_inference_dataloader(test_set, config_data, load_to_ram=True)
        train_loader = create_optimized_inference_dataloader(train_set, config_data, load_to_ram=True)

        model_args = {"is_asymmetric": is_asymmetric}

        if is_asymmetric:
            model_args.update(
                {
                    "teacher_architecture": config_data["teacher_model"]["architecture"],
                    "teacher_layers": config_data["teacher_model"]["layer"],
                    "student_architecture": config_data["student_model"]["architecture"],
                    "student_layers": config_data["student_model"]["layer"],
                     "projection_head_type": config_data["model_settings"]["projection_head_type"],
                }
            )
        else:
            model_args.update(
                {
                    "architecture": config_data["model"]["architecture"],
                    "layers": config_data["model"]["layers"],
                    "extract_stem": config_data["model_settings"]["shared_stem"],
                    "partial_share_depth": config_data["model_settings"].get("partial_share_depth", 0),
                 }
            )

        model = STFPM(**model_args)
        weight_key = "best_student"

        inference = Inference(
            model,
            test_loader,
            config_data,
            inference_output_dir,
            path_to_student_weight=summary_data.get("weight_paths", {}).get(weight_key),
             trainings_id=training_id,
        )

        print(f"Starte Inferenz für Konfiguration: {yaml_path}...")

        test_labels, test_scores, total_inference_time = inference.evaluate(
            detailed_profiling=detailed_profiling
        )

        inference.create_inference_summary(
            summary_data, test_labels, test_scores, total_inference_time
        )

        if len(test_labels) > 0 and len(test_scores) > 0:
            auroc_score = roc_auc_score(test_labels, test_scores)
        else:
             auroc_score = 0.0
        print(f"Inferenz für {yaml_path} abgeschlossen. AUROC: {auroc_score:.4f}")

        model_variant_path = os.path.join(inference.output_dir, dataset_identifier, model_folder)
        inference_model_variant_paths.add(model_variant_path)

    print("\n--- Alle Inferenzläufe abgeschlossen. Starte Aggregation der Inferenz-Zusammenfassungen. ---")
    if not inference_model_variant_paths:
        print("Keine Modellvarianten zum Aggregieren gefunden.")
    else:
        for variant_path in sorted(list(inference_model_variant_paths)):
            print(f"Aggregiere Inferenz-Zusammenfassung für Modellvariante: {variant_path}")
            try:
                 create_model_variant_inference_summary(variant_path)
            except Exception as e:
                print(f"Fehler beim Erstellen der Inferenz-Zusammenfassung für '{variant_path}': {e}")
    print("\n--- Skript beendet. ---")


def _run_single_inference_GKD(
    yaml_path,
    json_path,
    inference_output_dir,
    use_cached_inputs,
    load_to_ram,
    detailed_profiling,
    measure_memory
):
    """Führt eine einzelne Inferenz (Auswertung) für ein GKD-Modell aus."""
    with open(json_path, "r") as f:
        summary_data = json.load(f)
    with open(yaml_path, "r") as f:
        config_data = yaml.safe_load(f)

    training_id = summary_data.get("training_id")
    is_asymmetric = config_data.get("model_settings", {}).get("is_asymmetric", False)
    base_path = config_data["dataset"]["base_path"]

    if not use_cached_inputs:
        test_set = GKDDataset(
             img_size=config_data["dataset"]["img_size"],
            data_path=base_path,
            mode="test",
        )
        full_train_set = GKDDataset(
            img_size=config_data["dataset"]["img_size"],
            data_path=base_path,
            mode="train",
         )
    else:
        test_set = GKDDatasetCached(
            data_path=base_path,
            mode="test",
            enable_timing=config_data["dataset"]["enable_timing"],
            load_to_ram=load_to_ram,
        )
        full_train_set = GKDDatasetCached(
            data_path=base_path,
            mode="train",
            enable_timing=config_data["dataset"]["enable_timing"],
            load_to_ram=load_to_ram,
        )

    # 20% Validierungssplit, um den idealen Fehler-Schwellenwert (Threshold) zu berechnen
    val_size = int(0.2 * len(full_train_set))
    train_size = len(full_train_set) - val_size
    _, val_set = random_split(
        full_train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Validierungssplit: {val_size} Bilder fuer Schwellenwertberechnung zurueckgehalten.")

    test_loader = create_optimized_inference_dataloader(test_set, config_data, load_to_ram=load_to_ram)
    val_loader = create_optimized_inference_dataloader(val_set, config_data, load_to_ram=load_to_ram)

    model_args = {"is_asymmetric": is_asymmetric}
    if is_asymmetric:
        model_args.update(
            {
                "teacher_architecture": config_data["teacher_model"]["architecture"],
                 "teacher_layers": config_data["teacher_model"]["layer"],
                "student_architecture": config_data["student_model"]["architecture"],
                "student_layers": config_data["student_model"]["layer"],
                "projection_head_type": config_data["model_settings"]["projection_head_type"],
            }
        )
    else:
        model_args.update(
            {
                "architecture": config_data["model"]["architecture"],
                 "layers": config_data["model"]["layers"],
                "extract_stem": config_data["model_settings"]["shared_stem"],
                "partial_share_depth": config_data["model_settings"].get("partial_share_depth", 0),
            }
         )

    model = STFPM(**model_args)
    weight_key = ("best_student")

    inference = Inference(
        model,
        test_loader,
        config_data,
        inference_output_dir,
        path_to_student_weight=summary_data.get("weight_paths", {}).get(weight_key),
        trainings_id=training_id,
    )

    print(f"Starte Inferenz für: {yaml_path}...")
    clean_threshold = inference.calculate_threshold_on_val_set(val_loader, fpr=0.05)
    inference.optimal_threshold = clean_threshold

    test_labels, test_scores, total_inference_time = inference.evaluate(
        detailed_profiling=detailed_profiling, measure_memory=measure_memory
    )

    inference.create_inference_summary(
        summary_data, test_labels, test_scores, total_inference_time
     )


def inference_model_GKD(
    training_run_folder,
    inference_output_dir,
    allowed_models=None,
    use_cached_inputs=False,
    load_to_ram=False,
    detailed_profiling=False,
    measure_memory=False,
):
    """Sucht trainierte GKD-Modelle und wertet sie in isolierten Prozessen aus.

    Args:
        training_run_folder (str): Quellordner mit den Trainingsergebnissen.
        inference_output_dir (str): Zielordner für die Inferenz-Ergebnisse.
        allowed_models (list, optional): Filter für spezifische Modelle.
        use_cached_inputs (bool, optional): Tensoren-Cache verwenden?
        load_to_ram (bool, optional): Datensatz komplett in RAM laden?
        detailed_profiling (bool, optional): Profiler aktivieren?
        measure_memory (bool, optional): Peak-RAM messen?
    """
    print_system_info()

    inference_model_variant_paths = set()
    print(f"--- Starte Inferenz für alle Modelle in: '{training_run_folder}' ---")

    for dirpath, dirnames, filenames in os.walk(training_run_folder):
        if ".ipynb_checkpoints" in dirpath or "summary_metrics.json" not in filenames:
            continue

        json_path = os.path.join(dirpath, "summary_metrics.json")
        yaml_path = next((os.path.join(dirpath, f) for f in filenames if f.endswith(".yaml")), None)

        if not yaml_path:
            continue

        try:
            with open(json_path, "r") as f:
                summary_data = json.load(f)
            with open(yaml_path, "r") as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            print(f"Fehler beim Lesen von JSON/YAML: {e}")
            continue

        training_id = summary_data.get("training_id")
        if not training_id:
            continue

        config_model_type = config_data.get("model_settings", {}).get("model_type", "stfpm")
        if config_model_type != "stfpm":
            continue

        is_asymmetric = config_data.get("model_settings", {}).get("is_asymmetric", False)
        dataset_identifier = f"{config_data['dataset']['name']}"

        model_folder = (
            f"teacher-{config_data['teacher_model']['architecture']}_"
            f"student-{config_data['student_model']['architecture']}"
            if is_asymmetric
            else config_data["model"]["architecture"]
        )

        if allowed_models and model_folder not in allowed_models:
            continue

        expected_inference_summary_path = os.path.join(
            inference_output_dir, dataset_identifier, model_folder, training_id, "inference_summary.json",
        )

        if os.path.exists(expected_inference_summary_path):
            print(f"Inferenz für {training_id} bereits vorhanden. Überspringe.")
            inference_model_variant_paths.add(
                os.path.dirname(os.path.dirname(expected_inference_summary_path))
            )
            continue

        run_training_in_subprocess(
            _run_single_inference_GKD,
            (
                yaml_path,
                json_path,
                inference_output_dir,
                use_cached_inputs,
                load_to_ram,
                detailed_profiling,
                measure_memory
            ),
        )

        model_variant_path = os.path.join(inference_output_dir, dataset_identifier, model_folder)
        inference_model_variant_paths.add(model_variant_path)

    print("\n--- Alle Inferenzläufe abgeschlossen. Starte Aggregation. ---")
    for variant_path in sorted(list(inference_model_variant_paths)):
        try:
            create_model_variant_inference_summary(variant_path)
        except Exception as e:
            print(f"Fehler bei Aggregation für '{variant_path}': {e}")
    print("\n--- Skript beendet. ---")