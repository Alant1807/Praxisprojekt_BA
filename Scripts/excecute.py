import os
import yaml
from Scripts.dataset import *
from torch.utils.data import DataLoader
from Scripts.model import *
from Scripts.trainer import *
from Scripts.results_manager import *
from Scripts.asymmetric_model import *


def load_config(config_path):
    """
    Lädt eine YAML-Konfigurationsdatei.

    Args:
        config_path (str): Der Pfad zur YAML-Konfigurationsdatei.

    Returns:
        dict: Die Konfiguration als Dictionary, oder None bei Fehlern.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(
            f"Fehler: Konfigurationsdatei nicht gefunden unter {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Fehler beim Parsen der YAML-Datei {config_path}: {e}")
        return None


def load_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Fehler: JSON-Datei nicht gefunden unter {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Fehler beim Parsen der JSON-Datei {json_path}: {e}")
        return None


def training_selected_class(config_path, selected_class):
    trained_model_variant_paths = set()
    for model_folder in os.listdir(config_path):
        model_path = os.path.join(config_path, model_folder)
        if os.path.isdir(model_path):
            class_folder_path = os.path.join(model_path, selected_class)
            if os.path.isdir(class_folder_path):
                for config_file in os.listdir(class_folder_path):
                    if config_file.endswith('.yaml'):
                        full_config_path = os.path.join(
                            class_folder_path, config_file)

                        print(
                            f"\n--- Verarbeite Konfiguration: {full_config_path} ---")
                        config = load_config(full_config_path)

                        if config is None:
                            print(
                                f"Fehler beim Laden der Konfiguration {full_config_path}. Überspringe.")
                            continue

                        if config['dataset']['class'] != selected_class:
                            print(
                                f"Überspringe {full_config_path}, da die Klasse nicht '{selected_class}' ist.")
                            continue

                        training_set = MVTecDataset(
                            img_size=config['dataset']['img_size'],
                            base_path=config['dataset']['base_path'],
                            cls=config['dataset']['class'],
                            mode='train',
                            subfolders=['good']
                        )

                        test_set = MVTecDataset(
                            img_size=config['dataset']['img_size'],
                            base_path=config['dataset']['base_path'],
                            cls=config['dataset']['class'],
                            mode='test',
                        )

                        train_loader = DataLoader(
                            training_set,
                            batch_size=config['dataloader']['batch_size'],
                            shuffle=True
                        )

                        test_loader = DataLoader(
                            test_set,
                            batch_size=config['dataloader']['batch_size'],
                            shuffle=False
                        )
                        if config.get('model', {}).get('asymmetric', False):
                            print("Verwende asymmetrisches Modell.")
                            model = AsymmetricSTFPM(
                                teacher_architecture=config['model']['teacher_architecture'],
                                student_architecture=config['model']['student_architecture'],
                                layers=config['model']['layers']
                            )
                        else:
                            model = STFPM(
                                architecture=config['model']['architecture'],
                                layers=config['model']['layers']
                            )

                        trainer = Trainer(
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            config=config
                        )

                        print(
                            f"Starte Training für Konfiguration: {config_file}...")
                        trainer.train()

                        print(
                            f"Training für Konfiguration {config_file} abgeschlossen.")

                        model_variant_path = os.path.join(
                            trainer.train_folder_dir,
                            f"{config['dataset']['name']}_{config['dataset']['class']}",
                            config['model']['architecture']
                        )
                        trained_model_variant_paths.add(model_variant_path)

    print("\n--- Alle Trainingsläufe abgeschlossen. Starte Aggregation der Modellvarianten-Zusammenfassungen. ---")
    if not trained_model_variant_paths:
        print("Keine Modellvarianten zum Aggregieren gefunden.")
    else:
        for variant_path in trained_model_variant_paths:
            print(f"Aggregiere Ergebnisse für Modellvariante: {variant_path}")
            try:
                create_model_variant_summary(variant_path)
            except Exception as e:
                print(
                    f"Fehler beim Erstellen der Zusammenfassung für '{variant_path}': {e}")

    print("\n--- Skript beendet. ---")


def training_all_classes(config_path):
    trained_model_variant_paths = set()
    # selected_class = 'grid'

    for model_folder in os.listdir(config_path):
        model_path = os.path.join(config_path, model_folder)
        if os.path.isdir(model_path):
            for class_name in os.listdir(model_path):
                class_folder_path = os.path.join(model_path, class_name)
                if os.path.isdir(class_folder_path):
                    for config_file in os.listdir(class_folder_path):
                        if config_file.endswith('.yaml'):
                            full_config_path = os.path.join(
                                class_folder_path, config_file)

                            print(
                                f"\n--- Verarbeite Konfiguration: {full_config_path} ---")
                            config = load_config(full_config_path)

                            if config is None:
                                print(
                                    f"Fehler beim Laden der Konfiguration {full_config_path}. Überspringe.")
                                continue

                            if config['dataset']['class'] != class_name:
                                print(
                                    f"Überspringe {full_config_path}, da die Klasse '{config['dataset']['class']}' nicht mit dem Ordnernamen '{class_name}' übereinstimmt.")
                                continue

                            try:
                                training_set = MVTecDataset(
                                    img_size=config['dataset']['img_size'],
                                    base_path=config['dataset']['base_path'],
                                    cls=config['dataset']['class'],
                                    mode='train',
                                    subfolders=['good']
                                )

                                test_set = MVTecDataset(
                                    img_size=config['dataset']['img_size'],
                                    base_path=config['dataset']['base_path'],
                                    cls=config['dataset']['class'],
                                    mode='test',
                                    subfolders=['good']
                                )

                                train_loader = DataLoader(
                                    training_set,
                                    batch_size=config['dataloader']['batch_size'],
                                    shuffle=True
                                )

                                test_loader = DataLoader(
                                    test_set,
                                    batch_size=config['dataloader']['batch_size'],
                                    shuffle=False
                                )
                            except Exception as e:
                                print(
                                    f"Fehler beim Erstellen von Dataset/DataLoader für {full_config_path}: {e}")
                                continue

                            model = STFPM(
                                architecture=config['model']['architecture'],
                                layers=config['model']['layers']
                            )

                            trainer = Trainer(
                                model=model,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                config=config
                            )

                            print(
                                f"Starte Training für Konfiguration: {config_file}...")
                            trainer.train()

                            print(
                                f"Training für Konfiguration {config_file} abgeschlossen.")

                            model_variant_path = os.path.join(
                                trainer.train_folder_dir,
                                f"{config['dataset']['name']}_{config['dataset']['class']}",
                                config['model']['architecture']
                            )
                            trained_model_variant_paths.add(model_variant_path)

    print("\n--- Alle Trainingsläufe abgeschlossen. Starte Aggregation der Modellvarianten-Zusammenfassungen. ---")

    if not trained_model_variant_paths:
        print("Keine Modellvarianten zum Aggregieren gefunden.")
    else:
        for variant_path in trained_model_variant_paths:
            print(f"Aggregiere Ergebnisse für Modellvariante: {variant_path}")
            try:
                create_model_variant_summary(variant_path)
            except Exception as e:
                print(
                    f"Fehler beim Erstellen der Zusammenfassung für '{variant_path}': {e}")

    print("\n--- Skript beendet. ---")


def inference_model(training_run_folder, inference_output_dir):
    for dirpath, dirnames, filenames in os.walk(training_run_folder):
        if ".ipynb_checkpoints" in dirpath:
            continue

        yaml_filename = None
        for file in filenames:
            if file.endswith('.yaml'):
                yaml_filename = file
                break

        if yaml_filename is None:
            continue

        json_path = os.path.join(dirpath, "summary_metrics.json")
        yaml_path = os.path.join(dirpath, yaml_filename)

        with open(json_path, 'r') as f:
            summary_data = json.load(f)
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)

        training_id = summary_data.get('training_id')
        if not training_id:
            print(
                f"Skipping directory {dirpath}: Keine training_id in summary_metrics.json gefunden.")
            continue

        expected_inference_summary_path = os.path.join(
            inference_output_dir,
            f"{config_data['dataset']['name']}_{config_data['dataset']['class']}",
            config_data['model']['architecture'],
            training_id,
            "inference_summary.json"
        )

        if os.path.exists(expected_inference_summary_path):
            print(
                f"Inferenz für Trainingslauf {training_id} bereits vorhanden. Überspringe.")
            continue

        try:
            test_set = MVTecDataset(
                img_size=config_data['dataset']['img_size'],
                base_path=config_data['dataset']['base_path'],
                cls=config_data['dataset']['class'],
                mode='test',
                download_if_missing=False
            )
            test_loader = DataLoader(
                test_set,
                batch_size=config_data['dataloader']['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        except Exception as e:
            print(
                f"Fehler beim Laden des Test-Datensatzes für {yaml_path}: {e}")
            continue

        model = STFPM(
            architecture=config_data['model']['architecture'],
            layers=config_data['model']['layers']
        )

        inference = Inference(
            model,
            test_loader,
            config_data,
            path_to_student_weight=summary_data.get(
                "Pfad_der_Gewichte", {}).get("Pfad_bester_Gewichte"),
            trainings_id=summary_data.get('training_id')
        )

        print(f"Starte Inferenz für Konfiguration: {yaml_path}...")
        auroc_score, total_inference_time = inference.evaluate_loaded_model()
        inference.create_inference_summary(
            summary_data, auroc_score, total_inference_time)
        print(
            f"Inferenz für Konfiguration {yaml_path} abgeschlossen. \nAUROC: {auroc_score}, Inferenzzeit: {total_inference_time} Sekunden.")

        inference.generate_heatmaps_from_saved_maps()

    print("\n--- Alle Inferenzläufe abgeschlossen. ---")
