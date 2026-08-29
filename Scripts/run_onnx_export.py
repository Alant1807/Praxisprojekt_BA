import os
import shutil
import yaml
from pathlib import Path

from Scripts.utils import *
from Scripts.stfpm_arch import *
from Scripts.onnx_exporter import ONNX_Exporter
from Scripts.results_aggregator import *


def _find_and_collect_path(
    root_dir: Path, search_pattern: str, collection_dict: dict, dict_key: str, file_description: str, uuid: str
):
    """Sucht nach einer Datei und speichert sie im Dictionary. Bricht bei Fehlen nicht ab.

    Args:
        root_dir (Path): Das zu durchsuchende Hauptverzeichnis.
        search_pattern (str): Das Suchmuster (z.B. "*.json").
        collection_dict (dict): Das Dictionary zum Speichern des gefundenen Pfads.
        dict_key (str): Der Schlüssel für das Dictionary.
        file_description (str): Beschreibung der Datei für die Konsolenausgabe.
        uuid (str): Die eindeutige ID des Trainingslaufs.
    """
    try:
        found_path = next(root_dir.glob(search_pattern))
        collection_dict[dict_key] = found_path
        print(f"  -> {file_description} gefunden: {found_path}")
    except StopIteration:
        print(f"WARNUNG: {file_description} für UUID '{uuid}' nicht gefunden.")


def export_all_models_to_onnx(
    training_dir: str, inference_dir: str, output_base_dir: str = "exported_onnx_models"
):
    """Fasst Trainings- und Inferenz-Ergebnisse zusammen und exportiert die Modelle als ONNX.

    Args:
        training_dir (str): Pfad zu den Trainingsergebnissen.
        inference_dir (str): Pfad zu den Inferenz-Ergebnissen.
        output_base_dir (str, optional): Zielordner für die ONNX-Modelle. Standard ist "exported_onnx_models".
    """
    root_dir = Path(training_dir)
    inference_root_dir = Path(inference_dir)
    output_dir = Path(output_base_dir)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Exportiere Modelle nach: '{output_dir.resolve()}'")

    processed_model_variants = set()

    for dataset_dir in root_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        print(f"\n=================================================")
        print(f"Verarbeite Basis-Verzeichnis: {dataset_dir.name}")
        print(f"=================================================")

        for model_variant_dir in dataset_dir.iterdir():
            if not model_variant_dir.is_dir():
                continue

            processed_model_variants.add(str(model_variant_dir))

            for run_dir in model_variant_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                if run_dir.name.startswith(".") or run_dir.name == ".ipynb_checkpoints":
                    continue

                print(f"\n--- Verarbeite Trainingslauf: {run_dir.name} ---")

                collected_paths = {}

                search_patterns = [
                    "*.yaml",
                    "*_best_student.pth",
                    "*_final_student.pth",
                    "*_results.csv",
                    "summary_metrics.json",
                    "*auroc*.png",
                    "*loss*.png",
                    "trace_epoch*",
                ]

                all_files = []
                for pattern in search_patterns:
                    all_files.extend(run_dir.rglob(pattern))

                for file_path in all_files:
                    name = file_path.name

                    if name.endswith("_best_student.pth"):
                        collected_paths["best_weights_pth"] = file_path
                    elif name.endswith("_final_student.pth"):
                        collected_paths["final_weights_pth"] = file_path

                    elif file_path.suffix == ".yaml" and "onnx_model_config" not in name:
                        # Bevorzuge die YAML-Datei im Hauptverzeichnis des Laufs, um Konflikte zu vermeiden
                        if file_path.parent == run_dir:
                            collected_paths["yaml"] = file_path
                        elif "yaml" not in collected_paths:
                            collected_paths["yaml"] = file_path

                    elif name.startswith("trace_epoch"):
                        collected_paths["trace_epoch"] = file_path
                    elif name.endswith("_results.csv"):
                        collected_paths["results_csv"] = file_path
                    elif name == "summary_metrics.json":
                        collected_paths["metrics_json"] = file_path
                    elif "auroc" in name and file_path.suffix == ".png":
                        collected_paths["auroc_png"] = file_path
                    elif "loss" in name and file_path.suffix == ".png":
                        collected_paths["loss_png"] = file_path

                if "yaml" not in collected_paths or (
                    "best_weights_pth" not in collected_paths
                    and "final_weights_pth" not in collected_paths
                ):
                    print(
                        f"WARNUNG: Wichtige Dateien (YAML/PTH) in '{run_dir.name}' nicht gefunden. Überspringe."
                    )
                    if "yaml" not in collected_paths:
                        print("  -> Fehlend: Trainings-YAML")
                    if (
                        "best_weights_pth" not in collected_paths
                        and "final_weights_pth" not in collected_paths
                    ):
                        print("  -> Fehlend: _best_student.pth ODER _final_student.pth")
                    continue

                try:
                    config = load_config(str(collected_paths["yaml"]))
                except Exception as e:
                    print(f"FEHLER beim Laden der YAML {collected_paths['yaml']}: {e}. Überspringe.")
                    continue

                is_asymmetric = config.get("model_settings", {}).get("is_asymmetric", False)

                if is_asymmetric:
                    modellname = f"teacher-{config['teacher_model']['architecture']}_student-{config['student_model']['architecture']}"
                else:
                    modellname = config["model"]["architecture"]

                is_mvtec = config["dataset"]["name"] == "MVTecAD"
                if is_mvtec:
                    dataset_identifier = f"{config['dataset']['name']}_{config['dataset']['class']}"
                else:
                    dataset_identifier = config["dataset"]["name"]

                klasse = dataset_identifier
                run_id = run_dir.name
                uuid = run_id

                id_dir = output_dir / klasse / modellname / run_id

                # Überspringe bereits exportierte Modelle
                if id_dir.exists():
                    print(f"Zielverzeichnis für '{run_dir.name}' existiert schon. Überspringe.")
                    continue

                # Lade die zugehörigen Inferenz-Ergebnisse anhand der UUID
                _find_and_collect_path(
                    inference_root_dir,
                    f"**/*{uuid}*/**/inference_summary.json",
                    collected_paths,
                    "inference_json",
                    "Inferenz-Zusammenfassung",
                    uuid,
                )

                if is_asymmetric:
                    inference_modellname = f"teacher-{config['teacher_model']['architecture']}_student-{config['student_model']['architecture']}"
                else:
                    inference_modellname = config["model"]["architecture"]

                _find_and_collect_path(
                    inference_root_dir,
                    f"**/*{uuid}*/**/{klasse}_{inference_modellname}_{uuid}_profiler_trace.json",
                    collected_paths,
                    "profiler_json",
                    "Profiler-Zusammenfassung",
                    uuid,
                )

                training_results_dir = id_dir / "training_results"
                plots_dir = training_results_dir / "plots"
                weights_dir = training_results_dir / "weights"

                os.makedirs(plots_dir, exist_ok=True)
                os.makedirs(weights_dir, exist_ok=True)
                print(f"  -> Zielverzeichnis erstellt: {id_dir}")

                try:
                    model_pth = collected_paths.get("best_weights_pth") or collected_paths.get("final_weights_pth")
                    weight_suffix = "student"

                    onnx_exporter = ONNX_Exporter(
                        best_model_path=str(model_pth),
                        collected_paths=collected_paths,
                    )

                    onnx_model_filename = f"{modellname}_{run_id}_exported_onnx_model.onnx"
                    onnx_config_filename = f"{modellname}_{run_id}_onnx_model_config.yaml"

                    onnx_model_path = id_dir / onnx_model_filename
                    onnx_config_path = id_dir / onnx_config_filename

                    # Führt den Export aus
                    onnx_config = onnx_exporter.export_onnx(str(onnx_model_path))

                    with open(onnx_config_path, "w") as f:
                        yaml.dump(onnx_config, f, sort_keys=False, indent=4)

                    print(f"  -> ONNX-Config erfolgreich gespeichert.")

                except Exception as e:
                    print(f"FEHLER beim Exportieren von {run_dir.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                try:
                    shutil.copy2(collected_paths["yaml"], training_results_dir)

                    if "metrics_json" in collected_paths:
                        shutil.copy2(collected_paths["metrics_json"], training_results_dir)
                    if "results_csv" in collected_paths:
                        shutil.copy2(collected_paths["results_csv"], training_results_dir)
                    if "trace_epoch" in collected_paths:
                        shutil.copy2(collected_paths["trace_epoch"], training_results_dir)

                    if "inference_json" in collected_paths:
                        shutil.copy2(
                            collected_paths["inference_json"],
                            id_dir / "inference_summary.json",
                        )
                    if "profiler_json" in collected_paths:
                        shutil.copy2(
                            collected_paths["profiler_json"],
                            id_dir / f"{klasse}_{inference_modellname}_{uuid}_profiler_trace.json",
                        )

                    if "auroc_png" in collected_paths:
                        shutil.copy2(collected_paths["auroc_png"], plots_dir / "auroc.png")
                    if "loss_png" in collected_paths:
                        shutil.copy2(collected_paths["loss_png"], plots_dir / "loss.png")

                    if "best_weight_pth" in collected_paths:
                        shutil.copy2(
                            collected_paths["best_weight_pth"],
                            weights_dir / f"{modellname}_{klasse}_{uuid}_best_{weight_suffix}.pth",
                        )
                    elif "final_weight_pth" in collected_paths:
                        shutil.copy2(
                            collected_paths["final_weight_pth"],
                            weights_dir / f"{modellname}_{klasse}_{uuid}_final_{weight_suffix}.pth",
                        )

                    print(f"  -> Trainings-Artefakte erfolgreich kopiert.")

                except Exception as e:
                    print(f"FEHLER beim Kopieren der Artefakte für {run_dir.name}: {e}")

    print("\nExport aller Modelle abgeschlossen! Starte Zusammenführung der CSV-Zusammenfassungen.")
    print("Die exportierten Modelle befinden sich in:", output_dir)

    for dataset_dir in output_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for model_variant_dir in dataset_dir.iterdir():
            if not model_variant_dir.is_dir():
                continue

            variant_name = model_variant_dir.name
            dataset_name = dataset_dir.name

            original_training_variant_path = root_dir / dataset_name / variant_name
            original_inference_variant_path = inference_root_dir / dataset_name / variant_name

            if not original_training_variant_path.exists() or not original_inference_variant_path.exists():
                print(f"Warnung: Originalpfade für '{variant_name}' nicht gefunden. Überspringe Merge.")
                continue

            training_summary_path = original_training_variant_path / f"{variant_name}_model_variant_summary.csv"
            inference_summary_path = original_inference_variant_path / f"{variant_name}_model_variant_inference_summary.csv"

            target_save_path = model_variant_dir

            if training_summary_path.exists() and inference_summary_path.exists():
                try:
                    merge_and_save_variant_summaries(
                        str(target_save_path),
                        str(training_summary_path),
                        str(inference_summary_path),
                    )
                    print(f"Zusammenfassung für '{variant_name}' erfolgreich erstellt.")
                except Exception as e:
                    print(f"Fehler beim Zusammenführen der Summaries für '{variant_name}': {e}")
            else:
                print(f"Warnung: Eine oder beide CSV-Dateien für '{variant_name}' nicht gefunden. Überspringe Merge.")
                if not training_summary_path.exists():
                    print(f"  -> Fehlend: {training_summary_path}")
                if not inference_summary_path.exists():
                    print(f"  -> Fehlend: {inference_summary_path}")

    print("\n--- Skript beendet. ---")