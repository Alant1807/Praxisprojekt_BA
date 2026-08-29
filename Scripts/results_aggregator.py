import pandas as pd
import os
import json
import glob


def load_df(target_filename: str) -> pd.DataFrame:
    """Lädt eine CSV-Datei als Pandas DataFrame.

    Args:
        target_filename (str): Pfad zur CSV-Datei.

    Returns:
        pd.DataFrame: Das geladene DataFrame oder ein leeres DataFrame im Fehlerfall.
    """
    if not os.path.exists(target_filename):
        print(
            f"Datei '{target_filename}' existiert nicht. Ein neues DataFrame wird erstellt."
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(target_filename)
        print(f"DataFrame erfolgreich geladen aus: {target_filename}")
        return df
    except pd.errors.EmptyDataError:
        # Fängt leere Dateien (0 Byte) ab, die z.B. bei Systemabstürzen entstehen
        print(
            f"Die Datei '{target_filename}' ist leer. Ein leeres DataFrame wird zurückgegeben."
        )
        return pd.DataFrame()
    except Exception as e:
        print(f"Fehler beim Laden der Datei '{target_filename}': {e}")
        return pd.DataFrame()


def save_results(result_df: pd.DataFrame, target_filename: str):
    """Speichert ein DataFrame als CSV-Datei.

    Args:
        result_df (pd.DataFrame): Die zu speichernden Daten.
        target_filename (str): Zielpfad für die CSV-Datei.
    """
    try:
        # index=False verhindert das Speichern der Zeilennummern als eigene Spalte
        result_df.to_csv(target_filename, index=False)
        print(f"\nErgebnisse wurden gespeichert in: {target_filename}\n")
    except IOError as e:
        print(
            f"Fehler beim Speichern der Ergebnisse in '{target_filename}': {e}"
        )
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")


def create_result_df(result_data: dict, target_filename: str):
    """Erstellt ein DataFrame aus einem Dictionary und speichert es ab.

    Args:
        result_data (dict): Die Datenzeile als Dictionary.
        target_filename (str): Zielpfad für die CSV-Datei.
    """
    result_df = pd.DataFrame([result_data], columns=result_data.keys())
    save_results(result_df, target_filename)


def merge_and_save_variant_summaries(
    target_save_path: str, training_summary_path: str, inference_summary_path: str
):
    """Verbindet Trainings- und Inferenzmetriken in einer gemeinsamen Datei.

    Ermöglicht spätere Analysen, z.B. den Vergleich von Trainingszeit und AUROC-Score.

    Args:
        target_save_path (str): Ordner, in dem die zusammengeführte CSV gespeichert wird.
        training_summary_path (str): Pfad zur CSV mit den Trainingsdaten.
        inference_summary_path (str): Pfad zur CSV mit den Inferenzdaten.
    """
    print(
        f"Versuche, Zusammenfassungen für '{os.path.basename(target_save_path)}' zusammenzuführen..."
    )

    train_df = load_df(training_summary_path)
    inference_df = load_df(inference_summary_path)

    if train_df.empty or inference_df.empty:
        print(
            "Warnung: Eine oder beide Zusammenfassungs-Dateien sind leer. Zusammenführung wird übersprungen."
        )
        return

    # Inner Join behält nur Experimente, die erfolgreich trainiert UND evaluiert wurden
    merged_df = pd.merge(train_df, inference_df, on="training_id", how="inner")

    os.makedirs(target_save_path, exist_ok=True)

    output_csv_path = os.path.join(
        target_save_path,
        f"{os.path.basename(target_save_path)}_model_variant_summary.csv",
    )

    save_results(merged_df, output_csv_path)
    print(
        f"Kombinierte Zusammenfassung erfolgreich gespeichert: {output_csv_path}"
    )


def create_model_variant_summary(model_variant_path: str):
    """Fasst alle einzelnen Trainingsläufe einer Modellvariante in einer Tabelle zusammen.

    Wandelt die verschachtelte Ordnerstruktur in ein leicht analysierbares CSV-Format um.

    Args:
        model_variant_path (str): Pfad zum Hauptordner der Modellvariante (enthält die einzelnen Run-Ordner).
    """
    if not os.path.isdir(model_variant_path):
        print(
            f"Fehler: Modellvarianten-Verzeichnis '{model_variant_path}' nicht gefunden."
        )
        return

    all_run_summaries_data = list()

    training_run_dirs = [
        d
        for d in glob.glob(os.path.join(model_variant_path, "*"))
        if os.path.isdir(d)
    ]

    if not training_run_dirs:
        print(
            f"Keine Trainingslauf-Unterverzeichnisse in '{model_variant_path}' gefunden."
        )
        return

    print(
        f"Verarbeite {len(training_run_dirs)} Trainingsläufe aus '{model_variant_path}'..."
    )

    for run_dir in training_run_dirs:
        summary_file_path = os.path.join(run_dir, "summary_metrics.json")

        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)

                flat_summary = {}
                flat_summary["training_id"] = summary_data.get("training_id")
                flat_summary["timestamp"] = summary_data.get("timestamp")

                dataset = summary_data.get("dataset/dataloader", {})
                flat_summary["dataset_name"] = dataset.get("dataset_name")
                flat_summary["img_size"] = dataset.get("img_size")
                flat_summary["batch_size"] = dataset.get("batch_size")

                # Architektur-Infos formatieren (Symmetrisch vs. Asymmetrisch)
                if "teacher_architecture" in summary_data:
                    flat_summary["teacher_architecture"] = summary_data.get("teacher_architecture")
                    flat_summary["student_architecture"] = summary_data.get("student_architecture")
                    flat_summary["model_architecture"] = "N/A"
                else:
                    flat_summary["model_architecture"] = summary_data.get("model_architecture")
                    flat_summary["teacher_architecture"] = "N/A"
                    flat_summary["student_architecture"] = "N/A"

                training_param = summary_data.get("training_params", {})
                flat_summary["learning_rate"] = training_param.get("lr")
                flat_summary["epochs"] = training_param.get("epochs")

                training_metrics = summary_data.get("training_summary", {})
                flat_summary["training_duration"] = training_metrics.get("duration")
                flat_summary["avg_epoch_time"] = training_metrics.get("avg_epoch_duration")
                flat_summary["final_train_loss"] = training_metrics.get("final_loss")
                flat_summary["best_train_loss"] = training_metrics.get("best_loss")
                flat_summary["best_epoch"] = training_metrics.get("best_epoch")
                flat_summary["peak_ram_mb"] = training_metrics.get("peak_ram_mb")

                model_size_on_disk_metrics = summary_data.get("model_size_mb", {})
                flat_summary["final_model_mb"] = model_size_on_disk_metrics.get("final")
                flat_summary["best_model_mb"] = model_size_on_disk_metrics.get("best")

                all_run_summaries_data.append(flat_summary)

            except json.JSONDecodeError:
                print(
                    f"Warnung: Fehler beim Parsen der JSON-Datei (ungültiges Format): {summary_file_path}"
                )
            except Exception as e:
                print(
                    f"Warnung: Unerwarteter Fehler beim Verarbeiten von {summary_file_path}: {e}"
                )

    if not all_run_summaries_data:
        print(
            "Keine gültigen 'summary_metrics.json'-Dateien zum Aggregieren gefunden."
        )
        return

    output_csv_path = os.path.join(
        model_variant_path,
        f"{os.path.basename(model_variant_path)}_model_variant_summary.csv",
    )
    try:
        save_results(pd.DataFrame(all_run_summaries_data), output_csv_path)
        print(
            f"Modellvarianten-Zusammenfassung erfolgreich gespeichert: {output_csv_path}"
        )
    except Exception as e:
        print(
            f"Fehler beim Speichern der Modellvarianten-Zusammenfassung '{output_csv_path}': {e}"
        )


def create_model_variant_inference_summary(model_variant_path: str):
    """Fasst alle einzelnen Inferenzläufe einer Modellvariante in einer Tabelle zusammen.

    Wandelt die JSON-Daten (Erkennungsgüte, Performance) in eine CSV-Tabelle um.

    Args:
        model_variant_path (str): Pfad zum Hauptordner der Modellvariante.
    """
    if not os.path.isdir(model_variant_path):
        print(
            f"Fehler: Modellvarianten-Verzeichnis '{model_variant_path}' nicht gefunden."
        )
        return

    all_inference_run_summaries_data = list()

    inference_run_dirs = [
        d
        for d in glob.glob(os.path.join(model_variant_path, "*"))
        if os.path.isdir(d)
    ]

    if not inference_run_dirs:
        print(
            f"Keine Inferenzlauf-Unterverzeichnisse in '{model_variant_path}' gefunden."
        )
        return

    print(
        f"Verarbeite {len(inference_run_dirs)} Inferenzläufe aus '{model_variant_path}'..."
    )

    for inference_run_dir in inference_run_dirs:
        inference_summary_file = os.path.join(
            inference_run_dir, "inference_summary.json"
        )

        if os.path.exists(inference_summary_file):
            try:
                with open(inference_summary_file, "r", encoding="utf-8") as f:
                    inference_summary_data = json.load(f)

                flat_summary = {}

                # Die Training-ID ist notwendig, um diese Tabelle später mit den Trainingsdaten zu mergen
                model_used_metrics = inference_summary_data.get("model_used", {})
                flat_summary["training_id"] = model_used_metrics.get("training_id")

                performance_metrics = inference_summary_data.get("performance_metrics", {})
                flat_summary["auroc_score"] = performance_metrics.get("auroc_score")
                flat_summary["aupr_score"] = performance_metrics.get("aupr_score")
                flat_summary["optimal_threshold_youden_j"] = performance_metrics.get("optimal_threshold_youden_j")
                flat_summary["total_inference_time_sec"] = performance_metrics.get("total_inference_time_sec")
                flat_summary["avg_inference_time_per_image_ms"] = performance_metrics.get("avg_inference_time_per_image_ms")
                flat_summary["model_size_mb"] = performance_metrics.get("model_size_mb")
                flat_summary["peak_ram_mb"] = performance_metrics.get("peak_ram_mb")

                # Theoretische Modellkomplexität (hardwareunabhängig)
                model_complexity_metrics = inference_summary_data.get("model_complexity", {})
                flat_summary["gmacs"] = model_complexity_metrics.get("gmacs")
                flat_summary["gflops"] = model_complexity_metrics.get("gflops")
                flat_summary["mparams"] = model_complexity_metrics.get("mparams")

                all_inference_run_summaries_data.append(flat_summary)

            except json.JSONDecodeError:
                print(
                    f"Warnung: Fehler beim Parsen der JSON-Datei: {inference_summary_file}"
                )
            except Exception as e:
                print(
                    f"Warnung: Unerwarteter Fehler beim Verarbeiten von {inference_summary_file}: {e}"
                )

    if not all_inference_run_summaries_data:
        print(
            "Keine gültigen 'inference_summary.json'-Dateien zum Aggregieren gefunden."
        )
        return

    output_csv_path = os.path.join(
        model_variant_path,
        f"{os.path.basename(model_variant_path)}_model_variant_inference_summary.csv",
    )
    try:
        save_results(
            pd.DataFrame(all_inference_run_summaries_data), output_csv_path
        )
        print(
            f"Inferenzvarianten-Zusammenfassung erfolgreich gespeichert: {output_csv_path}"
        )
    except Exception as e:
        print(
            f"Fehler beim Speichern der Inferenzvarianten-Zusammenfassung '{output_csv_path}': {e}"
        )


def get_results(target_filename: str) -> pd.DataFrame:
    """Lädt die Ergebnisse aus einer Datei und gibt das DataFrame zurück.

    Args:
        target_filename (str): Dateipfad zur CSV.

    Returns:
        pd.DataFrame: Das resultierende DataFrame.
    """
    result_df = load_df(target_filename)
    return result_df