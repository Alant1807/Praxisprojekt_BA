import pandas as pd
import os
import json
import glob


def load_df(target_filename):
    """
    Lädt ein DataFrame aus einer CSV-Datei. Wenn die Datei nicht existiert oder leer ist,
    wird ein leeres DataFrame zurückgegeben.

    Args:
        target_filename (str): Der Pfad zur CSV-Datei, die geladen werden soll.

    Returns:
        pd.DataFrame: Ein DataFrame mit den geladenen Daten oder ein leeres DataFrame,
                      wenn die Datei nicht existiert oder leer ist.
    """

    if not os.path.exists(target_filename):
        print(
            f"Datei '{target_filename}' existiert nicht. Ein neues DataFrame wird erstellt.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(target_filename)
        print(f"DataFrame erfolgreich geladen aus: {target_filename}")
        return df
    except pd.errors.EmptyDataError:
        print(
            f"Die Datei '{target_filename}' ist leer. Ein leeres DataFrame wird zurückgegeben.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Fehler beim Laden der Datei '{target_filename}': {e}")
        return pd.DataFrame()


def save_results(result_df, target_filename):
    """
    Speichert die Ergebnisse in einer CSV-Datei. Wenn die Datei bereits existiert,
    wird sie überschrieben.

    Args:
        result_df (pd.DataFrame): Das DataFrame, das gespeichert werden soll.
        target_filename (str): Der Pfad zur Zieldatei, in der die Ergebnisse gespeichert werden sollen.
    """

    try:
        result_df.to_csv(target_filename, index=False)
        print(f"\nErgebnisse wurden gespeichert in: {target_filename}\n")
    except IOError as e:
        print(
            f"Fehler beim Speichern der Ergebnisse in '{target_filename}': {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")


def create_result_df(result_data: dict, target_filename):
    """ 
    Erstellt ein DataFrame aus den Ergebnissen und speichert es in einer CSV-Datei.

    Args:
        result_data (dict): Ein Dictionary mit den Ergebnissen
        target_filename (str): Der Pfad zur Zieldatei, in der die Ergebnisse gespeichert werden sollen.
    """

    result_df = pd.DataFrame([result_data], columns=result_data.keys())
    save_results(result_df, target_filename)


def create_model_variant_summary(model_variant_path):
    """
    Aggregiert die Ergebnisse aller Trainingsläufe innerhalb eines 
    <ModelArchitecture_Variant>-Verzeichnisses (z.B. .../STFPM_ResNet18/).

    Liest die 'summary_metrics.json' aus jedem Trainingslauf-Unterverzeichnis
    (z.B. .../STFPM_ResNet18/<training_id>/summary_metrics.json)
    und erstellt eine '_model_variant_summary.csv'-Datei im model_variant_path.

    Args:
        model_variant_path (str): Der Pfad zum <ModelArchitecture_Variant>-Verzeichnis.
                                  Beispiel: 'Training_Runs/MVTec_AD_grid/STFPM_ResNet18'
    """

    if not os.path.isdir(model_variant_path):
        print(
            f"Fehler: Modellvarianten-Verzeichnis '{model_variant_path}' nicht gefunden.")
        return

    all_run_summaries_data = list()

    training_run_dirs = [d for d in glob.glob(
        os.path.join(model_variant_path, '*')) if os.path.isdir(d)]

    if not training_run_dirs:
        print(
            f"Keine Trainingslauf-Unterverzeichnisse in '{model_variant_path}' gefunden.")
        return

    print(
        f"Verarbeite {len(training_run_dirs)} Trainingsläufe aus '{model_variant_path}'...")

    for run_dir in training_run_dirs:
        summary_file_path = os.path.join(run_dir, 'summary_metrics.json')

        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)

                flat_summary = {}
                flat_summary['training_id'] = summary_data.get('training_id')
                flat_summary['timestamp'] = summary_data.get('timestamp')
                flat_summary['dataset_name'] = summary_data.get('dataset_name')
                flat_summary['model_architecture'] = summary_data.get(
                    'model_architecture')
                flat_summary['used_memory_format'] = summary_data.get(
                    'used_memory_format')
                flat_summary['used_amp_mixed_precision'] = summary_data.get(
                    'used_amp_mixed_precision')

                key_params = summary_data.get('key_parameters', {})
                flat_summary['num_epochs'] = key_params.get('num_epochs')
                flat_summary['lernrate'] = key_params.get('lernrate')
                flat_summary['optimizer'] = key_params.get('optimizer'),
                flat_summary['scheduler'] = key_params.get('scheduler'),

                training_metrics = summary_data.get('training_summary', {})
                flat_summary['training_duration'] = training_metrics.get(
                    'training_duration')
                flat_summary['final_train_loss'] = training_metrics.get(
                    'final_train_loss')
                flat_summary['best_train_loss'] = training_metrics.get(
                    'best_train_loss')
                flat_summary['best_epoch'] = training_metrics.get('best_epoch')
                flat_summary['avg_epoch_duration'] = training_metrics.get(
                    'avg_epoch_duration')

                evaluation_metrics = summary_data.get(
                    'evaluation_performance', {})
                flat_summary['auroc_score'] = evaluation_metrics.get(
                    'auroc_score')
                flat_summary['best_auroc_score'] = evaluation_metrics.get(
                    'best_auroc_score')
                flat_summary['inference_time'] = evaluation_metrics.get(
                    'inference_time')

                all_run_summaries_data.append(flat_summary)

            except json.JSONDecodeError:
                print(
                    f"Warnung: Fehler beim Lesen der JSON-Datei: {summary_file_path}")
            except Exception as e:
                print(
                    f"Warnung: Unerwarteter Fehler beim Verarbeiten von {summary_file_path}: {e}")

    if not all_run_summaries_data:
        print("Keine gültigen 'summary_metrics.json'-Dateien zum Aggregieren gefunden.")
        return

    output_csv_path = os.path.join(
        model_variant_path, f"{os.path.basename(model_variant_path)}_model_variant_summary.csv")
    try:
        save_results(pd.DataFrame(all_run_summaries_data), output_csv_path)
        print(
            f"Modellvarianten-Zusammenfassung erfolgreich gespeichert: {output_csv_path}")
    except Exception as e:
        print(
            f"Fehler beim Speichern der Modellvarianten-Zusammenfassung '{output_csv_path}': {e}")


def get_results(target_filename):
    """
    Lädt die Ergebnisse aus einer CSV-Datei und gibt ein DataFrame zurück.

    Args:
        target_filename (str): Der Pfad zur CSV-Datei, die geladen werden soll.

    Returns:
        pd.DataFrame: Ein DataFrame mit den Ergebnissen, das aus der CSV-Datei geladen wurde.
    """

    result_df = load_df(target_filename)
    return result_df
