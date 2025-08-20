import os
from dotenv import load_dotenv
import pandas as pd
from supabase import create_client, Client
from pathlib import Path
import json


def load_env():
    script_dir = Path(__file__).parent
    dotenv_path = script_dir / '.env'

    load_dotenv(dotenv_path=dotenv_path)

    global SUPABASE_URL, SUPABASE_KEY, TRAINING_ROOT_DIRECTORY, INFERENCE_ROOT_DIRECTORY
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    TRAINING_ROOT_DIRECTORY = script_dir.parent / "Training_Runs"
    INFERENCE_ROOT_DIRECTORY = script_dir.parent / "Inference_Runs"

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            f"Fehler: SUPABASE_URL und SUPABASE_KEY konnten nicht aus {dotenv_path} geladen werden.")


def find_summary_csvs(class_dir_path):
    """Findet alle *_model_variant_summary.csv Dateien in einem spezifischen Klassen-Ordner."""
    return list(Path(class_dir_path).rglob("*_model_variant_summary.csv"))


def find_inference_summaries(root_path):
    return list(Path(root_path).rglob("inference_summary.json"))


def upload_summary_to_supabase(supabase: Client, csv_path):
    """Liest eine CSV-Datei und lädt ihren Inhalt in die Supabase-Tabelle hoch."""
    print(f"Verarbeite Datei: {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path)

        data_to_upload = df.to_dict('records')

        if not data_to_upload:
            print(f"Datei {csv_path.name} ist leer. Überspringe...")
            return

        supabase.table('model_summaries').upsert(data_to_upload).execute()

        print(f"{len(data_to_upload)} Zeile(n) erfolgreich hochgeladen.")

    except Exception as e:
        print(f"Fehler beim Verarbeiten von {csv_path.name}: {e}")


def upload_inference_summary_to_supabase(supabase: Client, json_path):
    print(f"Verarbeite Inferenz-Zusammenfassung: {json_path.name}...")
    try:
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        data_to_upload = {
            'training_id': json_data.get('model_used', {}).get('training_id'),
            'dataset_name': json_data.get('dataset_name'),
            'model_architecture': json_data.get('model_used', {}).get('model_architecture'),
            'student_model_weights_path': json_data.get('model_used', {}).get('student_model_weights_path'),
            'auroc_score': json_data.get('performance_metrics', {}).get('auroc_score'),
            'inference_time': json_data.get('performance_metrics', {}).get('inference_time')
        }

        supabase.table('inference_summaries').upsert(data_to_upload).execute()
        print(f"{len(data_to_upload)} Zeile(n) erfolgreich hochgeladen.")
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {json_path.name}: {e}")


def inference_to_db():
    print("Starte Supabase Inference Uploader...")
    load_env()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    root = Path(INFERENCE_ROOT_DIRECTORY)
    if not root.is_dir():
        print(
            f"Fehler: Das Verzeichnis '{INFERENCE_ROOT_DIRECTORY}' wurde nicht gefunden.")
        return

    json_paths = find_inference_summaries(root)
    if not json_paths:
        print(f"Keine Inferenz-JSONs in '{root.name}' gefunden.")
        return

    for path in json_paths:
        upload_inference_summary_to_supabase(supabase, path)

    print("\nProzess beendet.")


def metrics_to_db():
    print("Starte Supabase Summary Uploader (für mehrere Klassen)...")
    load_env()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    root = Path(TRAINING_ROOT_DIRECTORY)
    if not root.is_dir():
        print(
            f"Fehler: Das Hauptverzeichnis '{TRAINING_ROOT_DIRECTORY}' wurde nicht gefunden.")
        return

    summary_csv_paths = find_summary_csvs(root)

    if not summary_csv_paths:
        print(
            f"Keine Zusammenfassungs-CSV in '{root.name}' gefunden.")
        return

    for path in summary_csv_paths:
        upload_summary_to_supabase(supabase, path)

    print("\nProzess beendet.")
