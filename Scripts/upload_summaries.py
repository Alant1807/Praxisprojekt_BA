import os
from dotenv import load_dotenv
import pandas as pd
from supabase import create_client, Client
from pathlib import Path

def load_env():
    script_dir = Path(__file__).parent
    dotenv_path = script_dir / '.env'
    
    load_dotenv(dotenv_path=dotenv_path)

    global SUPABASE_URL, SUPABASE_KEY, ROOT_DIRECTORY
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    ROOT_DIRECTORY = script_dir.parent / "Training_Runs"

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(f"Fehler: SUPABASE_URL und SUPABASE_KEY konnten nicht aus {dotenv_path} geladen werden.")

def find_summary_csvs(class_dir_path):
    """Findet alle *_model_variant_summary.csv Dateien in einem spezifischen Klassen-Ordner."""
    return list(Path(class_dir_path).rglob("*_model_variant_summary.csv"))

def upload_summary_to_supabase(supabase: Client, csv_path):
    """Liest eine CSV-Datei und lädt ihren Inhalt in die Supabase-Tabelle hoch."""
    print(f"Verarbeite Datei: {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path)

        data_to_upload = df.to_dict('records')

        if not data_to_upload:
            print(f"Datei {csv_path.name} ist leer. Überspringe...")
            return

        data, count = supabase.table(
            'model_summaries').upsert(data_to_upload).execute()

        print(f"{len(data_to_upload)} Zeile(n) erfolgreich hochgeladen.")

    except Exception as e:
        print(f"Fehler beim Verarbeiten von {csv_path.name}: {e}")


def metrics_to_db():
    print("Starte Supabase Summary Uploader (für mehrere Klassen)...")
    load_env()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    root = Path(ROOT_DIRECTORY)
    if not root.is_dir():
        print(
            f"Fehler: Das Hauptverzeichnis '{ROOT_DIRECTORY}' wurde nicht gefunden.")
    else:
        for class_dir in root.iterdir():
            if not class_dir.is_dir():
                continue

            print(f"\n--- Durchsuche Klasse '{class_dir.name}' ---")

            summary_csv_paths = find_summary_csvs(class_dir)

            if not summary_csv_paths:
                print(
                    f"Keine Zusammenfassungs-CSV in '{class_dir.name}' gefunden.")
                continue

            for path in summary_csv_paths:
                upload_summary_to_supabase(supabase, path)

    print("\nProzess beendet.")