# main.py - Unser Launcher-Skript

import subprocess
import sys
from pathlib import Path


def main():
    # Finde den Pfad zum Basisverzeichnis (funktioniert für .py und .exe)
    if getattr(sys, 'frozen', False):
        # Wir laufen in einer PyInstaller-.exe-Datei
        base_dir = Path(sys._MEIPASS)
    else:
        # Wir laufen als normales Python-Skript
        base_dir = Path(__file__).parent

    # Pfad zur mitgelieferten app.py
    app_path = str(base_dir / 'app.py')

    # Pfad zur mitgelieferten streamlit.exe
    streamlit_path = str(base_dir / 'streamlit.exe')

    # Baue den Befehl zusammen, um die App zu starten
    command = [streamlit_path, 'run', app_path]

    # Führe den Befehl aus
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Starten von Streamlit: {e}")
        # Optional: Halte das Fenster offen, um den Fehler zu sehen
        input("Drücken Sie Enter, um das Fenster zu schließen...")
    except FileNotFoundError:
        print(f"Fehler: streamlit.exe oder app.py nicht im Verzeichnis gefunden.")
        input("Drücken Sie Enter, um das Fenster zu schließen...")


if __name__ == '__main__':
    main()
