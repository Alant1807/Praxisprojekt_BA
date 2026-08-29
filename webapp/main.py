import sys
import os
from streamlit.web import cli as stcli

def main():
    """
    Einstiegspunkt für die Standalone-Anwendung.
    
    Dieses Skript fungiert als Wrapper, um die Streamlit-Webanwendung wie ein 
    klassisches Desktop-Programm starten zu können. Es abstrahiert den 
    Befehlszeilenaufruf ('streamlit run app.py') und ermöglicht die Paketierung 
    als .exe Datei mittels PyInstaller.
    """
    # 1. Dynamische Pfad-Auflösung
    # Dies ist kritisch für die Funktionsfähigkeit als gepackte .exe Datei.
    if getattr(sys, 'frozen', False):
        # Fall A: Anwendung läuft als PyInstaller-Bundle.
        # PyInstaller entpackt temporäre Dateien in das Verzeichnis, auf das sys._MEIPASS zeigt.
        base_dir = sys._MEIPASS
    else:
        # Fall B: Anwendung läuft als normales Python-Skript (Entwicklung).
        # Wir nutzen den absoluten Pfad dieses Skripts als Referenz.
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Konstruktion des Pfades zur eigentlichen App-Logik
    app_path = os.path.join(base_dir, "app.py")

    # 2. Konfiguration der Server-Parameter
    # Erhöhung des Upload-Limits ist notwendig, da ONNX-Modelle und 
    # industrielle Bilddatensätze das Standardlimit (200MB) oft überschreiten.
    upload_limit_mb = 1000

    # 3. Simulation des CLI-Aufrufs
    # Anstatt einen Subprozess zu starten (was in gefrorenen Anwendungen instabil sein kann),
    # manipulieren wir sys.argv direkt. Das Streamlit-Framework "denkt" dadurch, 
    # es wäre über die Kommandozeile mit diesen Argumenten gestartet worden.
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",  # Deaktiviert Dev-Features (z.B. Watcher)
        "--server.maxUploadSize", str(upload_limit_mb),
        "--browser.gatherUsageStats=false" # Telemetrie deaktivieren (Datenschutz)
    ]

    # 4. Übergabe an Streamlit
    # Startet den Streamlit-Webserver im aktuellen Prozess.
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()