# Verwende ein offizielles Python-Image als Basis
FROM python:3.10-slim

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die Abhängigkeitsdateien
COPY requirements.txt .

# Installiere die Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den Rest des Anwendungscodes
COPY . .

# Gib den Port an, auf dem die App laufen wird
EXPOSE 8501

# Befehl zum Starten der App
CMD ["streamlit", "run", "webapp/app.py"]