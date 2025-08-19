FROM python:3.11-slim

# NEU: Installiere die notwendigen Build-Werkzeuge UND die fehlende Grafikbibliothek
RUN apt-get update && apt-get install -y build-essential libgl1 libglib2.0-0

WORKDIR /app

COPY ./docker/docker_jupyterNotebook/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./Images ./Images/
COPY ./Scripts ./Scripts/
COPY ./Configs ./Configs/
COPY ./main.ipynb .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--notebook-dir=/app"]