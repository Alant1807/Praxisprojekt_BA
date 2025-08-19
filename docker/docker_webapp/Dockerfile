# docker/docker_webapp/Dockerfile

FROM python:3.11-slim
WORKDIR /app

# Kopiere die requirements.txt aus dem richtigen Ordner
COPY ./docker/docker_webapp/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Kopiere NUR den Ordner mit der Streamlit-App
COPY ./webapp ./webapp

EXPOSE 8501
CMD ["streamlit", "run", "webapp/app.py"]