FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "main.ipynb", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]