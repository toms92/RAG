FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installa dipendenze di sistema necessarie per compilare le librerie (es. chroma-hnswlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia e installa le dipendenze del progetto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installa JupyterLab
RUN pip install --no-cache-dir jupyterlab

EXPOSE 8888

# Avvia JupyterLab senza token (solo per sviluppo locale)
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--notebook-dir=/home/jovyan/work", \
     "--NotebookApp.token=", \
     "--allow-root"]