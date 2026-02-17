# Usa un'immagine leggera ma completa
FROM python:3.12-slim

# Imposta variabili d'ambiente per Python (evita file .pyc e forza log in tempo reale)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installa le dipendenze di sistema (necessarie per compilare alcune librerie RAG)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e installa i requisiti (facciamo questo PRIMA di copiare il codice per sfruttare la cache di Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice dell'app
COPY . .

# Esponi la porta del tuo sistema RAG (es. 8001 come nel tuo compose)
EXPOSE 8001

# Avvia l'applicazione (sostituisci main.py con il tuo file principal
CMD ["python", "core/main.py"]