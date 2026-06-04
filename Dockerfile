# Image de base légère
FROM python:3.12-slim

# Éviter la mise en cache des fichiers .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système requises par OpenCV et les libs graphiques
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Création du dossier de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ /app/src/

# Copie du modèle de production (Assurez-vous qu'il est bien à la racine du build context)
COPY exp-14.pt /app/exp-14.pt

# Variable d'environnement pour le chemin du modèle
ENV MODEL_PATH=/app/exp-14.pt

# Exposition du port FastAPI
EXPOSE 8000

# Lancement de l'application via Uvicorn
CMD ["uvicorn", "src.serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
