"""
Backend de serving pour le projet SHARP.
Utilise FastAPI et WebSockets pour l'inférence YOLO en temps réel.
"""

import base64
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from src.config import settings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Stockage global du modèle
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application (Chargement du modèle)."""
    global model
    model_path = settings.PRODUCTION_MODEL_PATH

    if not model_path.exists():
        logger.error(f"Modèle de production introuvable à {model_path}")
        # On ne bloque pas forcément le démarrage pour permettre le debug,
        # mais les WS échoueront.
    else:
        logger.info(f"Chargement du modèle YOLO depuis {model_path}...")
        model = YOLO(model_path)
        logger.info("Modèle chargé avec succès.")

    yield
    # Nettoyage si nécessaire
    del model


app = FastAPI(title="SHARP API", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Vérifie que l'API est en ligne et que le modèle est chargé."""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "model_version": "exp-14 (V3)",
    }


@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    """
    Reçoit des frames vidéo du frontend et renvoie les détections.
    Format attendu : Chaine Base64.
    """
    if model is None:
        logger.error("WebSocket refusé : modèle non chargé.")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("Client connecté au flux vidéo WebSocket.")

    try:
        while True:
            # 1. Réception des données (Image Base64)
            data = await websocket.receive_text()

            # Décodage Base64 -> Image OpenCV
            try:
                header, encoded = data.split(",", 1) if "," in data else (None, data)
                img_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    continue
            except Exception as e:
                logger.warning(f"Erreur de décodage d'image : {e}")
                continue

            # 2. Inférence YOLO
            # stream=True est plus efficace pour la vidéo
            results = model(img, conf=0.5, verbose=False)

            # 3. Logique métier : Compteur de doigts
            detections = []
            total_fingers = 0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Extraction des données
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    # Accumulation (Somme des classes = total doigts)
                    total_fingers += cls

                    detections.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": round(conf, 2),
                            "class": cls,
                            "label": f"{cls}_doigts",
                        }
                    )

            # 4. Réponse JSON légère
            await websocket.send_json(
                {"boxes": detections, "total_fingers": total_fingers}
            )

    except WebSocketDisconnect:
        logger.info("Client déconnecté.")
    except Exception as e:
        logger.error(f"Erreur inattendue dans le WebSocket : {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# Montage du dossier frontend
app.mount(
    "/", StaticFiles(directory="src/serving/frontend", html=True), name="frontend"
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
