"""
Point d'entrée de l'API de serving.
Gère les interfaces de communication (HTTP/WS).
"""

import asyncio
import base64
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from src.serving.api.inference import InferenceService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

inference_service = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global inference_service
    inference_service = InferenceService()
    yield
    del inference_service


app = FastAPI(title="SHARP Real-time API", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Vérification de l'état du serveur."""
    return {"status": "ok", "model_loaded": inference_service.model is not None}


@app.websocket("/ws/video")
async def video_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            _, encoded = data.split(",", 1) if "," in data else (None, data)
            img_bytes = base64.b64decode(encoded)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            if img is not None:
                # Inférence CPU déportée dans un thread pour ne pas bloquer
                # la boucle asyncio (préserve le débit du WebSocket).
                result = await asyncio.to_thread(inference_service.predict, img)
                await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("Client déconnecté")
    except Exception as e:
        logger.error(f"Erreur WS : {e}")


app.mount("/", StaticFiles(directory="src/serving/frontend", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
