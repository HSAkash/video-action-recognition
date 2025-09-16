# main.py
import base64
from pathlib import Path
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .ai import Classifier

app = FastAPI()
classifier = Classifier()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

def decode_base64_jpeg_to_bgr(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode frame")
    return img

# 1) Define the WebSocket FIRST so it takes precedence
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            b64 = await ws.receive_text()
            frame = decode_base64_jpeg_to_bgr(b64)
            label = classifier.predict(frame)
            await ws.send_json({"label": label})
    except WebSocketDisconnect:
        pass

# 2) Serve index.html at "/"
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

# 3) Mount static files under /static (not "/")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
