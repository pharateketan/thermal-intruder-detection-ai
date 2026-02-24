from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import detection, health
from app.core.config import settings
from app.core.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[BOOT] Loading model from: {settings.MODEL_PATH}")
    ModelManager.load(settings.MODEL_PATH)
    yield
    print("[SHUTDOWN] Releasing model.")
    ModelManager.release()


app = FastAPI(title="ThermalEye API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,    prefix="/api", tags=["Health"])
app.include_router(detection.router, prefix="/api", tags=["Detection"])


@app.get("/")
def root():
    return {"service": "ThermalEye Detection API", "docs": "/docs"}
