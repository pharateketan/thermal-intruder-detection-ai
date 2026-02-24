from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.model_manager import ModelManager
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if ModelManager.is_loaded else "model_loading",
        model_loaded=ModelManager.is_loaded,
        model_path=settings.MODEL_PATH,
    )
