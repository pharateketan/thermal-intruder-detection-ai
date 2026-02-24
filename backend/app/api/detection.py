from fastapi import APIRouter, UploadFile, File, Query, HTTPException

from app.core.config import settings
from app.core.model_manager import ModelManager
from app.models.schemas import DetectionResponse
from app.utils.image_utils import validate_and_load_image
from app.utils.inference import run_inference

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(default=settings.DEFAULT_CONF, ge=0.01, le=0.99),
    iou:  float = Query(default=settings.DEFAULT_IOU,  ge=0.01, le=0.99),
    max_det:  int = Query(default=settings.DEFAULT_MAX_DET,  ge=1, le=1000),
    img_size: int = Query(default=settings.DEFAULT_IMG_SIZE, ge=320, le=1280),
):
    if not ModelManager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    image = await validate_and_load_image(file)

    try:
        return run_inference(
            image=image, conf=conf, iou=iou,
            max_det=max_det, img_size=img_size,
            model=ModelManager.get(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
