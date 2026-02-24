import io
from PIL import Image, ImageOps
from fastapi import HTTPException, UploadFile
from app.core.config import settings


async def validate_and_load_image(file: UploadFile) -> Image.Image:
    if file.content_type not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    data = await file.read()
    if len(data) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.MAX_UPLOAD_SIZE_MB} MB."
        )
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")
    return image
