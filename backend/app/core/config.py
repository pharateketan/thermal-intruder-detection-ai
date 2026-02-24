from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    MODEL_PATH: str = "app/models/flir_adas_model.pt"

    DEFAULT_CONF: float = 0.45
    DEFAULT_IOU: float = 0.50
    DEFAULT_MAX_DET: int = 100
    DEFAULT_IMG_SIZE: int = 640

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    MAX_UPLOAD_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: List[str] = [
        "image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"
    ]

    CLASS_NAMES: dict = {
        0: "Person", 1: "Bicycle", 2: "Vehicle"
    }
    CLASS_TYPES: dict = {
        0: "person", 1: "bicycle", 2: "vehicle"
    }
    CLASS_COLORS: dict = {
        0: "#ff2d55", 1: "#00e5ff", 2: "#ffd60a"
    }

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
