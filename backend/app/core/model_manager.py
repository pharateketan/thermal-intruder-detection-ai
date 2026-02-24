from ultralytics import YOLO
from pathlib import Path


class _ModelManager:
    def __init__(self):
        self._model = None
        self._path = ""

    def load(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path.resolve()}\n"
                "Place thermal_intruder_model.pt in the models/ folder "
                "or set MODEL_PATH in .env to the full absolute path."
            )
        self._model = YOLO(str(path))
        self._path = str(path)
        print(f"[OK] Model loaded: {path.name}")

    def get(self) -> YOLO:
        if self._model is None:
            raise RuntimeError("Model is not loaded.")
        return self._model

    def release(self) -> None:
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> str:
        return self._path


ModelManager = _ModelManager()
