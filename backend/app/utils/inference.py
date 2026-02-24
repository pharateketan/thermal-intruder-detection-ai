import time
from PIL import Image
from ultralytics import YOLO

from app.core.config import settings
from app.models.schemas import Detection, BoundingBox, DetectionSummary, DetectionResponse


def run_inference(
    image: Image.Image,
    conf: float,
    iou: float,
    max_det: int,
    img_size: int,
    model: YOLO,
) -> DetectionResponse:
    img_w, img_h = image.size

    t0 = time.perf_counter()
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=img_size,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    detections = []
    persons = vehicles = bicycles = other = 0
    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        for idx, box in enumerate(result.boxes):
            cid = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            bw, bh = x2 - x1, y2 - y1

            class_name  = settings.CLASS_NAMES.get(cid, f"class_{cid}")
            class_type  = settings.CLASS_TYPES.get(cid, "other")
            color       = settings.CLASS_COLORS.get(cid, "#ffffff")

            detections.append(Detection(
                id=idx + 1,
                class_id=cid,
                class_name=class_name,
                class_type=class_type,
                color=color,
                confidence=round(score, 4),
                bbox=BoundingBox(
                    x1=round(x1, 2), y1=round(y1, 2),
                    x2=round(x2, 2), y2=round(y2, 2),
                    width=round(bw, 2), height=round(bh, 2),
                    x1_norm=round(x1 / img_w, 4), y1_norm=round(y1 / img_h, 4),
                    x2_norm=round(x2 / img_w, 4), y2_norm=round(y2 / img_h, 4),
                ),
            ))

            if class_type == "person":   persons += 1
            elif class_type == "vehicle": vehicles += 1
            elif class_type == "bicycle": bicycles += 1
            else:                         other += 1

    return DetectionResponse(
        success=True,
        image_width=img_w,
        image_height=img_h,
        inference_ms=round(inference_ms, 2),
        conf_threshold=conf,
        iou_threshold=iou,
        summary=DetectionSummary(
            total=len(detections),
            persons=persons,
            vehicles=vehicles,
            bicycles=bicycles,
            other=other,
            threat_detected=len(detections) > 0,
        ),
        detections=detections,
    )
