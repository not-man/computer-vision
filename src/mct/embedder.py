"""Единственная точка доступа к ReID. Остальной код не знает, какая ступень работает внутри."""
from typing import Protocol

import cv2
import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Возвращает нормированные векторы, одна строка на рамку."""
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def without_background(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Вырезка, где всё вне маски человека закрашено средним цветом ImageNet: после нормировки входа это почти ноль."""
    kept = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    painted = np.empty_like(crop_bgr)
    painted[:] = (104, 116, 124)
    painted[kept] = crop_bgr[kept]
    return painted


def _crops(frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray]:
    height, width = frame_bgr.shape[:2]
    crops = []
    for x1, y1, x2, y2 in np.asarray(boxes_xyxy).round().astype(int):
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        crops.append(frame_bgr[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)])
    return crops


class BoxmotEmbedder:
    """Ступень 1: BoxMOT только как кодировщик, без его трекеров."""

    def __init__(self, model: str = "osnet_x0_25_msmt17", device: str = "cpu") -> None:
        from boxmot.reid.factory import create_reid_encoder

        self._encoder = create_reid_encoder(model, device=device)
        self.name = f"boxmot:{model}"

    def embed(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        import torch
        from boxmot.structures import Boxes, Detections, Frame

        count = len(boxes_xyxy)
        image = torch.from_numpy(frame_bgr[:, :, ::-1].copy()).permute(2, 0, 1).contiguous()
        frame = Frame(image=image, sample_id="frame")
        detections = Detections(
            geometry=Boxes(torch.as_tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4).contiguous()),
            scores=torch.ones(count, dtype=torch.float32),
            class_ids=torch.zeros(count, dtype=torch.int64),
            sample_id="frame",
        )
        return _normalize(self._encoder.encode([frame], [detections])[0].numpy())

    def embed_crops(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Векторы готовых вырезок: у каждой свой фон, поэтому одним кадром их не передать."""
        import torch
        from boxmot.structures import Boxes, Detections, Frame

        frames, detections = [], []
        for number, crop in enumerate(crops_bgr):
            height, width = crop.shape[:2]
            frames.append(Frame(image=torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1).contiguous(), sample_id=str(number)))
            detections.append(Detections(
                geometry=Boxes(torch.tensor([[0.0, 0.0, float(width), float(height)]], dtype=torch.float32)),
                scores=torch.ones(1, dtype=torch.float32),
                class_ids=torch.zeros(1, dtype=torch.int64),
                sample_id=str(number),
            ))
        return _normalize(np.vstack([part.numpy() for part in self._encoder.encode(frames, detections)]))


class DinoEmbedder:
    """Универсальная модель без обучения на людях: проверка, не лучше ли она специализированной на этой записи."""

    SIZE = (196, 392)  # ширина и высота, кратные патчу 14
    BATCH = 48  # камеры считаются одновременно на одной видеокарте
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model: str = "facebook/dinov2-large", device: str = "cpu") -> None:
        from transformers import AutoModel

        self._device = f"cuda:{device}" if str(device).isdigit() else device
        self._model = AutoModel.from_pretrained(model).to(self._device).eval()
        self.name = f"hf:{model}"

    def embed_crops(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        import torch

        parts = []
        for start in range(0, len(crops_bgr), self.BATCH):
            images = [cv2.resize(crop, self.SIZE, interpolation=cv2.INTER_AREA if crop.shape[0] > self.SIZE[1] else cv2.INTER_LINEAR)[:, :, ::-1] for crop in crops_bgr[start:start + self.BATCH]]
            batch = (np.stack(images).astype(np.float32) / 255 - self.MEAN) / self.STD
            with torch.no_grad():
                tokens = self._model(pixel_values=torch.from_numpy(batch).permute(0, 3, 1, 2).to(self._device)).last_hidden_state
            parts.append(torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1).cpu().numpy())
        return _normalize(np.vstack(parts))


class UltralyticsEmbedder:
    """Ступень 2: признаки классификатора ImageNet. Слабее, но не требует BoxMOT."""

    def __init__(self, model: str = "yolo11n-cls.pt", device: str = "cpu") -> None:
        from ultralytics import YOLO

        self._model = YOLO(model)
        self._device = device
        self.name = f"ultralytics:{model}"

    def embed(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        crops = _crops(frame_bgr, boxes_xyxy)
        if not crops:
            return np.empty((0, 0), np.float32)
        vectors = self._model.embed(crops, device=self._device, verbose=False)
        return _normalize(np.stack([v.cpu().numpy() for v in vectors]))


class HistogramEmbedder:
    """Ступень 3: цвет верха и низа, только OpenCV. Различает людей хуже всех."""

    name = "histogram"

    def embed(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        vectors = []
        for crop in _crops(frame_bgr, boxes_xyxy):
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            middle = hsv.shape[0] // 2
            parts = [hsv[:middle] if middle else hsv, hsv[middle:]]
            hists = [cv2.calcHist([p], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel() for p in parts]
            vectors.append(np.concatenate(hists))
        return _normalize(np.array(vectors, np.float32).reshape(len(vectors), -1))


def crop_embedder(model: str, device: str = "cpu") -> "BoxmotEmbedder | DinoEmbedder":
    """Кодировщик готовых вырезок по имени модели: с косой чертой значит Hugging Face."""
    return DinoEmbedder(model, device) if "/" in model else BoxmotEmbedder(model, device=device)


def create_embedder(device: str = "cpu", model: str = "osnet_ain_x1_0_msmt17") -> Embedder:
    """Берёт первую рабочую ступень сверху вниз."""
    for factory in (lambda: BoxmotEmbedder(model, device=device), lambda: UltralyticsEmbedder(device=device)):
        try:
            return factory()
        except Exception as error:
            print(f"ReID: ступень недоступна ({type(error).__name__}: {error})")
    return HistogramEmbedder()
