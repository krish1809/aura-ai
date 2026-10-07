"""Facial emotion recognition: face detection (OpenCV) + a CNN trained on FER2013."""

from __future__ import annotations

import contextlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

from aura_ml.labels import FER2013_LABELS, FER2013_TO_CANONICAL, project_distribution

INPUT_SIZE = 96
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


class AuraCNN(nn.Module):
    """VGG-style CNN for 48x48 faces - the lightweight option (about 2.4M params)."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.1),
            )

        self.features = nn.Sequential(
            block(3, 64), block(64, 128), block(128, 256), block(256, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(arch: str = "resnet18", num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    if arch == "aura_cnn":
        return AuraCNN(num_classes)
    if arch == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
        return model
    raise ValueError(f"unknown arch {arch!r}")


def preprocess(gray48: np.ndarray) -> torch.Tensor:
    """uint8 [N,48,48] grayscale -> normalised float [N,3,96,96] (same path as training eval)."""
    x = torch.from_numpy(gray48).float().div(255).unsqueeze(1)
    x = nn.functional.interpolate(x, size=INPUT_SIZE, mode="bilinear", align_corners=False).repeat(
        1, 3, 1, 1
    )
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


class FaceDetector:
    """YuNet (OpenCV DNN, 230 KB ONNX) and, on OpenCV 4.x, a Haar cascade fallback.
    OpenCV 5 dropped the Haar classifier, so there YuNet is required."""

    def __init__(self, model_dir: Path | None = None):
        import cv2

        self.cv2 = cv2
        self.yunet = None
        onnx = (model_dir or Path(".")) / "face_detection_yunet_2023mar.onnx"
        if model_dir is not None and not onnx.exists():
            with contextlib.suppress(OSError):
                urllib.request.urlretrieve(YUNET_URL, onnx)
        if onnx.exists():
            self.yunet = cv2.FaceDetectorYN.create(str(onnx), "", (320, 320), 0.7, 0.3, 5000)
        self.haar = None
        if hasattr(cv2, "CascadeClassifier"):
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        if self.yunet is None and self.haar is None:
            raise RuntimeError(f"no face detector: download {YUNET_URL} to {onnx}")

    def largest_face(self, bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        h, w = bgr.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(bgr)
            if faces is not None:
                boxes = [tuple(int(v) for v in f[:4]) for f in faces]
        elif self.haar is not None:
            gray = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2GRAY)
            boxes = [
                tuple(int(v) for v in b)
                for b in self.haar.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
            ]
        if not boxes:
            return None
        return max(boxes, key=lambda b: b[2] * b[3])


class FacialEmotionRecognizer:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        meta = json.loads((model_dir / "config.json").read_text())
        self.model = build_model(meta["arch"], pretrained=False)
        self.model.load_state_dict(
            torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        )
        self.model.to(self.device).eval()
        self.detector = FaceDetector(model_dir)

    @torch.inference_mode()
    def predict_image(self, bgr: np.ndarray) -> dict:
        box = self.detector.largest_face(bgr)
        if box is None:
            return {"available": False, "reason": "no face detected"}
        x, y, w, h = box
        pad = int(0.15 * max(w, h))
        H, W = bgr.shape[:2]
        crop = bgr[max(0, y - pad) : min(H, y + h + pad), max(0, x - pad) : min(W, x + w + pad)]
        gray = self.detector.cv2.resize(
            self.detector.cv2.cvtColor(crop, self.detector.cv2.COLOR_BGR2GRAY), (48, 48)
        )
        batch = preprocess(np.stack([gray, gray[:, ::-1].copy()])).to(
            self.device
        )  # + horizontal-flip TTA
        probs = self.model(batch).float().softmax(-1).mean(0).tolist()
        raw = dict(zip(FER2013_LABELS, probs, strict=True))
        canonical = project_distribution(raw, FER2013_TO_CANONICAL)
        top = max(canonical, key=canonical.get)
        return {
            "available": True,
            "box": [int(v) for v in box],
            "probs": {k: round(v, 4) for k, v in canonical.items()},
            "emotion": top,
            "confidence": round(canonical[top], 4),
        }

    def predict_jpeg(self, data: bytes) -> dict:
        img = self.detector.cv2.imdecode(
            np.frombuffer(data, np.uint8), self.detector.cv2.IMREAD_COLOR
        )
        if img is None:
            return {"available": False, "reason": "could not decode image"}
        return self.predict_image(img)
