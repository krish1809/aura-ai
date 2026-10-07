"""Build the fake camera/microphone media for e2e/voice-video.mjs from the local datasets:
a face video from a FER2013 test image and a speech clip from an unseen RAVDESS speaker.

    uv run python frontend/e2e/make_media.py
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

from aura_ml.config import RAVDESS_DIR
from aura_ml.data import fer2013

OUT = Path(__file__).parent / "media"
OUT.mkdir(exist_ok=True)
x, y = fer2013.load()["test"]
face = x[next(i for i in range(300) if y[i] == 4)]  # a "sad" test face
frame = np.full((480, 640), 120, np.uint8)
frame[90:390, 170:470] = cv2.resize(face, (300, 300), interpolation=cv2.INTER_CUBIC)
jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[
    1
].tobytes()
(OUT / "face.mjpeg").write_bytes(jpg * 60)  # MJPEG = concatenated JPEG frames
shutil.copy(RAVDESS_DIR / "Actor_22/03-01-04-02-02-01-22.wav", OUT / "speech.wav")
print(f"wrote {OUT}/face.mjpeg and {OUT}/speech.wav")
