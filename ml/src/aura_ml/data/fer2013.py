"""Download FER2013 (canonical Training / PublicTest / PrivateTest splits) and cache
it as one compact .npz of 48x48 grayscale faces.

    python -m aura_ml.data.fer2013
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np

from aura_ml.config import FER2013_DIR, PROCESSED_DIR

HF_REPO = "https://huggingface.co/datasets/Aaryan333/fer2013_train_publicTest_privateTest/resolve/main/data"
FILES = {
    "train": "train-00000-of-00001-5eab84e1c6a2fc27.parquet",
    "validation": "publicTest-00000-of-00001-f41bb7384b8aad6e.parquet",
    "test": "privateTest-00000-of-00001-4b8a0715cf1b7560.parquet",
}
CACHE = PROCESSED_DIR / "fer2013.npz"


def download(raw_dir: Path = FER2013_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        dest = raw_dir / name
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(f"{HF_REPO}/{name}", dest)


def build_cache(raw_dir: Path = FER2013_DIR, cache: Path = CACHE) -> Path:
    import pyarrow.parquet as pq
    from PIL import Image

    arrays = {}
    for split, name in FILES.items():
        table = pq.read_table(raw_dir / name).to_pydict()
        images = [
            np.array(Image.open(io.BytesIO(img["bytes"])).convert("L"), dtype=np.uint8)
            for img in table["image"]
        ]
        arrays[f"{split}_x"] = np.stack(images)
        arrays[f"{split}_y"] = np.array(table["label"], dtype=np.int64)
        print(split, arrays[f"{split}_x"].shape, np.bincount(arrays[f"{split}_y"], minlength=7))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    return cache


def load(cache: Path = CACHE) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not cache.exists():
        download()
        build_cache()
    z = np.load(cache)
    return {s: (z[f"{s}_x"], z[f"{s}_y"]) for s in FILES}


if __name__ == "__main__":
    download()
    print(build_cache())
