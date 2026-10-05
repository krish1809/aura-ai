"""Filesystem layout shared by data prep, training and inference.

Every path can be overridden with an environment variable so the same code runs
from the repo, inside Docker, or on a Colab/Kaggle box with a mounted drive.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("AURA_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = Path(os.environ.get("AURA_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.environ.get("AURA_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))

ESCONV_DIR = RAW_DIR / "esconv"
RAVDESS_DIR = RAW_DIR / "ravdess"
FER2013_DIR = RAW_DIR / "fer2013"

ECE_DATA_DIR = PROCESSED_DIR / "ece"
SFT_DATA_DIR = PROCESSED_DIR / "sft"
STRATEGY_DATA_DIR = PROCESSED_DIR / "strategy"

ECE_MODEL_DIR = ARTIFACTS_DIR / "ece"
SER_MODEL_DIR = ARTIFACTS_DIR / "ser"
FER_MODEL_DIR = ARTIFACTS_DIR / "fer"
STRATEGY_MODEL_DIR = ARTIFACTS_DIR / "strategy"
LLM_ADAPTER_DIR = ARTIFACTS_DIR / "llm" / "aura-llama-3.2-3b-lora"

# Pretrained checkpoints pulled from the Hugging Face Hub
HF_ECE_BASE = "roberta-base"
HF_SER_BASE = "superb/wav2vec2-base-superb-er"
HF_STT = os.environ.get("AURA_STT_MODEL", "openai/whisper-base.en")
HF_TEXT_EMOTION = "j-hartmann/emotion-english-distilroberta-base"
HF_LLM_BASE = os.environ.get("AURA_LLM_BASE", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
SPACY_MODEL = "en_core_web_sm"

SEED = 42
