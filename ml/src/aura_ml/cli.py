"""`aura-ml` command: one entry point for every data, training and demo step.

aura-ml build-ece | build-sft | get-fer2013
aura-ml train-ece | train-ser | train-fer | train-strategy | train-topic | train-llm   [script args...]
aura-ml analyze "I can't sleep because of my exams"   (lite mode, no GPU needed)
aura-ml status                                      (which trained artifacts exist)
"""

from __future__ import annotations

import importlib
import json
import sys

COMMANDS = {
    "build-ece": "aura_ml.data.ece_dataset",
    "build-sft": "aura_ml.data.sft_dataset",
    "get-fer2013": "aura_ml.data.fer2013",
    "train-ece": "aura_ml.training.train_ece",
    "train-ser": "aura_ml.training.train_ser",
    "train-fer": "aura_ml.training.train_fer",
    "train-strategy": "aura_ml.training.train_strategy",
    "train-topic": "aura_ml.training.train_topic",
    "train-llm": "aura_ml.training.train_llm",
}


def _status() -> None:
    from aura_ml.config import ARTIFACTS_DIR

    for name in ("ece", "ser", "fer", "strategy", "topic", "llm/aura-llama-3.2-3b-lora"):
        metrics = ARTIFACTS_DIR / name / "metrics.json"
        print(f"{name:32s} {'trained' if metrics.exists() else 'missing'}")


def _analyze(text: str) -> None:
    from aura_ml.pipeline import HubConfig, ModelHub, TurnState, analyze_turn

    hub = ModelHub(HubConfig(use_torch=False, llm_provider="mock")).load()
    analysis = analyze_turn(hub, text, TurnState())
    print(
        json.dumps(
            {
                k: analysis[k]
                for k in ("safety", "emotion", "cause", "entities", "topic", "strategy")
            },
            indent=2,
        )
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "status":
        _status()
    elif cmd == "analyze":
        _analyze(" ".join(rest))
    elif cmd in COMMANDS:
        sys.argv = [f"aura-ml {cmd}", *rest]
        importlib.import_module(COMMANDS[cmd]).main()
    else:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
