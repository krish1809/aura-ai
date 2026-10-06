# Training runbook

Every model is trained by one script that writes `artifacts/<model>/` and a `metrics.json` with its test results.
Run everything from the repository root. The `aura-ml` command wraps each step (`aura-ml --help`).

| Step | Command | Hardware | Time (RTX 4050 6 GB) | Download |
|---|---|---|---|---|
| Text emotion (CPU) | `aura-ml train-text-emotion` | CPU | ~3 min | GoEmotions, 4 MB |
| Strategy predictor | `aura-ml train-strategy` | CPU | ~3 min | none |
| Topic classifier | `aura-ml train-topic` | CPU | <1 min | none |
| Emotion-cause extraction | `aura-ml build-ece && aura-ml train-ece` | GPU | ~5 min | roberta-base, 500 MB |
| Voice emotion | `aura-ml train-ser` | GPU | ~15 min | wav2vec2-base, 380 MB |
| Facial emotion | `aura-ml get-fer2013 && aura-ml train-fer` | GPU | ~15 min | FER2013 133 MB, ResNet-18 45 MB |
| Reply LLM | `aura-ml build-sft && aura-ml train-llm` | GPU | ~1–2 h | Llama 3.2 3B 4-bit, 2.2 GB |

The three CPU models ship with the repository, so lite mode works right after cloning.

## 0. Environment

```bash
uv sync --extra train        # torch + transformers + peft + unsloth + trl (several GB)
```

Unsloth pins compatible torch versions. If its install fails on your platform, use a Colab or Kaggle GPU
notebook: clone the repo, `pip install -e "ml[train]"`, and run the same commands.

## 1. Data

| Dataset | Location | How to get it |
|---|---|---|
| ESConv | `data/raw/esconv/{train,validation,test}.jsonl` | [thu-coai/esconv](https://huggingface.co/datasets/thu-coai/esconv) |
| RAVDESS (speech) | `data/raw/ravdess/Actor_XX/*.wav` | [Zenodo 1188976](https://zenodo.org/records/1188976), `Audio_Speech_Actors_01-24.zip` |
| GoEmotions | `data/raw/goemotions/` | downloaded automatically |
| FER2013 | `data/raw/fer2013/` | `aura-ml get-fer2013` |

All splits are fixed and documented in the scripts. ESConv uses its official conversation-level split, so no
conversation appears in both training and test data.

## 2. Emotion-cause extraction (ECE)

```bash
aura-ml build-ece          # data/processed/ece/*.jsonl  (silver labels from aura_ml.data.causes)
aura-ml train-ece --epochs 8
```

ESConv has no cause annotations, so causes are labelled by high-precision rules (explicit markers like *because*,
*worried about*, *makes me feel*). Utterances without an explicit cause become negative examples. The model is selected
on validation span-overlap F1 and scored once on test. If the 2025 model is present in `artifacts/legacy/`, it is
scored on the same test set for comparison.

## 3. Voice emotion (SER)

```bash
aura-ml train-ser --epochs 20
```

Fine-tunes `superb/wav2vec2-base-superb-er` on RAVDESS with a **speaker-independent** split (actors 1–18 / 19–20 /
21–24). The CNN feature encoder is frozen; augmentation is random crops, gain and noise.

## 4. Facial emotion (FER)

```bash
aura-ml get-fer2013
aura-ml train-fer --arch resnet18 --epochs 30     # or --arch aura_cnn for a small from-scratch CNN
```

Standard protocol: Training / PublicTest (model selection) / PrivateTest (reported). Augmentation runs on the GPU;
the test score uses horizontal-flip test-time augmentation, as the app does.

## 5. Reply LLM (QLoRA)

```bash
aura-ml build-sft                  # data/processed/sft/*.jsonl, built with aura_ml.prompts
aura-ml train-llm --epochs 1
aura-ml train-llm --eval-only      # re-run the evaluation on a saved adapter
```

- Base model: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`. LoRA r=16, alpha=32, applied to every linear layer.
- Loss is computed **only on the final assistant reply** of each example. History turns were written under other
  strategies, so learning from them would blur the conditioning.
- Evaluation: perplexity of the gold test replies under the base vs fine-tuned model (same weights, adapter off/on),
  ROUGE-L and distinct-n of sampled replies, and `samples.jsonl` with side-by-side generations for reading.

To serve the model with Ollama instead of transformers, merge and export it to GGUF with Unsloth
(`model.save_pretrained_gguf(...)`), create an Ollama model named `aura`, and run the backend with
`AURA_LLM_PROVIDER=openai`.

## 6. Publishing trained models

The GPU artifacts are too large for git. Push each `artifacts/<model>` folder to a Hugging Face Hub model repo
(`huggingface-cli upload <user>/aura-<model> artifacts/<model>`) and document the download step for deployments.
