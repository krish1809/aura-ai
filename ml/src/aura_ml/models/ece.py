"""Two-head RoBERTa model for emotion-cause extraction (ECE).

Head 1 (clause): does this utterance state an explicit cause?  [CLS] -> 2 classes
Head 2 (span):   which tokens are the cause?                    tokens -> O / B / I

The module layout is identical to the 2025 (v1) model so that checkpoint can
still be loaded and benchmarked; v1 used the same O=0, B=1, I=2 tag ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, RobertaConfig, RobertaModel, RobertaPreTrainedModel
from transformers.utils import ModelOutput

from aura_ml.data.bio import IGNORE_INDEX, decode_bio


@dataclass
class ECEOutput(ModelOutput):
    loss: torch.Tensor | None = None
    clause_logits: torch.Tensor | None = None
    token_logits: torch.Tensor | None = None


class RoBERTaForECE(RobertaPreTrainedModel):
    clause_loss_weight = 0.3
    token_loss_weight = 0.7

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(drop)
        h = config.hidden_size
        self.clause_classifier = nn.Sequential(
            nn.Linear(h, h), nn.Tanh(), nn.Dropout(drop), nn.Linear(h, 2)
        )
        self.token_classifier = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, 3)
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        clause_labels: torch.Tensor | None = None,
        token_labels: torch.Tensor | None = None,
        token_class_weights: torch.Tensor | None = None,
    ) -> ECEOutput:
        hidden = self.dropout(self.roberta(input_ids=input_ids, attention_mask=attention_mask)[0])
        clause_logits = self.clause_classifier(hidden[:, 0])
        token_logits = self.token_classifier(hidden)
        loss = None
        if clause_labels is not None and token_labels is not None:
            clause_loss = nn.functional.cross_entropy(clause_logits, clause_labels)
            token_loss = nn.functional.cross_entropy(
                token_logits.reshape(-1, 3).float(),
                token_labels.reshape(-1),
                weight=token_class_weights,
                ignore_index=IGNORE_INDEX,
            )
            loss = self.clause_loss_weight * clause_loss + self.token_loss_weight * token_loss
        return ECEOutput(loss=loss, clause_logits=clause_logits, token_logits=token_logits)


@dataclass
class Cause:
    text: str
    start: int
    end: int
    confidence: float


@dataclass
class ECEPrediction:
    has_cause: bool
    clause_confidence: float
    causes: list[Cause]

    def to_dict(self) -> dict:
        return {
            "has_cause": self.has_cause,
            "clause_confidence": round(self.clause_confidence, 4),
            "causes": [c.__dict__ for c in self.causes],
        }


class EmotionCauseExtractor:
    """Inference wrapper: text in, character-accurate cause spans out."""

    def __init__(self, model_dir: str | Path, device: str | None = None, max_length: int = 128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        self.model = RoBERTaForECE.from_pretrained(str(model_dir)).to(self.device).eval()
        self.max_length = max_length

    @torch.inference_mode()
    def predict_batch(self, texts: list[str], threshold: float = 0.5) -> list[ECEPrediction]:
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping").tolist()
        out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        clause_p = out.clause_logits.float().softmax(-1)[:, 1].tolist()
        token_p = out.token_logits.float().softmax(-1)
        token_pred = token_p.argmax(-1).tolist()
        token_conf = token_p.max(-1).values.tolist()
        results = []
        for i, text in enumerate(texts):
            causes: list[Cause] = []
            if clause_p[i] >= threshold:
                for s, e in decode_bio(token_pred[i], [tuple(o) for o in offsets[i]]):
                    toks = [j for j, (a, b) in enumerate(offsets[i]) if a < e and b > s and a != b]
                    conf = sum(token_conf[i][j] for j in toks) / max(len(toks), 1)
                    causes.append(Cause(text[s:e], s, e, round(conf, 4)))
            results.append(ECEPrediction(bool(causes), clause_p[i], causes))
        return results

    def predict(self, text: str, threshold: float = 0.5) -> ECEPrediction:
        return self.predict_batch([text], threshold)[0]
