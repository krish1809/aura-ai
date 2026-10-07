# Results

Measured on held-out test sets by the training scripts (`artifacts/*/metrics.json`). Nothing in this file is copied from
a paper or estimated. Models that have not been trained yet are listed as pending.

## Text emotion: CPU model (GoEmotions → 7 emotions)

TF-IDF (word 1–2-grams + character 2–5-grams) → class-balanced logistic regression. Comments whose labels map to more
than one Ekman category are left out. Test set: 4,968 comments.

| | Accuracy | Macro-F1 |
|---|---|---|
| **Aura CPU model** | **64.2%** | **58.7%** |
| Keyword lexicon (the previous fallback) | 43.7% | 30.8% |

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.45 | 0.55 | 0.49 | 572 |
| disgust | 0.48 | 0.54 | 0.51 | 76 |
| fear | 0.55 | 0.76 | 0.64 | 80 |
| joy | 0.85 | 0.72 | 0.78 | 1,863 |
| neutral | 0.63 | 0.60 | 0.62 | 1,606 |
| sadness | 0.48 | 0.57 | 0.52 | 283 |
| surprise | 0.49 | 0.62 | 0.55 | 488 |

Known limitation: the official Ekman mapping folds *confusion* into surprise, so "I don't know what to do anymore" is
often read as surprise. In full mode, the DistilRoBERTa emotion model replaces this one.

## Support-strategy predictor (ESConv, 8 strategies)

TF-IDF over the seeker's message and Aura's previous reply, plus state features (previous strategy, position in the
conversation, emotion, topic) → logistic regression. Test set: 2,290 supporter turns from conversations never seen in
training.

| | Accuracy | Macro-F1 |
|---|---|---|
| **Aura strategy model** | **31.0%** | **22.8%** |
| Repeat the previous strategy | 22.0% | 19.0% |
| Always predict the majority class | 16.5% | 3.5% |

| Strategy | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Question | 0.41 | 0.43 | 0.42 | 411 |
| Others | 0.37 | 0.52 | 0.43 | 377 |
| Providing Suggestions | 0.30 | 0.45 | 0.36 | 414 |
| Affirmation and Reassurance | 0.21 | 0.30 | 0.25 | 375 |
| Restatement or Paraphrasing | 0.30 | 0.14 | 0.20 | 139 |
| Self-disclosure | 0.26 | 0.06 | 0.10 | 218 |
| Reflection of feelings | 0.14 | 0.03 | 0.05 | 201 |
| Information | 0.13 | 0.01 | 0.02 | 155 |

Strategy choice in ESConv is genuinely ambiguous: several strategies are often equally good, and published neural
systems report around 28–32% accuracy. The model rarely predicts the less frequent strategies (reflection, information,
self-disclosure). A class-balanced model or a fine-tuned transformer is the obvious next step.

## Topic classifier (ESConv, 6 topics)

TF-IDF over everything the user has said so far → logistic regression.

| | Accuracy | Macro-F1 |
|---|---|---|
| All test predictions | 68.6% | 62.5% |
| After the user's 1st message | 45.6% | |
| After 3 messages | 71.3% | |
| After 5 messages | 75.9% | |

Per-topic F1: job crisis 0.74, breakup 0.74, academic pressure 0.72, friends 0.71, ongoing depression 0.66, other
(rare topics such as sleep problems or bullying) 0.18.

## Emotion-cause extraction (RoBERTa, two heads)

roberta-base with a has-cause head and a BIO span head, trained for 8 epochs (2.6 minutes on an RTX 4050). The
checkpoint and the has-cause threshold are both chosen on validation. Test set: 2,217 seeker messages from unseen
conversations, 154 of them with an explicit cause (the natural ratio). Labels are the silver rules described in
[TRAINING.md](TRAINING.md), so these scores measure how well the model reproduces and generalises those rules.

| | Span F1 (overlap) | Span F1 (exact) | Has-cause accuracy | Has-cause F1 |
|---|---|---|---|---|
| **Aura model, calibrated threshold (0.98)** | **72.7%** | **61.5%** | **96.2%** | **77.7%** |
| Aura model, default threshold 0.5 | 70.2% | 57.6% | 95.1% | 73.0% |
| 2025 model on the same test set | 5.0% | 0.0% | 6.9% | 13.0% |

The 2025 model was trained without any negative examples, so it says "this message has a cause" for every message
(recall 100%, precision 7%). Its span scores are not a like-for-like comparison, because it learned different silver
labels. The new model's recall is high (89% overlap recall) and its precision is lower (61%): it often finds causes
the rules missed or marks slightly longer spans.

## Voice emotion (wav2vec2, RAVDESS)

`superb/wav2vec2-base-superb-er` fine-tuned with the CNN feature encoder frozen, for 20 epochs (6 minutes).
Speaker-independent split: the 4 test actors (2 female, 2 male) are never heard during training.

| | Accuracy (8 classes) | Macro-F1 |
|---|---|---|
| **Aura model** | **72.9%** | **70.2%** |
| 2025 model on the same test speakers | 44.6% | 29.5% |

| Emotion | Precision | Recall | F1 | Clips |
|---|---|---|---|---|
| angry | 0.86 | 0.94 | 0.90 | 32 |
| disgust | 0.76 | 0.91 | 0.83 | 32 |
| calm | 0.69 | 0.97 | 0.81 | 32 |
| surprised | 0.80 | 0.75 | 0.77 | 32 |
| fearful | 0.90 | 0.56 | 0.69 | 32 |
| sad | 0.69 | 0.56 | 0.62 | 32 |
| happy | 0.58 | 0.59 | 0.58 | 32 |
| neutral | 0.46 | 0.38 | 0.41 | 16 |

Neutral is most often confused with calm, which the app merges anyway. RAVDESS is acted speech, so expect lower
accuracy on spontaneous microphone audio; this is why voice gets a 0.25 weight in fusion.

## Facial emotion (ResNet-18, FER2013)

ImageNet-pretrained ResNet-18 at 96×96, GPU augmentation, label smoothing, 30 epochs (5 minutes). The model is chosen
on PublicTest and reported once on PrivateTest (3,589 faces), with horizontal-flip test-time augmentation.

| | Accuracy (7 classes) | Macro-F1 |
|---|---|---|
| **Aura model** | **72.0%** | **70.7%** |
| Human agreement on FER2013 (reported) | ~65% | |
| Best published models (approx.) | 73–76% | |

| Emotion | Precision | Recall | F1 | Faces |
|---|---|---|---|---|
| happy | 0.90 | 0.90 | 0.90 | 879 |
| surprise | 0.81 | 0.84 | 0.83 | 416 |
| disgust | 0.76 | 0.67 | 0.71 | 55 |
| neutral | 0.71 | 0.70 | 0.70 | 626 |
| angry | 0.63 | 0.66 | 0.64 | 491 |
| sad | 0.58 | 0.61 | 0.59 | 594 |
| fear | 0.61 | 0.54 | 0.57 | 528 |

## Reply LLM (Llama 3.2 3B + LoRA)

*Training in progress.* Results will be test-reply perplexity (base vs fine-tuned), ROUGE-L and distinct-n.

## Differences from the 2025 prototype

The 2025 course version reported numbers its saved artifacts don't support. For example, the paper claims 84.7% cause
F1 and 68.1% voice accuracy, while the artifacts measured 64.7% and 47.9%. That version also had evaluation problems
that this rebuild fixes:

| Problem in 2025 | Fix in this version |
|---|---|
| Cause labels came from noisy keywords (*to*, *for*, *on*) and the wrong side of *so*/*therefore* | High-precision rules, backward markers handled, feeling required for *because*/*since* |
| No negative examples: the cause classifier's 100% accuracy was meaningless | Utterances without a cause are negatives; validation/test keep the natural ratio |
| Inference decoded BIO tags as B=0/I=1/O=2, but training used O=0/B=1/I=2 | Character-offset BIO with one shared definition and tests |
| Voice-emotion train/test split shared speakers | Speaker-independent split |
| Fine-tuning data labelled every history turn "Aura" and guessed the topic from keywords | Real roles and ESConv's annotated topic and situation |
| The LLM was trained before the fine-tuning dataset existed, with a different prompt than inference | One prompt template shared by training and inference; loss only on the final reply |
