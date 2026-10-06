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

## Pending GPU training

These models' code and evaluation are in place (see [TRAINING.md](TRAINING.md)), but they have not been trained in
this version yet:

| Model | Protocol | Reported metric |
|---|---|---|
| Emotion-cause extraction (RoBERTa, two heads) | ESConv test split, silver labels | span F1 (overlap & exact), has-cause F1 |
| Voice emotion (wav2vec2) | RAVDESS, 4 unseen test speakers | accuracy, macro-F1, confusion matrix |
| Facial emotion (ResNet-18) | FER2013 PrivateTest | accuracy, macro-F1 |
| Reply LLM (Llama 3.2 3B + LoRA) | ESConv test replies | perplexity vs base, ROUGE-L, distinct-n |

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
