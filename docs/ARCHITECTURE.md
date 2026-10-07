# Architecture

## Components

| Package | Role |
|---|---|
| `ml/` → `aura_ml` | Datasets, model wrappers, training and evaluation, plus the per-turn analysis pipeline (`pipeline.py`). It has no web code. |
| `backend/` → `aura_api` | FastAPI app: auth, conversations, insights, memory, the WebSocket turn orchestrator and Alembic migrations. Imports `aura_ml`. |
| `frontend/` | React single-page app. In production it is built into static files and served by the backend. |

## Modes

`ModelHub` (in `aura_ml/pipeline.py`) loads each component independently and falls back instead of crashing:

| Component | Full mode (`AURA_ML_MODE=full`) | Lite mode (default) |
|---|---|---|
| Text emotion | DistilRoBERTa (j-hartmann) | TF-IDF model trained on GoEmotions, then keyword lexicon |
| Cause | Aura RoBERTa ECE model, rules as backup when it finds nothing | the silver-label rules |
| Entities | spaCy + relation patterns | relation patterns ("my mom", "my boss") |
| Topic, strategy | scikit-learn models | same |
| Speech-to-text, voice emotion, face | Whisper, wav2vec2, YuNet + ResNet-18 | unavailable (the UI disables mic/camera) |
| Replies | Llama 3.2 3B + Aura LoRA, or an OpenAI-compatible server | mock templates (or an OpenAI-compatible server) |

`GET /api/system/models` reports what loaded and what each component fell back to.

## One turn

```
client ──{"type":"message", text, audio?}──▶  WebSocket /api/ws/conversations/{id}
                                               │
  1. decode audio (16 kHz PCM WAV)  ─▶ Whisper ─▶ {"type":"transcript"}
  2. load history (last 8 messages), previous strategy, last known cause
  3. analyze_turn (perception thread):
       safety check · text/voice/face emotion → fusion · cause · entities · topic · strategy
  4. store user message + analysis            ─▶ {"type":"analysis", message}
  5. crisis?            → fixed safe reply + helplines  ─▶ {"type":"safety", resources} + tokens
     "are you real?"     → fixed honest answer
     otherwise           → build prompt (aura_ml.prompts) → stream LLM (LLM thread) ─▶ {"type":"token"}…
                           → drop any sentence claiming a human life before storing
  6. store assistant message (strategy, model)  ─▶ {"type":"done", message, conversation}
  7. background task (survives disconnects): update knowledge graph; every 4 user turns
     re-summarise the conversation with the base model (adapter off) into the user's notes
```

Messages on one connection are processed strictly in order. One sent while Aura is still replying waits instead of
being rejected, and a crisis message is accepted even when the queue is full. At start-up every model is run once
(warm-up), so the first real message doesn't pay for CUDA initialisation.

Webcam frames arrive separately (`{"type":"frame"}`, about one per second while the camera is on). They are analysed
on the perception thread, smoothed over the last 8 seconds, and returned as `{"type":"face"}` for the live preview.
The smoothed reading is attached to the next message. Frames are never written to disk or the database.

The model hub is synchronous. The backend runs it on **two single-thread executors**, one for the LLM and one for
perception, so GPU work is serialised but live face updates keep flowing while a reply streams.

## Emotion fusion

Each modality gives a distribution over 7 emotions (anger, disgust, fear, joy, sadness, surprise, neutral). The
RAVDESS and FER2013 label sets are projected onto these (e.g. *calm → neutral*). Fusion is a weighted average with
default weights text 0.5, voice 0.25, face 0.25. Missing modalities are dropped and the remaining weights renormalised.
Text weighs most because the voice and face models are trained on acted or posed data (RAVDESS, FER2013), which differ
from real microphones and webcams. The result records agreement between modalities and flags *mixed* when the top two
emotions are within 0.1.

## Prompt contract

`aura_ml.prompts.system_prompt` is the only place the LLM's instructions are written. Training data
(`aura_ml.data.sft_dataset`) and the live app both call it, so the fine-tuned model always sees the same fields:

```
Feeling: anxious or afraid            ← fused emotion
Possible cause: my exams               ← ECE (or the last cause in this conversation)
Topic: academic pressure               ← topic classifier
Mentioned: mom, sister                 ← entities
What you remember about them: …        ← user notes (ESConv "situation" during training)
For this reply, ask a gentle, open-ended question…   ← strategy predictor (gold strategy during training)
```

## Data model

```
users ─┬─< conversations ─< messages     message.analysis = JSON from analyze_turn (user turns)
       │                                 message.meta     = {strategy, model, safety} (assistant turns)
       ├── user_memories                 rolling notes, rebuilt from the latest conversation summaries
       └─< graph_nodes ─< graph_edges    person/entity ─involved_in→ cause ─causes→ emotion
                                         cause ─part_of→ topic, person ─associated_with→ emotion
```

Everything is scoped by `user_id`, and deleting a user cascades. `DELETE /api/memory` erases the notes, the graph and
the summaries but keeps the chat history.

## Security

- Passwords are hashed with bcrypt. Sessions are HS256 JWTs in an `httpOnly`, `SameSite=Lax` cookie (`Secure` behind
  HTTPS). WebSockets authenticate with the same cookie.
- Every query is scoped to the signed-in user; other users' conversations return 404, including over WebSocket.
- The app refuses to start in production with the default JWT secret.
- The static-file route resolves paths inside `frontend/dist` only, so path traversal is blocked (covered by a test).
- The UI renders user text with React, never as HTML. The knowledge-graph tooltip is a React panel, not the library's
  HTML tooltip.
