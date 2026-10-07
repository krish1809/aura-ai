"""LLM providers behind one streaming interface.

* `LocalLLM`        - Llama 3.2 3B (4-bit) + Aura's LoRA adapter, via transformers/peft.
                      The adapter can be switched off per call, so the same weights
                      also write neutral memory summaries with the base model.
* `OpenAICompatLLM` - any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio...),
                      e.g. the merged Aura model exported to GGUF and served by Ollama.
* `MockLLM`         - deterministic replies for tests, CI and frontend development.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path


class LLM:
    name = "llm"

    def stream(self, messages: list[dict], *, use_adapter: bool = True, **gen) -> Iterator[str]:
        raise NotImplementedError

    def generate(self, messages: list[dict], **kw) -> str:
        return "".join(self.stream(messages, **kw)).strip()


class LocalLLM(LLM):
    def __init__(
        self, base_model: str, adapter_dir: str | Path | None = None, device: str = "cuda"
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        adapter = Path(adapter_dir) if adapter_dir and Path(adapter_dir).exists() else None
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter or base_model))
        kwargs = {}
        if "bnb-4bit" not in base_model:  # pre-quantized checkpoints carry their own config
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map=device, dtype=torch.bfloat16, **kwargs
        )
        self.has_adapter = adapter is not None
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter))
        self.model.eval()
        self.name = f"{base_model}{' + ' + adapter.name if adapter else ''}"

    def stream(
        self, messages, *, use_adapter=True, max_new_tokens=160, temperature=0.7, top_p=0.9, **_
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120
        )
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-3),
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        ctx = (
            self.model.disable_adapter()
            if (self.has_adapter and not use_adapter)
            else nullcontext()
        )

        def run():
            with ctx:
                self.model.generate(**kwargs)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        yield from streamer
        thread.join()


class OpenAICompatLLM(LLM):
    def __init__(
        self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120.0
    ):
        self.base_url, self.model_id, self.api_key, self.timeout = (
            base_url.rstrip("/"),
            model,
            api_key,
            timeout,
        )
        self.name = f"{model} @ {self.base_url}"

    def stream(
        self, messages, *, use_adapter=True, max_new_tokens=160, temperature=0.7, top_p=0.9, **_
    ) -> Iterator[str]:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers=headers,
            timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta


class MockLLM(LLM):
    """Deterministic template replies keyed on the strategy line of the system prompt
    (varied per message), and transcript-based notes for summary requests."""

    name = "mock"
    REPLIES = {
        "gentle, open-ended question": [
            "That sounds really heavy. Could you tell me a little more about what has been weighing on you?",
            "I'd like to understand this better. When did you first start feeling this way?",
            "What has been the hardest part of this for you?",
        ],
        "restate": [
            "So if I'm hearing you right, this has been building up for a while and it's wearing you down.",
            "It sounds like you've been carrying a lot on your own lately.",
        ],
        "name and validate": [
            "It makes complete sense that you feel this way - anyone in your place would be struggling too.",
            "That sounds genuinely painful, and your feelings about it are valid.",
        ],
        "relatable experience": [
            "I've been through something similar, and I remember how lonely it felt. You're not alone in this.",
            "I remember feeling that way once too, and it took time before it eased.",
        ],
        "affirm their strengths": [
            "Reaching out like this takes real strength, and I believe you can get through this.",
            "You're handling more than most people realise, and you're still showing up. That matters.",
        ],
        "practical, concrete suggestion": [
            "Would it help to take one small step today, like writing down what worries you most?",
            "Maybe try breaking it into one small task for tomorrow - just one, nothing more.",
        ],
        "factual information": [
            "Many people go through this, and talking to someone you trust can genuinely ease the load.",
            "Stress like this is very common, and short breaks and sleep make a real difference to how we cope.",
        ],
        "warm, general support": [
            "I'm here with you. Take your time - I'm listening.",
            "Thank you for sharing that with me. I'm glad you're talking about it.",
        ],
    }

    def stream(self, messages, *, use_adapter=True, **_) -> Iterator[str]:
        if not use_adapter:  # memory-summary request: bullet the user's own recent lines
            prompt = messages[-1]["content"] if messages else ""
            said = [ln[6:].strip() for ln in prompt.splitlines() if ln.startswith("User: ")][-2:]
            reply = (
                "\n".join(f"- They said: {line[:140]}" for line in said)
                or "- They shared how their day went."
            )
        else:
            system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            options = next(
                (r for key, r in self.REPLIES.items() if key in system),
                self.REPLIES["warm, general support"],
            )
            seed = sum(map(ord, messages[-1]["content"])) if messages else 0
            reply = options[seed % len(options)]
        for i, word in enumerate(reply.split(" ")):
            yield word if i == 0 else " " + word
