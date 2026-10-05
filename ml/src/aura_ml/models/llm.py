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
        quant = None
        if "bnb-4bit" not in base_model:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map=device, torch_dtype=torch.bfloat16, quantization_config=quant
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
    """Template replies keyed on the strategy line of the system prompt."""

    name = "mock"
    REPLIES = {
        "gentle, open-ended question": "That sounds really heavy. Could you tell me a little more about what has been weighing on you?",
        "restate": "So if I'm hearing you right, this has been building up for a while and it's wearing you down.",
        "name and validate": "It makes complete sense that you feel this way - anyone in your place would be struggling too.",
        "relatable experience": "I've been through something similar, and I remember how lonely it felt. You're not alone in this.",
        "affirm their strengths": "Reaching out like this takes real strength, and I believe you can get through this.",
        "practical, concrete suggestion": "Would it help to take one small step today, like writing down what worries you most?",
        "factual information": "Many people go through this, and talking to someone you trust can genuinely ease the load.",
        "warm, general support": "I'm here with you. Take your time - I'm listening.",
    }

    def stream(self, messages, *, use_adapter=True, **_) -> Iterator[str]:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if not use_adapter:  # summary request
            user_lines = [m["content"] for m in messages if m["role"] == "user"]
            reply = "- They talked about: " + (user_lines[-1][:160] if user_lines else "their day")
        else:
            reply = next(
                (r for key, r in self.REPLIES.items() if key in system),
                self.REPLIES["warm, general support"],
            )
        for i, word in enumerate(reply.split(" ")):
            yield word if i == 0 else " " + word
