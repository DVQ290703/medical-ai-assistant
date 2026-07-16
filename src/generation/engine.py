"""Generation — Inference engine LINH HOẠT: Groq API (chạy ngay) | local transformers.

Backend chọn qua configs/serving.yaml -> generation.backend:
  - groq : POST api.groq.com (OpenAI-compatible), chạy ngay, không cần GPU. Dùng khi model
           fine-tune chưa xong / máy yếu.
  - local: transformers + LoRA adapter (cần GPU khỏe). Deploy sau khi có adapter fine-tune.

Mỗi engine có generate(system, user) -> str. Log structured (model/latency) — chỗ này sau
gắn Langfuse observability.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import yaml

# Nạp .env (GROQ_API_KEY...) khi chạy local. Không có python-dotenv -> bỏ qua.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class GenConfig:
    backend: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_tokens: int = 1024
    api_key_env: str = "GROQ_API_KEY"
    timeout_s: int = 60
    local_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    adapter_path: str = "artifacts/adapters"
    system_prompt_path: str = "prompts/system_prompt_v1.txt"


def gen_config_from_yaml(path: str = "configs/serving.yaml") -> GenConfig:
    c = GenConfig()
    if not os.path.exists(path):
        return c
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    g = y.get("generation", {}) or {}
    c.backend = g.get("backend", c.backend)
    c.model = g.get("model", c.model)
    c.temperature = g.get("temperature", c.temperature)
    c.max_tokens = g.get("max_tokens", c.max_tokens)
    c.api_key_env = g.get("api_key_env", c.api_key_env)
    c.timeout_s = g.get("timeout_s", c.timeout_s)
    c.local_model = g.get("local_model", c.local_model)
    c.adapter_path = g.get("adapter_path", c.adapter_path)
    c.system_prompt_path = g.get("system_prompt", c.system_prompt_path)
    return c


class Engine:
    """Base. generate(system, user) -> câu trả lời (str).

    generate_messages(system, turns): multi-turn — turns là list {role, content}
    (lịch sử hội thoại đã cắt gọn), engine tự ghép với system. Mặc định fallback về
    generate() 1 lượt nếu engine con chưa override (an toàn ngược).
    """
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError

    def generate_messages(self, system: str, turns: list[dict]) -> str:
        # fallback: chỉ lấy lượt user cuối (mất ngữ cảnh, nhưng không vỡ)
        last_user = next((t["content"] for t in reversed(turns)
                          if t.get("role") == "user"), "")
        return self.generate(system, last_user)


class GroqEngine(Engine):
    """Gọi Groq API (OpenAI-compatible) qua requests — không cần SDK nặng."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.api_key_env)
        if not self.api_key:
            raise SystemExit(
                f"Thiếu {cfg.api_key_env} trong môi trường/.env. Lấy key free tại "
                "https://console.groq.com/keys"
            )

    def generate(self, system: str, user: str) -> str:
        return self.generate_messages(system, [{"role": "user", "content": user}])

    def generate_messages(self, system: str, turns: list[dict]) -> str:
        import requests

        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}] + turns,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        t0 = time.time()
        r = requests.post(self.URL, json=payload, headers=headers, timeout=self.cfg.timeout_s)
        if r.status_code == 429:
            raise SystemExit("[groq] 429 rate limit. Chờ chút hoặc giảm tần suất gọi.")
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        print(f"[gen] groq model={self.cfg.model} {time.time()-t0:.1f}s "
              f"tokens={usage.get('total_tokens','?')}")
        return text


class LocalEngine(Engine):
    """transformers + LoRA (cần GPU khỏe). Lazy-load. KHÔNG chạy nổi GPU 4GB."""

    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[gen] load local {self.cfg.local_model}...")
        self._tok = AutoTokenizer.from_pretrained(self.cfg.local_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.local_model, device_map="auto")
        if self.cfg.adapter_path and os.path.isdir(self.cfg.adapter_path):
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, self.cfg.adapter_path)
                print(f"[gen] đã gắn LoRA adapter: {self.cfg.adapter_path}")
            except Exception as e:
                print(f"[gen] không gắn được adapter ({e}); dùng base model.")

    def generate(self, system: str, user: str) -> str:
        return self.generate_messages(system, [{"role": "user", "content": user}])

    def generate_messages(self, system: str, turns: list[dict]) -> str:
        self._load()
        msgs = [{"role": "system", "content": system}] + turns
        prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=self.cfg.max_tokens,
                                   temperature=self.cfg.temperature,
                                   do_sample=self.cfg.temperature > 0)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()


def engine_from_config(cfg: GenConfig | None = None) -> Engine:
    cfg = cfg or gen_config_from_yaml()
    if cfg.backend == "groq":
        return GroqEngine(cfg)
    if cfg.backend == "local":
        return LocalEngine(cfg)
    raise SystemExit(f"backend '{cfg.backend}' chưa hỗ trợ (groq | local).")
