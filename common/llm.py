"""Minimal async OpenRouter client with on-disk response cache, retries and cost tracking."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", Path(__file__).resolve().parents[1] / "results" / ".cache"))

MODELS = {
    "haiku-5.5": "anthropic/claude-haiku-5.5",
    "gpt-6-luna": "openai/gpt-6-luna",
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
}


def _load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    for p in [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError("Set OPENROUTER_API_KEY (env var or .env)")


class LLM:
    def __init__(self, model: str, max_tokens: int = 8000, temperature: float = 0.0,
                 concurrency: int = 16, use_cache: bool = True):
        self.model = MODELS.get(model, model)
        self.max_tokens, self.temperature, self.use_cache = max_tokens, temperature, use_cache
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(timeout=420)
        self.key = _load_key()
        self.cost = 0.0
        self.tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, messages, tag) -> Path:
        h = hashlib.sha256(json.dumps([self.model, self.max_tokens, self.temperature, messages, tag],
                                      sort_keys=True).encode()).hexdigest()
        return CACHE_DIR / f"{h}.json"

    async def complete(self, prompt: str | list[dict], tag: str = "") -> dict:
        """Returns {text, finish_reason, usage, error}. Never raises on API failure (error is recorded)."""
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        cp = self._cache_path(messages, tag)
        if self.use_cache and cp.exists():
            return json.loads(cp.read_text())
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                "temperature": self.temperature}
        out = {"text": "", "finish_reason": None, "usage": {}, "error": None}
        async with self.sem:
            for attempt in range(6):
                try:
                    r = await self.client.post(URL, json=body, headers={"Authorization": f"Bearer {self.key}"})
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                    d = r.json()
                    if "error" in d:
                        raise RuntimeError(str(d["error"])[:300])
                    ch = d["choices"][0]
                    u = d.get("usage", {})
                    out = {"text": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
                           "usage": {"prompt": u.get("prompt_tokens", 0), "completion": u.get("completion_tokens", 0),
                                     "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                                     "cost": u.get("cost", 0.0)},
                           "error": None}
                    break
                except Exception as e:  # noqa: BLE001
                    out["error"] = f"{type(e).__name__}: {e}"[:300]
                    await asyncio.sleep(min(2 ** attempt, 30))
        if out["error"] is None:
            if self.use_cache:
                cp.write_text(json.dumps(out))
            u = out["usage"]
            self.cost += u.get("cost", 0.0)
            for k in self.tokens:
                self.tokens[k] += u.get(k, 0)
        return out

    async def aclose(self):
        await self.client.aclose()
