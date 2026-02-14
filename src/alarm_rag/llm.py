from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request


@dataclass(slots=True)
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    timeout_seconds: int = 120

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        return str(parsed.get("response", "")).strip()
