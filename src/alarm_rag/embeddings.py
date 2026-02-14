from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


@dataclass(slots=True)
class HashingEmbedder:
    dim: int = 384

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        return _normalize(vec)
