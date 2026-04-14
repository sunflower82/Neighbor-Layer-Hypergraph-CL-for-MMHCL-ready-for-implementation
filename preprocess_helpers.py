"""
preprocess_helpers.py — Shared utilities for Amazon data preprocessing scripts.

Consolidates duplicated patterns across:
  - preprocess_amazon_baby.py
  - reextract_image_features.py
  - verify_and_fix_text_features.py
"""

from __future__ import annotations

from collections.abc import Iterator
import gzip
import json
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Gzip JSON-lines parsing (Amazon metadata may use Python-eval format)
# ---------------------------------------------------------------------------


def iter_gzip_jsonlines(path: str) -> Iterator[dict[str, Any]]:
    """Yield parsed dicts from a gzip-compressed JSON-lines file.

    Amazon metadata files sometimes use Python repr format (single quotes,
    True/False instead of true/false) rather than strict JSON, so we fall
    back to ``eval`` when ``json.loads`` fails.
    """
    with gzip.open(path, "rb") as f:
        for line in f:
            try:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield eval(line)
            except Exception:
                continue


# ---------------------------------------------------------------------------
# Image URL selection (prefer high-res, fall back to imUrl)
# ---------------------------------------------------------------------------


def select_image_url(meta: dict[str, Any], *, force_https: bool = False) -> str:
    """Pick the best image URL from an Amazon metadata record."""
    url = ""
    hi_res = meta.get("imageURLHighRes")
    if hi_res:
        url = hi_res[0] if hi_res else ""
    if not url:
        url = meta.get("imUrl", "") or ""
    if force_https and url.startswith("http://"):
        url = "https://" + url[7:]
    return url


# ---------------------------------------------------------------------------
# CLIP embedding extraction (compatible with transformers v4 and v5+)
# ---------------------------------------------------------------------------


def clip_text_embeddings(model, inputs) -> torch.Tensor:
    """Extract (B, 512) text embeddings from a CLIP model forward pass."""
    raw = model.get_text_features(**inputs)
    if isinstance(raw, torch.Tensor):
        return raw
    if hasattr(raw, "pooler_output"):
        return model.text_projection(raw.pooler_output)
    return model.text_projection(raw[1])


def clip_image_embeddings(model, inputs) -> torch.Tensor:
    """Extract (B, 512) image embeddings from a CLIP model forward pass."""
    raw = model.get_image_features(**inputs)
    if isinstance(raw, torch.Tensor):
        return raw
    if hasattr(raw, "pooler_output"):
        return raw.pooler_output
    return raw[1]
