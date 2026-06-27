"""
DermoScan – Lazy-loaded predictor singleton.
"""
from __future__ import annotations

from typing import Optional

from predictor import SkinPredictor

_predictor: Optional[SkinPredictor] = None


def get_predictor() -> SkinPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SkinPredictor()
    return _predictor
