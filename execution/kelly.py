from __future__ import annotations

import numpy as np


def estimate_variance(pnls: list[float], window: int = 50) -> float:
    if len(pnls) < 10:
        return 0.25  # fallback (≈ random)

    recent = pnls[-int(window) :]
    return float(max(float(np.var(recent)), 1e-4))


def compute_kelly_fraction(edge: float, variance: float) -> float:
    v = float(max(float(variance), 1e-12))
    return float(edge) / v


def compute_kelly_size(
    *,
    edge: float,
    confidence: float,
    alpha_multiplier: float,
    variance: float,
    k: float = 0.5,
    min_size: float = 0.02,
    max_size: float = 0.25,
    max_kelly: float = 10.0,
) -> tuple[float, float]:
    edge_score = float(edge) * float(alpha_multiplier)
    kelly = compute_kelly_fraction(edge_score, variance)
    if kelly > float(max_kelly):
        kelly = float(max_kelly)
    if kelly < -float(max_kelly):
        kelly = -float(max_kelly)
    size = float(k) * float(kelly) * float(confidence)
    size = max(float(min_size), min(float(size), float(max_size)))
    return float(size), float(kelly)

