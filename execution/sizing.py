from __future__ import annotations


def compute_position_size(
    prob: float,
    confidence: float,
    alpha_multiplier: float,
    *,
    k: float = 2.0,
    min_size: float = 0.02,
    max_size: float = 0.25,
) -> float:
    # 1) edge vs fair price 0.5
    edge = abs(float(prob) - 0.5)

    # 2) alpha-weighted edge
    edge_score = edge * float(alpha_multiplier)

    # 3) sizing
    size = float(k) * edge_score * float(confidence)

    # 4) clip
    if size < min_size:
        return float(min_size)
    if size > max_size:
        return float(max_size)
    return float(size)


def apply_dd_risk_scaling(size: float, current_dd: float) -> float:
    dd = float(current_dd)
    s = float(size)
    if dd > 0.4:
        return s * 0.5
    if dd > 0.3:
        return s * 0.7
    return s

