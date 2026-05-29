import numpy as np


def compute_context(prices):
    returns = np.diff(prices)

    volatility = np.std(returns)
    trend = np.mean(returns)

    context = {}

    # volatility
    if volatility > np.percentile(np.abs(returns), 70):
        context["volatility"] = "high"
    else:
        context["volatility"] = "low"

    # trend
    if trend > 0:
        context["regime"] = "trend"
    else:
        context["regime"] = "range"

    return context
