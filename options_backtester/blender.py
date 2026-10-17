"""Blended options pricing: pick the best model for the job.

``blended_price()`` selects the most appropriate pricing model based on
the option's characteristics:

- Exercise style: American -> BAW / binomial / Longstaff-Schwartz;
  European -> Black-Scholes.
- Time to expiry: very short dated -> binomial (discrete effects dominate);
  long dated -> Heston (volatility dynamics matter more).
- Path dependency: barrier / Asian payoffs -> Monte Carlo.
- Speed vs accuracy: a ``speed`` parameter (0=most accurate, 1=fastest)
  shifts the choice toward faster or more precise models.

Returns the price plus which model was selected and why. An optional
weighted-blend mode averages N models instead of picking one.
"""

from __future__ import annotations

from . import models

# Model metadata: exercise styles handled, typical use
_MODEL_INFO = {
    "black_scholes": {
        "styles": {"european"},
        "description": "Closed-form European price; fastest and exact under constant vol.",
    },
    "binomial": {
        "styles": {"european", "american"},
        "description": "CRR tree; handles discrete dividends and early exercise.",
    },
    "trinomial": {
        "styles": {"european", "american"},
        "description": "Trinomial tree; smoother convergence than binomial.",
    },
    "monte_carlo": {
        "styles": {"european"},
        "description": "Monte Carlo; flexible for path-dependent payoffs.",
    },
    "longstaff_schwartz": {
        "styles": {"american"},
        "description": "Least-squares Monte Carlo; American exercise via regression.",
    },
    "baw": {
        "styles": {"american"},
        "description": "Barone-Adesi-Whaley; fast analytic American approximation.",
    },
    "heston": {
        "styles": {"european"},
        "description": "Stochastic volatility; captures skew/smile for longer dated.",
    },
    "bjerksund_stensland": {
        "styles": {"american"},
        "description": "Bjerksund-Stensland; refined American approximation.",
    },
}


def select_model(style="european", T=1.0, path_dependent=False, speed=0.5,
                 american=False):
    """Choose the best model name for the given characteristics.

    Returns (model_name, reasoning).
    """
    if american:
        style = "american"
    style = style.lower()
    if style not in ("european", "american"):
        raise ValueError("style must be 'european' or 'american'")

    if path_dependent:
        if style == "american":
            return ("longstaff_schwartz",
                    "Path-dependent American payoff: least-squares Monte Carlo "
                    "handles both early exercise and path dependency.")
        return ("monte_carlo",
                "Path-dependent European payoff: Monte Carlo is the flexible choice.")

    if style == "american":
        # American exercise
        if T < 0.08:  # very short dated: discrete effects dominate
            return ("binomial",
                    "Very short-dated American: binomial tree captures discrete "
                    "time steps and early exercise precisely.")
        if speed >= 0.7:
            return ("baw",
                    "American, speed prioritized: Barone-Adesi-Whaley is a fast "
                    "analytic approximation.")
        if speed <= 0.3:
            return ("longstaff_schwartz",
                    "American, accuracy prioritized: Longstaff-Schwartz gives "
                    "the most accurate American value via simulation.")
        return ("bjerksund_stensland",
                "American, balanced: Bjerksund-Stensland refines the analytic "
                "approximation with good speed.")
    else:
        # European exercise
        if T >= 1.0 and speed <= 0.6:
            return ("heston",
                    "Long-dated European: Heston stochastic volatility captures "
                    "volatility dynamics and skew that constant-vol misses.")
        if speed >= 0.8:
            return ("black_scholes",
                    "European, speed prioritized: Black-Scholes closed form is "
                    "exact under constant volatility.")
        return ("black_scholes",
                "European: Black-Scholes closed form is the standard choice.")


def blended_price(S, K, T, r, q=0.0, sigma=0.2, option_type="call",
                  style="european", path_dependent=False, speed=0.5,
                  blend_mode=False, blend_models=None, **kwargs):
    """Price an option with the best-suited model.

    Returns a dict with:
    - price: the selected model's price (or weighted average in blend mode)
    - model: name of the selected model (or "weighted_blend")
    - reasoning: why this model was chosen
    - components: per-model prices in blend mode (else None)
    """
    if blend_mode:
        return _weighted_blend(S, K, T, r, q, sigma, option_type,
                               style=style, models_list=blend_models, **kwargs)
    model_name, reasoning = select_model(style=style, T=T,
                                         path_dependent=path_dependent,
                                         speed=speed)
    model_kwargs = dict(kwargs)
    if model_name in ("binomial", "trinomial") and style.lower() == "american":
        model_kwargs["american"] = True
    price = models.price_with(model_name, S, K, T, r, q, sigma, option_type,
                              **model_kwargs)
    return {
        "price": price,
        "model": model_name,
        "reasoning": reasoning,
        "components": None,
    }


def _weighted_blend(S, K, T, r, q, sigma, option_type, style="european",
                    models_list=None, weights=None, **kwargs):
    """Average N models with optional weights."""
    style = style.lower()
    if models_list is None:
        if style == "american":
            models_list = ["baw", "bjerksund_stensland", "binomial"]
        else:
            models_list = ["black_scholes", "binomial", "monte_carlo"]
    # Filter to models that support the exercise style
    valid = [m for m in models_list
             if style in _MODEL_INFO[m]["styles"]]
    if not valid:
        raise ValueError(f"no models support style '{style}'")
    if weights is None:
        weights = [1.0 / len(valid)] * len(valid)
    if len(weights) != len(valid):
        raise ValueError("weights length must match models")
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    components = {}
    for name, w in zip(valid, weights):
        mk = dict(kwargs)
        if name in ("binomial", "trinomial") and style == "american":
            mk["american"] = True
        # Keep MC fast in blends
        if name == "monte_carlo" and "paths" not in mk:
            mk["paths"] = 20_000
        if name == "longstaff_schwartz" and "paths" not in mk:
            mk["paths"] = 10_000
        try:
            p = models.price_with(name, S, K, T, r, q, sigma, option_type, **mk)
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        components[name] = {"price": p, "weight": w}

    if not components:
        raise RuntimeError("all blend models failed")
    # Renormalize weights over successful models
    wsum = sum(c["weight"] for c in components.values())
    price = sum(c["price"] * c["weight"] / wsum for c in components.values())
    return {
        "price": price,
        "model": "weighted_blend",
        "reasoning": (f"Weighted average of {len(components)} models "
                      f"({', '.join(components)}) for robustness."),
        "components": components,
    }


def list_models():
    """Return metadata for all available models."""
    return {name: dict(info) for name, info in _MODEL_INFO.items()}
