"""Implied-volatility surface construction and plotting.

Builds a strike x expiry IV grid from option quotes (each quote needs
``strike``, ``dte`` and either ``iv`` or a ``mid`` price to invert), then
renders a 3D surface and per-expiry smile slices with matplotlib.
"""

from __future__ import annotations

from .implied_vol import implied_volatility


def build_surface(
    quotes: list[dict],
    S: float,
    r: float,
) -> dict:
    """Build an IV surface grid from quotes.

    Each quote needs ``strike``, ``dte`` and either ``iv`` directly or a
    ``mid`` price (plus ``type``) from which IV is inverted. Returns
    ``{"strikes": [...], "dtes": [...], "ivs": [[...]]}`` where ``ivs[i][j]``
    is the IV for ``dtes[i]``, ``strikes[j]`` (None where no quote exists).
    """
    strikes = sorted({q["strike"] for q in quotes})
    dtes = sorted({q["dte"] for q in quotes})
    lookup: dict[tuple[int, float], float] = {}
    for q in quotes:
        if "iv" in q and q["iv"] is not None:
            iv = q["iv"]
        elif "mid" in q:
            iv = implied_volatility(q["mid"], S, q["strike"], q["dte"] / 365.0, r, q.get("type", "call"))
        else:
            continue
        lookup[(q["dte"], q["strike"])] = iv
    grid = [[lookup.get((d, k)) for k in strikes] for d in dtes]
    return {"strikes": strikes, "dtes": dtes, "ivs": grid}


def plot_surface(surface: dict, path: str, title: str = "Implied Volatility Surface") -> str:
    """Render a 3D IV surface PNG. Returns the output path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    strikes = np.array(surface["strikes"])
    dtes = np.array(surface["dtes"])
    ivs = np.array(
        [[v if v is not None else float("nan") for v in row] for row in surface["ivs"]]
    )
    X, Y = np.meshgrid(strikes, dtes)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, ivs, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Days to expiry")
    ax.set_zlabel("Implied volatility")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_smile(
    quotes: list[dict], dte: int, path: str, title: str | None = None
) -> str:
    """Render an IV-vs-strike smile slice PNG for one expiry. Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls = sorted(
        (q for q in quotes if q["dte"] == dte and q.get("type", "call") == "call"),
        key=lambda q: q["strike"],
    )
    xs = [q["strike"] for q in calls]
    ys = [q["iv"] for q in calls]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.set_title(title or f"IV smile, {dte} DTE (calls)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
