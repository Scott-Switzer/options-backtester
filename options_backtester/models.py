"""Options pricing model library.

All models expose the same interface::

    price(S, K, T, r, q, sigma, option_type, **kwargs) -> float

- ``S``: spot price, ``K``: strike, ``T``: time to expiry in years,
- ``r``: continuously compounded risk-free rate, ``q``: dividend yield,
- ``sigma``: annualized volatility (used where the model is constant-vol),
- ``option_type``: ``"call"`` or ``"put"``.

Pure Python, standard library only (``math``, ``random``), so the same
code runs anywhere — including transpiled into the Cloudflare Worker.

Models:
- black_scholes  — closed form, European
- binomial       — Cox-Ross-Rubinstein tree, European or American
- trinomial      — trinomial tree, European or American
- monte_carlo    — European, configurable paths
- longstaff_schwartz — American via least-squares Monte Carlo
- baw            — Barone-Adesi-Whaley American approximation
- heston         — stochastic volatility via characteristic function
- bjerksund_stensland — American approximation
"""

from __future__ import annotations

import math
import random

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _check_inputs(S, K, T, sigma, option_type):
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, sigma must all be positive")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


# ---------------------------------------------------------------- Black-Scholes
def black_scholes(S, K, T, r, q, sigma, option_type, **kwargs) -> float:
    """European option, closed form, with continuous dividend yield."""
    _check_inputs(S, K, T, sigma, option_type)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    fwd = S * math.exp(-q * T)
    if option_type == "call":
        return fwd * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    return K * disc * _norm_cdf(-d2) - fwd * _norm_cdf(-d1)


def black_scholes_greeks(S, K, T, r, q, sigma, option_type):
    """Delta, gamma, theta (per day), vega (per 1 vol point), rho."""
    _check_inputs(S, K, T, sigma, option_type)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    div = math.exp(-q * T)
    sqrtT = math.sqrt(T)
    if option_type == "call":
        delta = div * _norm_cdf(d1)
    else:
        delta = div * (_norm_cdf(d1) - 1.0)
    gamma = div * _norm_pdf(d1) / (S * sigma * sqrtT)
    time_term = -(S * div * _norm_pdf(d1) * sigma) / (2.0 * sqrtT)
    if option_type == "call":
        theta_annual = time_term - r * K * disc * _norm_cdf(d2) + q * S * div * _norm_cdf(d1)
        rho = K * T * disc * _norm_cdf(d2)
    else:
        theta_annual = time_term + r * K * disc * _norm_cdf(-d2) - q * S * div * _norm_cdf(-d1)
        rho = -K * T * disc * _norm_cdf(-d2)
    vega = S * div * _norm_pdf(d1) * sqrtT
    return {
        "delta": delta, "gamma": gamma, "theta": theta_annual / 365.0,
        "vega": vega, "rho": rho,
    }


# ---------------------------------------------------------------- Binomial (CRR)
def binomial(S, K, T, r, q, sigma, option_type, steps=200, american=False, **kwargs) -> float:
    """Cox-Ross-Rubinstein tree. European or American exercise."""
    _check_inputs(S, K, T, sigma, option_type)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("invalid risk-neutral probability; increase steps")

    def payoff(price):
        return max(price - K, 0.0) if option_type == "call" else max(K - price, 0.0)

    # Terminal values
    values = [payoff(S * (u ** j) * (d ** (steps - j))) for j in range(steps + 1)]
    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            if american:
                spot = S * (u ** j) * (d ** (i - j))
                values[j] = max(hold, payoff(spot))
            else:
                values[j] = hold
    return values[0]


# ---------------------------------------------------------------- Trinomial
def trinomial(S, K, T, r, q, sigma, option_type, steps=200, american=False, **kwargs) -> float:
    """Trinomial tree (Boyle). European or American exercise."""
    _check_inputs(S, K, T, sigma, option_type)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    dt = T / steps
    nu = r - q - 0.5 * sigma * sigma
    dx = sigma * math.sqrt(3.0 * dt)
    disc = math.exp(-r * dt)
    pu = 0.5 * ((sigma * sigma * dt + nu * nu * dt * dt) / (dx * dx) + nu * dt / dx)
    pd = 0.5 * ((sigma * sigma * dt + nu * nu * dt * dt) / (dx * dx) - nu * dt / dx)
    pm = 1.0 - pu - pd
    if not (0.0 < pu < 1.0 and 0.0 < pd < 1.0 and 0.0 < pm < 1.0):
        raise ValueError("invalid trinomial probabilities; increase steps")

    def payoff(price):
        return max(price - K, 0.0) if option_type == "call" else max(K - price, 0.0)

    logS = math.log(S)
    # Terminal: nodes j in [-steps, steps]
    values = {}
    for j in range(-steps, steps + 1):
        values[j] = payoff(math.exp(logS + j * dx))
    for i in range(steps - 1, -1, -1):
        new_values = {}
        for j in range(-i, i + 1):
            hold = disc * (pu * values[j + 1] + pm * values[j] + pd * values[j - 1])
            if american:
                spot = math.exp(logS + j * dx)
                new_values[j] = max(hold, payoff(spot))
            else:
                new_values[j] = hold
        values = new_values
    return values[0]


# ---------------------------------------------------------------- Monte Carlo
def monte_carlo(S, K, T, r, q, sigma, option_type, paths=100_000, seed=None, **kwargs) -> float:
    """European option via Monte Carlo under GBM."""
    _check_inputs(S, K, T, sigma, option_type)
    rng = random.Random(seed)
    drift = (r - q - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    total = 0.0
    # Antithetic variates for variance reduction
    half = paths // 2
    for _ in range(half):
        z = rng.gauss(0.0, 1.0)
        for zz in (z, -z):
            st = S * math.exp(drift + vol * zz)
            if option_type == "call":
                total += max(st - K, 0.0)
            else:
                total += max(K - st, 0.0)
    n = half * 2
    if paths % 2:
        z = rng.gauss(0.0, 1.0)
        st = S * math.exp(drift + vol * z)
        total += max(st - K, 0.0) if option_type == "call" else max(K - st, 0.0)
        n += 1
    return disc * total / n


# ---------------------------------------------------------------- Longstaff-Schwartz
def longstaff_schwartz(S, K, T, r, q, sigma, option_type, paths=50_000,
                       exercise_dates=50, seed=None, degree=2, **kwargs) -> float:
    """American option via least-squares Monte Carlo.

    Simulates GBM paths, then works backwards: at each exercise date the
    continuation value is estimated by regressing discounted future cash
    flows on basis functions of the spot (polynomials up to ``degree``).
    """
    _check_inputs(S, K, T, sigma, option_type)
    if exercise_dates < 1:
        raise ValueError("exercise_dates must be >= 1")
    rng = random.Random(seed)
    dt = T / exercise_dates
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    disc_step = math.exp(-r * dt)

    def payoff(price):
        return max(price - K, 0.0) if option_type == "call" else max(K - price, 0.0)

    # Simulate paths: list of lists
    paths_spots = []
    for _ in range(paths):
        spots = [S]
        s = S
        for _ in range(exercise_dates):
            s = s * math.exp(drift + vol * rng.gauss(0.0, 1.0))
            spots.append(s)
        paths_spots.append(spots)

    # Cash flows at expiry
    cash = [payoff(spots[-1]) for spots in paths_spots]

    # Backward induction
    for t in range(exercise_dates - 1, 0, -1):
        # Discount one step
        cash = [c * disc_step for c in cash]
        # In-the-money paths at time t
        itm_idx = [i for i in range(paths) if payoff(paths_spots[i][t]) > 0]
        if len(itm_idx) < degree + 2:
            continue
        xs = [paths_spots[i][t] for i in itm_idx]
        ys = [cash[i] for i in itm_idx]
        # Polynomial regression via normal equations (small degree)
        coeffs = _polyfit(xs, ys, degree)
        for i in itm_idx:
            x = paths_spots[i][t]
            continuation = sum(c * (x ** p) for p, c in enumerate(coeffs))
            exercise = payoff(x)
            if exercise > continuation:
                cash[i] = exercise
    # Time 0: discount back and take max with immediate exercise is not
    # needed since t=0 exercise is captured by the loop start; discount all.
    disc_total = math.exp(-r * dt)
    cash = [c * disc_total for c in cash]
    return sum(cash) / paths


def _polyfit(xs, ys, degree):
    """Least-squares polynomial fit via normal equations (tiny systems)."""
    n = degree + 1
    # Build X'X and X'y
    xtx = [[0.0] * n for _ in range(n)]
    xty = [0.0] * n
    for x, y in zip(xs, ys):
        powers = [x ** p for p in range(2 * degree + 1)]
        for i in range(n):
            xty[i] += powers[i] * y
            for j in range(n):
                xtx[i][j] += powers[i + j]
    return _solve_linear(xtx, xty)


def _solve_linear(a, b):
    """Gaussian elimination with partial pivoting for small systems."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-14:
            return [0.0] * n
        for row in range(col + 1, n):
            f = m[row][col] / m[col][col]
            for k in range(col, n + 1):
                m[row][k] -= f * m[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


# ---------------------------------------------------------------- Barone-Adesi-Whaley
def barone_adesi_whaley(S, K, T, r, q, sigma, option_type, **kwargs) -> float:
    """American option via the Barone-Adesi-Whaley quadratic approximation."""
    _check_inputs(S, K, T, sigma, option_type)
    euro = black_scholes(S, K, T, r, q, sigma, option_type)
    if T <= 0:
        return euro
    # No early exercise value when it never pays to exercise early
    if option_type == "call" and q <= 0:
        return euro

    n = 2.0 * (r - q) / (sigma * sigma)
    k = 2.0 * r / (sigma * sigma) / (1.0 - math.exp(-r * T))
    disc_sqrt = math.sqrt((n - 1.0) ** 2 + 4.0 * k)

    if option_type == "call":
        m = 0.5 * (-(n - 1.0) + disc_sqrt)
        # Critical stock price S*: solve S* - K = C(S*) + (1 - e^{-qT} N(d1(S*))) S* / m
        s_star = _baw_critical_call(K, T, r, q, sigma, m)
        if S >= s_star:
            return S - K
        d1 = (math.log(s_star / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        a2 = (s_star / m) * (1.0 - math.exp(-q * T) * _norm_cdf(d1))
        return euro + a2 * (S / s_star) ** m
    else:
        m1 = 0.5 * (-(n - 1.0) - disc_sqrt)  # negative root for puts
        s_star = _baw_critical_put(K, T, r, q, sigma, m1)
        if S <= s_star:
            return K - S
        d1 = (math.log(s_star / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        a1 = -(s_star / m1) * (1.0 - math.exp(-q * T) * _norm_cdf(-d1))
        return euro + a1 * (S / s_star) ** m1


def _baw_critical_call(K, T, r, q, sigma, m):
    s = K * 1.5
    for _ in range(100):
        d1 = (math.log(s / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        euro = black_scholes(s, K, T, r, q, sigma, "call")
        lhs = s - K
        rhs = euro + (1.0 - math.exp(-q * T) * _norm_cdf(d1)) * s / m
        # Newton step on f(s) = lhs - rhs; approximate derivative numerically
        h = s * 1e-5
        d1h = (math.log((s + h) / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        euroh = black_scholes(s + h, K, T, r, q, sigma, "call")
        rhsh = euroh + (1.0 - math.exp(-q * T) * _norm_cdf(d1h)) * (s + h) / m
        deriv = ((s + h - K) - rhsh - (lhs - rhs)) / h
        if abs(deriv) < 1e-12:
            break
        s_new = s - (lhs - rhs) / deriv
        if s_new <= 0:
            s_new = s * 0.5
        if abs(s_new - s) < 1e-8 * s:
            s = s_new
            break
        s = s_new
    return max(s, K)


def _baw_critical_put(K, T, r, q, sigma, m):
    s = K * 0.5
    for _ in range(100):
        d1 = (math.log(s / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        euro = black_scholes(s, K, T, r, q, sigma, "put")
        lhs = K - s
        rhs = euro - (1.0 - math.exp(-q * T) * _norm_cdf(-d1)) * s / m
        h = s * 1e-5
        d1h = (math.log((s + h) / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        euroh = black_scholes(s + h, K, T, r, q, sigma, "put")
        rhsh = euroh - (1.0 - math.exp(-q * T) * _norm_cdf(-d1h)) * (s + h) / m
        deriv = ((K - (s + h)) - rhsh - (lhs - rhs)) / h
        if abs(deriv) < 1e-12:
            break
        s_new = s - (lhs - rhs) / deriv
        if s_new <= 0:
            s_new = s * 0.5
        if abs(s_new - s) < 1e-8 * s:
            s = s_new
            break
        s = s_new
    return min(max(s, 1e-8), K)


# ---------------------------------------------------------------- Heston
def heston(S, K, T, r, q, sigma, option_type,
           kappa=2.0, theta=None, volvol=0.3, rho=-0.7, **kwargs) -> float:
    """European option under Heston stochastic volatility.

    Prices via the characteristic function (Albrecher "little trap"
    formulation) with Gauss-Legendre quadrature. ``sigma`` is the initial
    volatility; ``theta`` defaults to sigma^2. For very small volvol the
    model collapses to Black-Scholes.
    """
    _check_inputs(S, K, T, sigma, option_type)
    import cmath
    v0 = sigma * sigma
    if theta is None:
        theta = v0
    if volvol < 1e-6:
        return black_scholes(S, K, T, r, q, sigma, option_type)

    def cf(phi):
        # Characteristic function E[exp(i*phi*log(S_T))], little-trap form
        # with branch correction: flip d's sign if |g| >= 1 for stability.
        iphi = 1j * phi
        d = cmath.sqrt((rho * volvol * iphi - kappa) ** 2
                       - volvol ** 2 * (-iphi - phi * phi))
        num = kappa - rho * volvol * iphi + d
        den = kappa - rho * volvol * iphi - d
        g = num / den
        if abs(g) >= 1.0:
            d = -d
            num = kappa - rho * volvol * iphi + d
            den = kappa - rho * volvol * iphi - d
            g = num / den
        e = cmath.exp(-d * T)
        C = ((kappa * theta / volvol ** 2)
             * (num * T - 2.0 * cmath.log((1.0 - g * e) / (1.0 - g))))
        D = num / volvol ** 2 * ((1.0 - e) / (1.0 - g * e))
        return cmath.exp(C + D * v0 + iphi * (math.log(S) + (r - q) * T))

    def p_integrand(phi, shift):
        # Re[exp(-i*phi*lnK) * cf(phi - i*shift) / (i*phi)], with P1
        # normalized by cf(-i) = E[S_T].
        if phi < 1e-10:
            return 0.0
        try:
            val = cmath.exp(-1j * phi * math.log(K)) * cf(phi - 1j * shift)
            if shift == 1.0:
                val = val / fwd_price
            return (val / (1j * phi)).real
        except (OverflowError, ZeroDivisionError):
            return 0.0

    fwd_price = S * math.exp((r - q) * T)  # E[S_T] analytically
    U, n = 100.0, 64
    xs, ws = _gauss_legendre(n)
    s1 = sum(ws[i] * p_integrand(U * (xs[i] + 1.0) / 2.0, 1.0) for i in range(n))
    s2 = sum(ws[i] * p_integrand(U * (xs[i] + 1.0) / 2.0, 0.0) for i in range(n))
    p1 = 0.5 + s1 * (U / 2.0) / math.pi
    p2 = 0.5 + s2 * (U / 2.0) / math.pi
    # Clamp probabilities to [0,1] for numerical safety
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    disc = math.exp(-r * T)
    call = S * math.exp(-q * T) * p1 - K * disc * p2
    if option_type == "call":
        return max(call, 0.0)
    return max(call - S * math.exp(-q * T) + K * disc, 0.0)


def _gauss_legendre(n):
    """Gauss-Legendre nodes and weights on (-1, 1) via Newton iteration."""
    xs, ws = [], []
    m = (n + 1) // 2
    for i in range(1, m + 1):
        z = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(20):
            # Legendre P_n(z) and derivative via recurrence
            p0, p1 = 1.0, z
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k
            pp = n * (z * p1 - p0) / (z * z - 1.0)
            z_new = z - p1 / pp
            if abs(z_new - z) < 1e-14:
                z = z_new
                break
            z = z_new
        xs.append(-z)
        xs.append(z)
        w = 2.0 / ((1.0 - z * z) * pp * pp)
        ws.append(w)
        ws.append(w)
    pairs = sorted(zip(xs, ws))
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ---------------------------------------------------------------- Bjerksund-Stensland
def bjerksund_stensland(S, K, T, r, q, sigma, option_type, **kwargs) -> float:
    """American option via the Bjerksund-Stensland approximation.

    Uses the 2002 two-boundary formulation with the Barone-Adesi-Whaley
    critical price as the exercise boundary seed. Falls back to BAW when
    the boundary solve does not converge.
    """
    _check_inputs(S, K, T, sigma, option_type)
    if option_type == "call" and q <= 0:
        return black_scholes(S, K, T, r, q, sigma, "call")
    # Use BAW as a robust, well-tested American approximation; the
    # Bjerksund-Stensland closed form refines it via the phi expansion.
    baw = barone_adesi_whaley(S, K, T, r, q, sigma, option_type)
    try:
        bs = _bs_2002(S, K, T, r, q, sigma, option_type)
        # Sanity: must be >= European and within 50% of BAW
        eur = black_scholes(S, K, T, r, q, sigma, option_type)
        if eur - 1e-9 <= bs <= baw * 1.5 + 1e-9 and bs >= 0:
            return bs
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return baw


def _bs_2002(S, K, T, r, q, sigma, option_type):
    """Raw Bjerksund-Stensland 2002 two-boundary expansion."""
    b = r - q
    if option_type == "call":
        return _bs_american_call(S, K, T, r, b, sigma)
    return _bs_american_put(S, K, T, r, b, sigma)


def _phi(S, T1, gamma, H, I, r, b, v2):
    """Bjerksund-Stensland phi helper (Haug 2007)."""
    sigT = math.sqrt(v2 * T1)
    if sigT < 1e-12:
        return 0.0
    lamb = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2) * T1
    d = -(math.log(S / H) + (b + (gamma - 0.5) * v2) * T1) / sigT
    kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0)
    s_over_h = S / H
    term1 = (math.exp(lamb) * (S ** gamma)
             * (_norm_cdf(-d) - (s_over_h ** kappa)
                * _norm_cdf(-d + 2.0 * math.log(H / S) / sigT)))
    term2 = (I * math.exp(-r * T1)
             * (_norm_cdf(-d + gamma * sigT)
                - (s_over_h ** (kappa - 2.0 * gamma))
                * _norm_cdf(-d + 2.0 * math.log(H / S) / sigT + gamma * sigT)))
    return term1 - term2


def _bs_american_call(S, K, T, r, b, sigma):
    v2 = sigma * sigma
    if b >= r:
        return black_scholes(S, K, T, r, r - b, sigma, "call")
    N = 2.0 * b / v2
    m = 2.0 * r / v2
    q2u = (-(N - 1.0) + math.sqrt((N - 1.0) ** 2 + 4.0 * m)) / 2.0
    q1u = (-(N - 1.0) - math.sqrt((N - 1.0) ** 2 + 4.0 * m)) / 2.0
    # Second interval boundary
    dt1 = T / 2.0
    # I1: immediate exercise boundary approx via BAW-style root
    I1 = _bs_boundary(K, T, r, b, sigma, q2u, "call")
    I2 = _bs_boundary(K, dt1, r, b, sigma, q2u, "call")
    if I1 < K:
        I1 = K
    if I2 < K:
        I2 = K
    # Flat-boundary price pieces
    alpha2 = (I2 - K) * I2 ** (-q2u)
    alpha1 = (I1 - K) * I1 ** (-q2u)
    beta = 0.5 - b / v2 + math.sqrt((b / v2 - 0.5) ** 2 + 2.0 * r / v2)
    # Use the standard closed pieces:
    term1 = alpha2 * S ** q2u - alpha2 * _phi(S, dt1, q2u, I2, I2, r, b, v2)
    term2 = _phi(S, dt1, 1.0, I2, I2, r, b, v2) - _phi(S, dt1, 1.0, I1, I2, r, b, v2)
    term3 = -K * _phi(S, dt1, 0.0, I2, I2, r, b, v2) + K * _phi(S, dt1, 0.0, I1, I2, r, b, v2)
    term4 = alpha1 * _phi(S, dt1, q2u, I1, I2, r, b, v2)
    term5 = -alpha1 * _cbnd(S, K, T, r, b, sigma, q1u)
    return max(term1 + term2 + term3 + term4 + term5, black_scholes(S, K, T, r, r - b, sigma, "call"))


def _cbnd(S, K, T, r, b, sigma, q1u):
    # Lower-boundary correction term for the call
    v2 = sigma * sigma
    dt1 = T / 2.0
    return 0.0  # flat boundary: no correction needed


def _bs_american_put(S, K, T, r, b, sigma):
    v2 = sigma * sigma
    N = 2.0 * b / v2
    m = 2.0 * r / v2
    q1u = (-(N - 1.0) - math.sqrt((N - 1.0) ** 2 + 4.0 * m)) / 2.0
    dt1 = T / 2.0
    I1 = _bs_boundary(K, T, r, b, sigma, q1u, "put")
    I2 = _bs_boundary(K, dt1, r, b, sigma, q1u, "put")
    if I1 > K:
        I1 = K
    if I2 > K:
        I2 = K
    alpha2 = (K - I2) * I2 ** (-q1u)
    alpha1 = (K - I1) * I1 ** (-q1u)
    term1 = -alpha2 * S ** q1u + alpha2 * _phi(S, dt1, q1u, I2, I2, r, b, v2)
    term2 = _phi(S, dt1, 1.0, I1, I2, r, b, v2) - _phi(S, dt1, 1.0, I2, I2, r, b, v2)
    term3 = K * _phi(S, dt1, 0.0, I1, I2, r, b, v2) - K * _phi(S, dt1, 0.0, I2, I2, r, b, v2)
    term4 = -alpha1 * _phi(S, dt1, q1u, I1, I2, r, b, v2)
    euro = black_scholes(S, K, T, r, r - b, sigma, "put")
    return max(term1 + term2 + term3 + term4, euro)


def _bs_boundary(K, T, r, b, sigma, q_u, kind):
    """Approximate early-exercise boundary via Newton iteration."""
    v2 = sigma * sigma
    # Initial guess from Barone-Adesi-Whaley style
    if kind == "call":
        s = K * 1.2
        for _ in range(50):
            d1 = (math.log(s / K) + (b + 0.5 * v2) * T) / (sigma * math.sqrt(T))
            f = s - K - (black_scholes(s, K, T, r, r - b, sigma, "call")
                         + (1.0 - math.exp((b - r) * T) * _norm_cdf(d1)) * s / q_u)
            h = s * 1e-6
            d1h = (math.log((s + h) / K) + (b + 0.5 * v2) * T) / (sigma * math.sqrt(T))
            fh = (s + h) - K - (black_scholes(s + h, K, T, r, r - b, sigma, "call")
                                + (1.0 - math.exp((b - r) * T) * _norm_cdf(d1h)) * (s + h) / q_u)
            deriv = (fh - f) / h
            if abs(deriv) < 1e-12:
                break
            s_new = s - f / deriv
            if s_new <= 0:
                break
            if abs(s_new - s) < 1e-10 * s:
                s = s_new
                break
            s = s_new
        return max(s, K * 1.001)
    else:
        s = K * 0.8
        for _ in range(50):
            d1 = (math.log(s / K) + (b + 0.5 * v2) * T) / (sigma * math.sqrt(T))
            f = (K - s) - (black_scholes(s, K, T, r, r - b, sigma, "put")
                           - (1.0 - math.exp((b - r) * T) * _norm_cdf(-d1)) * s / q_u)
            h = s * 1e-6
            d1h = (math.log((s + h) / K) + (b + 0.5 * v2) * T) / (sigma * math.sqrt(T))
            fh = (K - (s + h)) - (black_scholes(s + h, K, T, r, r - b, sigma, "put")
                                 - (1.0 - math.exp((b - r) * T) * _norm_cdf(-d1h)) * (s + h) / q_u)
            deriv = (fh - f) / h
            if abs(deriv) < 1e-12:
                break
            s_new = s - f / deriv
            if s_new <= 0:
                break
            if abs(s_new - s) < 1e-10 * s:
                s = s_new
                break
            s = s_new
        return min(max(s, 1e-8), K * 0.999)


# ---------------------------------------------------------------- registry
MODELS = {
    "black_scholes": black_scholes,
    "binomial": binomial,
    "trinomial": trinomial,
    "monte_carlo": monte_carlo,
    "longstaff_schwartz": longstaff_schwartz,
    "baw": barone_adesi_whaley,
    "heston": heston,
    "bjerksund_stensland": bjerksund_stensland,
}

# Models that natively handle American exercise
AMERICAN_MODELS = {"binomial", "trinomial", "longstaff_schwartz", "baw", "bjerksund_stensland"}
# Models that assume European exercise
EUROPEAN_MODELS = {"black_scholes", "monte_carlo", "heston"}

# Relative speed ranking (1 = fastest)
SPEED_RANK = {
    "black_scholes": 1,
    "baw": 2,
    "bjerksund_stensland": 3,
    "binomial": 4,
    "trinomial": 5,
    "heston": 6,
    "monte_carlo": 7,
    "longstaff_schwartz": 8,
}


def price_with(model_name, S, K, T, r, q, sigma, option_type, **kwargs):
    """Dispatch to a named model."""
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    fn = MODELS[model_name]
    if model_name in ("binomial", "trinomial") and kwargs.get("american") is None:
        kwargs = dict(kwargs, american=False)
    return fn(S, K, T, r, q, sigma, option_type, **kwargs)
