/**
 * options-backtester Cloudflare Worker
 *
 * HTTP API for options pricing models. Dependency-free.
 *
 * POST /price  - price with a specific model
 *   {"model": "black_scholes"|"binomial"|"baw", "S": 100, "K": 100,
 *    "T": 1.0, "r": 0.05, "q": 0.0, "sigma": 0.2, "option_type": "call",
 *    "steps": 200, "american": false}
 *
 * POST /blend  - smart model selection
 *   {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "q": 0.0, "sigma": 0.2,
 *    "option_type": "call", "style": "european"|"american",
 *    "speed": 0.5, "blend_mode": false}
 *
 * GET /models  - list available models
 */

// ---------- math helpers ----------
function normCdf(x) {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}
function erf(x) {
  // Abramowitz & Stegun approximation
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}
function normPdf(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

// ---------- Black-Scholes ----------
function blackScholes(S, K, T, r, q, sigma, optionType) {
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  const disc = Math.exp(-r * T);
  const fwd = S * Math.exp(-q * T);
  if (optionType === "call") return fwd * normCdf(d1) - K * disc * normCdf(d2);
  return K * disc * normCdf(-d2) - fwd * normCdf(-d1);
}
function bsGreeks(S, K, T, r, q, sigma, optionType) {
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  const disc = Math.exp(-r * T), div = Math.exp(-q * T), sqrtT = Math.sqrt(T);
  const delta = optionType === "call" ? div * normCdf(d1) : div * (normCdf(d1) - 1);
  const gamma = div * normPdf(d1) / (S * sigma * sqrtT);
  const timeTerm = -(S * div * normPdf(d1) * sigma) / (2 * sqrtT);
  let thetaA, rho;
  if (optionType === "call") {
    thetaA = timeTerm - r * K * disc * normCdf(d2) + q * S * div * normCdf(d1);
    rho = K * T * disc * normCdf(d2);
  } else {
    thetaA = timeTerm + r * K * disc * normCdf(-d2) - q * S * div * normCdf(-d1);
    rho = -K * T * disc * normCdf(-d2);
  }
  return { delta, gamma, theta: thetaA / 365, vega: S * div * normPdf(d1) * sqrtT, rho };
}

// ---------- Binomial (CRR) ----------
function binomial(S, K, T, r, q, sigma, optionType, steps = 200, american = false) {
  const dt = T / steps;
  const u = Math.exp(sigma * Math.sqrt(dt));
  const d = 1 / u;
  const disc = Math.exp(-r * dt);
  const p = (Math.exp((r - q) * dt) - d) / (u - d);
  const payoff = (px) => optionType === "call" ? Math.max(px - K, 0) : Math.max(K - px, 0);
  let vals = [];
  for (let j = 0; j <= steps; j++) vals[j] = payoff(S * Math.pow(u, j) * Math.pow(d, steps - j));
  for (let i = steps - 1; i >= 0; i--) {
    for (let j = 0; j <= i; j++) {
      const hold = disc * (p * vals[j + 1] + (1 - p) * vals[j]);
      vals[j] = american ? Math.max(hold, payoff(S * Math.pow(u, j) * Math.pow(d, i - j))) : hold;
    }
  }
  return vals[0];
}

// ---------- Barone-Adesi-Whaley ----------
function bawCritical(S, K, T, r, q, sigma, m, isCall) {
  let s = isCall ? K * 1.5 : K * 0.5;
  const euro = (x) => blackScholes(x, K, T, r, q, sigma, isCall ? "call" : "put");
  for (let iter = 0; iter < 100; iter++) {
    const d1 = (Math.log(s / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const n_d1 = isCall ? normCdf(d1) : normCdf(-d1);
    const lhs = isCall ? s - K : K - s;
    const rhs = euro(s) + (isCall ? 1 : -1) * (1 - Math.exp(-q * T) * n_d1) * s / m;
    const h = s * 1e-6;
    const d1h = (Math.log((s + h) / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const n_d1h = isCall ? normCdf(d1h) : normCdf(-d1h);
    const lhsh = isCall ? (s + h) - K : K - (s + h);
    const rhsh = euro(s + h) + (isCall ? 1 : -1) * (1 - Math.exp(-q * T) * n_d1h) * (s + h) / m;
    const deriv = ((lhsh - rhsh) - (lhs - rhs)) / h;
    if (Math.abs(deriv) < 1e-12) break;
    const sNew = s - (lhs - rhs) / deriv;
    if (sNew <= 0) break;
    if (Math.abs(sNew - s) < 1e-10 * s) { s = sNew; break; }
    s = sNew;
  }
  return s;
}
function baw(S, K, T, r, q, sigma, optionType) {
  const euro = blackScholes(S, K, T, r, q, sigma, optionType);
  if (optionType === "call" && q <= 0) return euro;
  const n = 2 * (r - q) / (sigma * sigma);
  const k = 2 * r / (sigma * sigma) / (1 - Math.exp(-r * T));
  const disc = Math.sqrt((n - 1) * (n - 1) + 4 * k);
  const isCall = optionType === "call";
  const m = 0.5 * (-(n - 1) + (isCall ? disc : -disc));
  const sStar = bawCritical(S, K, T, r, q, sigma, m, isCall);
  if (isCall) {
    if (S >= sStar) return S - K;
    const d1 = (Math.log(sStar / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const a2 = (sStar / m) * (1 - Math.exp(-q * T) * normCdf(d1));
    return euro + a2 * Math.pow(S / sStar, m);
  }
  if (S <= sStar) return K - S;
  const d1 = (Math.log(sStar / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const a1 = -(sStar / m) * (1 - Math.exp(-q * T) * normCdf(-d1));
  return euro + a1 * Math.pow(S / sStar, m);
}

// ---------- blender ----------
function selectModel(style, T, speed) {
  if (style === "american") {
    if (T < 0.08) return ["binomial", "Very short-dated American: binomial tree captures discrete steps."];
    if (speed >= 0.7) return ["baw", "American, speed prioritized: Barone-Adesi-Whaley analytic approximation."];
    return ["baw", "American, balanced: Barone-Adesi-Whaley with good speed."];
  }
  if (speed >= 0.8) return ["black_scholes", "European, speed prioritized: Black-Scholes closed form."];
  return ["black_scholes", "European: Black-Scholes closed form is the standard choice."];
}
function priceWith(model, p) {
  const { S, K, T, r, q = 0, sigma, option_type = "call" } = p;
  if (model === "black_scholes") return blackScholes(S, K, T, r, q, sigma, option_type);
  if (model === "binomial") return binomial(S, K, T, r, q, sigma, option_type, p.steps || 200, !!p.american);
  if (model === "baw") return baw(S, K, T, r, q, sigma, option_type);
  throw new Error("unknown model: " + model);
}

// ---------- request handling ----------
async function handlePrice(body) {
  const price = priceWith(body.model || "black_scholes", body);
  const out = { model: body.model || "black_scholes", price };
  if ((body.model || "black_scholes") === "black_scholes") {
    out.greeks = bsGreeks(body.S, body.K, body.T, body.r, body.q || 0, body.sigma, body.option_type || "call");
  }
  return out;
}
async function handleBlend(body) {
  const style = (body.style || "european").toLowerCase();
  const [model, reasoning] = selectModel(style, body.T || 1, body.speed ?? 0.5);
  const p = { ...body, american: style === "american" };
  return { model, price: priceWith(model, p), reasoning };
}

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const cors = { "Access-Control-Allow-Origin": "*", "Content-Type": "application/json" };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });
    try {
      if (url.pathname === "/models" && request.method === "GET") {
        return Response.json({ models: ["black_scholes", "binomial", "baw"], note: "Full 8-model library runs in Python; worker serves the fast subset." }, { headers: cors });
      }
      if (url.pathname === "/price" && request.method === "POST") {
        return Response.json(await handlePrice(await request.json()), { headers: cors });
      }
      if (url.pathname === "/blend" && request.method === "POST") {
        return Response.json(await handleBlend(await request.json()), { headers: cors });
      }
      return Response.json({ error: "Use POST /price, POST /blend, or GET /models" }, { status: 404, headers: cors });
    } catch (e) {
      return Response.json({ error: e.message }, { status: 400, headers: cors });
    }
  }
};
