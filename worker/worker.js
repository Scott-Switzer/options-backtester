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
 * POST /price-by-ticker - price from served underlying history
 *   {"ticker": "AAPL", "date": "2025-09-26", "strike": 250,
 *    "years_to_expiry": 0.5, "sigma": 0.25, "type": "call"}
 *   Fetches the spot from the Zion thin serving layer through the
 *   ZION_FINANCIAL_QUERY service binding, authorized with the
 *   ZION_SERVING_TOKEN secret. "date" defaults to the latest served
 *   session; "expiry" (YYYY-MM-DD) may replace "years_to_expiry";
 *   "sigma" may be omitted, in which case 30-day realized volatility is
 *   computed from the served history. Without the binding or the secret
 *   the route fails closed with HTTP 503 SERVING_NOT_CONFIGURED.
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

// ---------- served history (Zion thin serving layer) ----------
const SERVING_NAME = "zion-financial-serving-v2-thin-staging";
const SERVING_MAX_PAGES = 25;

function httpError(status, code, message) {
  const e = new Error(message || code);
  e.status = status;
  e.code = code;
  return e;
}
function shiftDate(iso, days) {
  const d = new Date(iso + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}
async function fetchServingPage(env, params) {
  const u = new URL("https://internal/history");
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) u.searchParams.set(k, String(v));
  }
  const resp = await env.ZION_FINANCIAL_QUERY.fetch(
    new Request(u.toString(), { headers: { Authorization: "Bearer " + env.ZION_SERVING_TOKEN } })
  );
  const text = await resp.text();
  let page = null;
  try { page = JSON.parse(text); } catch { /* keep null */ }
  if (resp.status !== 200) {
    const code = page && typeof page.error === "string" ? page.error : null;
    throw httpError(502, "SERVING_UPSTREAM_ERROR",
      "Serving layer returned HTTP " + resp.status + (code ? " (" + code + ")" : ""));
  }
  if (!page || !Array.isArray(page.rows)) {
    throw httpError(502, "SERVING_BAD_RESPONSE", "Serving layer returned an unusable history page.");
  }
  return page;
}
async function fetchServingHistory(env, params) {
  const query = { limit: 500, ...params };
  let release = null;
  const allRows = [];
  for (let i = 0; i < SERVING_MAX_PAGES; i++) {
    const page = await fetchServingPage(env, query);
    if (release !== null && page.input_sha256 !== release) {
      throw httpError(502, "SERVING_RELEASE_CHANGED", "Serving release changed mid-pagination.");
    }
    release = page.input_sha256;
    allRows.push(...page.rows);
    if (!page.next) return { ...page, rows: allRows };
    query.part = page.next.part;
    query.offset = page.next.offset;
    if (page.next.expected_input_sha256) {
      query.expected_input_sha256 = page.next.expected_input_sha256;
    }
  }
  throw httpError(502, "SERVING_PAGINATION_BUDGET", "Serving history pagination exceeded the page budget.");
}
function servedSpot(row) {
  const raw = row.raw_observed_price;
  if (typeof raw === "number" && Number.isFinite(raw) && raw > 0) {
    return { spot: raw, basis: "raw_observed_price" };
  }
  const adj = row.split_adjusted_price;
  if (typeof adj === "number" && Number.isFinite(adj) && adj > 0) {
    return { spot: adj, basis: "split_adjusted_price" };
  }
  return null;
}
function realizedVol(rows, basis) {
  const closes = rows.map((row) => row[basis])
    .filter((v) => typeof v === "number" && Number.isFinite(v) && v > 0)
    .slice(-31);
  if (closes.length < 11) return null;
  const returns = [];
  for (let i = 1; i < closes.length; i++) returns.push(Math.log(closes[i] / closes[i - 1]));
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (returns.length - 1);
  return Math.sqrt(variance) * Math.sqrt(252);
}

async function handlePriceByTicker(body, env) {
  if (!env || !env.ZION_FINANCIAL_QUERY || !env.ZION_SERVING_TOKEN) {
    throw httpError(503, "SERVING_NOT_CONFIGURED",
      "POST /price-by-ticker needs the ZION_FINANCIAL_QUERY service binding and the ZION_SERVING_TOKEN secret on this Worker. Neither is set here, so no served price was fetched.");
  }
  const ticker = String(body.ticker || "").toUpperCase();
  if (!/^[A-Z][A-Z0-9.-]{0,24}$/.test(ticker)) {
    throw httpError(400, "INVALID_TICKER", "A valid ticker is required, for example \"AAPL\".");
  }
  const K = body.strike ?? body.K;
  if (typeof K !== "number" || !Number.isFinite(K) || K <= 0) {
    throw httpError(400, "INVALID_STRIKE", "strike must be a positive number.");
  }
  const optionType = String(body.type || body.option_type || "call").toLowerCase();
  if (optionType !== "call" && optionType !== "put") {
    throw httpError(400, "INVALID_TYPE", "type must be \"call\" or \"put\".");
  }
  if (body.date !== undefined && !/^\d{4}-\d{2}-\d{2}$/.test(body.date)) {
    throw httpError(400, "INVALID_DATE", "date must be YYYY-MM-DD.");
  }
  if (body.expiry !== undefined && !/^\d{4}-\d{2}-\d{2}$/.test(body.expiry)) {
    throw httpError(400, "INVALID_EXPIRY", "expiry must be YYYY-MM-DD.");
  }
  const meta = await fetchServingPage(env, { symbol: ticker, metadata_only: "true", limit: 1 });
  const coverage = meta.coverage || {};
  if (!coverage.date_max) {
    throw httpError(502, "SERVING_NO_COVERAGE", "Serving layer reported no price coverage for " + ticker + ".");
  }
  const endDate = body.date && body.date < coverage.date_max ? body.date : coverage.date_max;
  const startDate = shiftDate(endDate, -400) > (coverage.date_min || "") ? shiftDate(endDate, -400) : coverage.date_min;
  const history = await fetchServingHistory(env, { symbol: ticker, date_from: startDate, date_to: endDate });
  const rows = [...history.rows].sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  if (!rows.length) {
    throw httpError(404, "NO_SERVED_PRICE", "No served price for " + ticker + " on or before " + endDate + ".");
  }
  const spotRow = rows[rows.length - 1];
  const spotInfo = servedSpot(spotRow);
  if (!spotInfo) {
    throw httpError(502, "SERVED_PRICE_MISSING", "Served row for " + ticker + " on " + spotRow.date + " carries no usable price.");
  }
  const { spot, basis } = spotInfo;
  const priceDate = spotRow.date;
  let T;
  if (typeof body.years_to_expiry === "number") T = body.years_to_expiry;
  else if (body.expiry) T = (Date.parse(body.expiry + "T00:00:00Z") - Date.parse(priceDate + "T00:00:00Z")) / (365 * 86400000);
  else T = 1.0;
  if (!Number.isFinite(T) || T <= 0) {
    throw httpError(400, "INVALID_EXPIRY", "Expiry must fall after the served price date " + priceDate + ".");
  }
  const r = body.r ?? body.rate ?? 0.05;
  const q = body.q ?? body.dividend_yield ?? 0;
  let sigma = body.sigma;
  let sigmaSource = "provided";
  let realized = null;
  if (typeof sigma !== "number" || !Number.isFinite(sigma) || sigma <= 0) {
    realized = realizedVol(rows, basis);
    if (realized === null || !(realized > 0)) {
      throw httpError(422, "INSUFFICIENT_SERVED_HISTORY",
        "sigma was not provided and the served history is too short or too flat to compute realized volatility.");
    }
    sigma = realized;
    sigmaSource = "realized_vol_from_served_history";
  }
  const provenance = {
    provider: "CRSP",
    serving: SERVING_NAME,
    release: history.input_sha256 || null,
    permno: history.permno ?? spotRow.permno ?? null,
    price_date: priceDate,
    spot,
    basis,
  };
  const priced = { S: spot, K, T, r, q, sigma, option_type: optionType, steps: body.steps || 200 };
  const out = { ticker, strike: K, years_to_expiry: T, option_type: optionType, sigma, sigma_source: sigmaSource, provenance };
  if (realized !== null) out.realized_vol_from_served_history = realized;
  if (body.model) {
    if (!["black_scholes", "binomial", "baw"].includes(body.model)) {
      throw httpError(400, "UNKNOWN_MODEL", "model must be one of black_scholes, binomial, baw.");
    }
    const style = String(body.style || "european").toLowerCase();
    out.model = body.model;
    out.price = priceWith(body.model, { ...priced, american: !!body.american || style === "american" });
    if (body.model === "black_scholes") out.greeks = bsGreeks(spot, K, T, r, q, sigma, optionType);
  } else {
    const blended = await handleBlend({ ...body, ...priced });
    out.model = blended.model;
    out.price = blended.price;
    out.reasoning = blended.reasoning;
  }
  return out;
}

const LANDING_HTML = `<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Options Pricer</title><style>body{font-family:system-ui;max-width:480px;margin:20px auto;padding:0 16px}input,select{width:100%;padding:8px;margin:4px 0;box-sizing:border-box}button{background:#06c;color:#fff;border:none;padding:12px;width:100%;font-size:16px;border-radius:6px}pre{background:#f4f4f4;padding:12px;border-radius:6px;overflow-x:auto}</style></head><body><h2>Options Pricer</h2><label>Spot <input id="S" type="number" value="100"></label><label>Strike <input id="K" type="number" value="100"></label><label>Years to expiry <input id="T" type="number" step="0.01" value="1"></label><label>Rate <input id="r" type="number" step="0.01" value="0.05"></label><label>Vol <input id="s" type="number" step="0.01" value="0.2"></label><label>Type <select id="t"><option value="call">Call</option><option value="put">Put</option></select></label><label>Style <select id="st"><option value="european">European</option><option value="american">American</option></select></label><button onclick="go()">Price it</button><pre id="out">Enter values and tap Price it.</pre><script>async function go(){const b={S:+document.getElementById("S").value,K:+document.getElementById("K").value,T:+document.getElementById("T").value,r:+document.getElementById("r").value,sigma:+document.getElementById("s").value,option_type:document.getElementById("t").value,style:document.getElementById("st").value};const x=await(await fetch("/blend",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)})).json();document.getElementById("out").textContent=JSON.stringify(x,null,2)}</script></body></html>
`;

export default {
  async fetch(request, env) {
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
      if (url.pathname === "/price-by-ticker" && request.method === "POST") {
        try {
          return Response.json(await handlePriceByTicker(await request.json(), env), { headers: cors });
        } catch (e) {
          const out = { error: e.code || e.message };
          if (e.code) out.message = e.message;
          return Response.json(out, { status: e.status || 400, headers: cors });
        }
      }
      if (url.pathname === "/" && request.method === "GET") {
        return new Response(LANDING_HTML, { headers: { "Content-Type": "text/html" } });
      }
      return Response.json({ error: "Use POST /price, POST /blend, or GET /models" }, { status: 404, headers: cors });
    } catch (e) {
      return Response.json({ error: e.message }, { status: 400, headers: cors });
    }
  }
};
