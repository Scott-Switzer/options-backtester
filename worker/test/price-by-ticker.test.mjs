/**
 * Tests for POST /price-by-ticker and the untouched legacy routes.
 * The Zion serving binding is faked; nothing touches the network.
 * Run: node --test test/   (from the worker/ directory)
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import worker from "../worker.js";

const RELEASE = "f".repeat(64);
const COVERAGE = {
  security_id: "CRSP:PERMNO:14593",
  permno: 14593,
  date_min: "2025-01-01",
  date_max: "2025-09-26",
  rows: 40,
  parts: 1,
};

function row(date, close) {
  return {
    date,
    permno: 14593,
    raw_observed_price: close,
    split_adjusted_price: close,
    distribution_adjusted_price: close,
    source_total_return: 0.01,
    source_return_ex_ordinary_dividends: 0.009,
    missing_adjustment_factors: 0,
  };
}

function makeRows(endDate, closes) {
  const rows = [];
  const end = Date.parse(endDate + "T00:00:00Z");
  closes.forEach((close, i) => {
    const d = new Date(end - (closes.length - 1 - i) * 86400000);
    rows.push(row(d.toISOString().slice(0, 10), close));
  });
  return rows;
}

function page(rows, next = null) {
  return {
    symbol: "AAPL",
    permno: 14593,
    input_sha256: RELEASE,
    coverage: COVERAGE,
    parts: 1,
    rows,
    next,
  };
}

// A fake ZION_FINANCIAL_QUERY binding serving `allRows` with window filters.
function makeBinding(allRows, requests) {
  return {
    async fetch(request) {
      const url = new URL(request.url);
      requests.push({ url, auth: request.headers.get("Authorization") });
      if (url.searchParams.get("metadata_only") === "true") {
        return Response.json({ ...page([]), rows: [] });
      }
      const from = url.searchParams.get("date_from");
      const to = url.searchParams.get("date_to");
      const rows = allRows.filter((r) => (!from || r.date >= from) && (!to || r.date <= to));
      return Response.json(page(rows));
    },
  };
}

function makeEnv(binding) {
  return { ZION_FINANCIAL_QUERY: binding, ZION_SERVING_TOKEN: "test-token" };
}

function post(path, body, env) {
  return worker.fetch(
    new Request("https://options-backtester.test" + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
    env
  );
}

const DRIFT = Array.from({ length: 40 }, (_, i) => 240 + i * 0.4 + (i % 3));

test("fails closed with 503 when the binding is missing", async () => {
  const resp = await post("/price-by-ticker", { ticker: "AAPL", strike: 250 }, {});
  assert.equal(resp.status, 503);
  const body = await resp.json();
  assert.equal(body.error, "SERVING_NOT_CONFIGURED");
});

test("fails closed with 503 when the token secret is missing", async () => {
  const env = { ZION_FINANCIAL_QUERY: makeBinding([], []) };
  const resp = await post("/price-by-ticker", { ticker: "AAPL", strike: 250 }, env);
  assert.equal(resp.status, 503);
  assert.equal((await resp.json()).error, "SERVING_NOT_CONFIGURED");
});

test("prices from the served spot with provenance and price parity", async () => {
  const rows = makeRows("2025-09-26", DRIFT.map((v, i) => (i === 39 ? 255.46 : v)));
  const requests = [];
  const env = makeEnv(makeBinding(rows, requests));
  const resp = await post("/price-by-ticker", {
    ticker: "AAPL", date: "2025-09-26", strike: 250,
    years_to_expiry: 0.5, sigma: 0.25, type: "call", model: "black_scholes",
  }, env);
  assert.equal(resp.status, 200);
  const body = await resp.json();
  assert.equal(body.provenance.spot, 255.46);
  assert.equal(body.provenance.price_date, "2025-09-26");
  assert.equal(body.provenance.release, RELEASE);
  assert.equal(body.provenance.permno, 14593);
  assert.equal(body.provenance.provider, "CRSP");
  assert.equal(body.provenance.basis, "raw_observed_price");
  assert.equal(body.sigma_source, "provided");
  assert.ok(body.greeks, "black_scholes parity includes greeks");
  assert.ok(requests.every((r) => r.auth === "Bearer test-token"));
  const legacy = await post("/price", {
    model: "black_scholes", S: 255.46, K: 250, T: 0.5, r: 0.05, q: 0,
    sigma: 0.25, option_type: "call",
  }, {});
  const legacyBody = await legacy.json();
  assert.ok(Math.abs(body.price - legacyBody.price) < 1e-9);
});

test("defaults to the latest served session", async () => {
  const rows = makeRows("2025-09-26", DRIFT);
  const requests = [];
  const env = makeEnv(makeBinding(rows, requests));
  const resp = await post("/price-by-ticker", {
    ticker: "AAPL", strike: 250, years_to_expiry: 0.5, sigma: 0.25,
  }, env);
  assert.equal(resp.status, 200);
  const body = await resp.json();
  assert.equal(body.provenance.price_date, "2025-09-26");
  assert.equal(requests[0].url.searchParams.get("metadata_only"), "true");
});

test("computes realized vol from served history when sigma is omitted", async () => {
  const rows = makeRows("2025-09-26", DRIFT);
  const env = makeEnv(makeBinding(rows, []));
  const resp = await post("/price-by-ticker", {
    ticker: "AAPL", strike: 250, years_to_expiry: 0.5, type: "put",
  }, env);
  assert.equal(resp.status, 200);
  const body = await resp.json();
  assert.equal(body.sigma_source, "realized_vol_from_served_history");
  assert.ok(body.realized_vol_from_served_history > 0);
  assert.equal(body.sigma, body.realized_vol_from_served_history);
  assert.ok(body.price >= 0);
  assert.ok(body.model, "default blend behavior picks a model");
});

test("derives years to expiry from an expiry date", async () => {
  const rows = makeRows("2025-09-26", DRIFT);
  const env = makeEnv(makeBinding(rows, []));
  const resp = await post("/price-by-ticker", {
    ticker: "AAPL", date: "2025-09-26", strike: 250, expiry: "2026-09-26", sigma: 0.25,
  }, env);
  assert.equal(resp.status, 200);
  const body = await resp.json();
  assert.ok(Math.abs(body.years_to_expiry - 1.0) < 0.01);
});

test("follows serving pagination", async () => {
  const rows = makeRows("2025-09-26", DRIFT);
  const seen = [];
  const binding = {
    async fetch(request) {
      const url = new URL(request.url);
      seen.push(url);
      if (url.searchParams.get("metadata_only") === "true") {
        return Response.json({ ...page([]), rows: [] });
      }
      if (!url.searchParams.has("offset")) {
        return Response.json(page(rows.slice(-3, -1),
          { part: 0, offset: 2, expected_input_sha256: RELEASE }));
      }
      return Response.json(page(rows.slice(-1)));
    },
  };
  const resp = await post("/price-by-ticker", {
    ticker: "AAPL", strike: 250, years_to_expiry: 0.5, sigma: 0.25,
  }, makeEnv(binding));
  assert.equal(resp.status, 200);
  const body = await resp.json();
  assert.equal(body.provenance.spot, rows[rows.length - 1].raw_observed_price);
  const paged = seen.find((u) => u.searchParams.get("offset") === "2");
  assert.ok(paged, "second page was requested");
  assert.equal(paged.searchParams.get("expected_input_sha256"), RELEASE);
});

test("maps an upstream auth failure to 502", async () => {
  const binding = {
    async fetch() {
      return Response.json({ error: "AUTH_REQUIRED" }, { status: 401 });
    },
  };
  const resp = await post("/price-by-ticker", { ticker: "AAPL", strike: 250 }, makeEnv(binding));
  assert.equal(resp.status, 502);
  assert.equal((await resp.json()).error, "SERVING_UPSTREAM_ERROR");
});

test("legacy routes behave as before", async () => {
  const models = await worker.fetch(new Request("https://x.test/models"), {});
  assert.deepEqual((await models.json()).models, ["black_scholes", "binomial", "baw"]);
  const priced = await post("/price", {
    model: "black_scholes", S: 100, K: 100, T: 1, r: 0.05, sigma: 0.2, option_type: "call",
  }, {});
  assert.ok(Math.abs((await priced.json()).price - 10.4506) < 0.01);
  const blended = await post("/blend", {
    S: 100, K: 100, T: 1, r: 0.05, sigma: 0.2, option_type: "call",
  }, {});
  assert.equal((await blended.json()).model, "black_scholes");
  const landing = await worker.fetch(new Request("https://x.test/"), {});
  assert.equal(landing.status, 200);
  assert.match(await landing.text(), /Options Pricer/);
});
