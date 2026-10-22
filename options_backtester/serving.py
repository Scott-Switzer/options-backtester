"""Client for the Zion thin serving layer price-history contract.

The serving Worker (``zion-financial-serving-v2-thin-staging``) exposes
materialized CRSP price history at ``GET /history`` with query parameters
``symbol``, ``date_from``, ``date_to`` and ``limit`` (max 500), authorized
with a Bearer token. Each response page carries ``rows`` plus ``permno``,
``input_sha256`` (the source release ID), ``coverage`` and a ``next``
pagination pointer (``part`` / ``offset`` / ``expected_input_sha256``).
Rows include a raw observed price and split/distribution adjusted prices.

This module is standard library only, like the rest of the package. The
HTTP transport is injectable so tests never touch the network: pass any
callable ``transport(url, headers) -> (status, body_text)``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

DEFAULT_BASE_URL = "https://zion-financial-serving-v2-thin-staging.scswitzer.workers.dev"
HISTORY_PATH = "/history"
MAX_PAGES = 1000

# A transport maps (url, headers) to (http_status, response_body_text).
Transport = Callable[[str, dict], tuple[int, str]]


class ServingError(Exception):
    """The serving layer rejected a request or returned unusable data."""

    def __init__(self, message: str, status: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.code = code


def _number(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


@dataclass
class PriceRow:
    """One served daily price row."""

    date: str
    permno: Optional[int] = None
    raw_observed_price: Optional[float] = None
    split_adjusted_price: Optional[float] = None
    distribution_adjusted_price: Optional[float] = None
    source_total_return: Optional[float] = None
    source_return_ex_ordinary_dividends: Optional[float] = None
    missing_adjustment_factors: object = None

    @classmethod
    def from_json(cls, row: dict) -> "PriceRow":
        permno = row.get("permno")
        return cls(
            date=str(row.get("date")),
            permno=int(permno) if permno is not None else None,
            raw_observed_price=_number(row.get("raw_observed_price")),
            split_adjusted_price=_number(row.get("split_adjusted_price")),
            distribution_adjusted_price=_number(row.get("distribution_adjusted_price")),
            source_total_return=_number(row.get("source_total_return")),
            source_return_ex_ordinary_dividends=_number(
                row.get("source_return_ex_ordinary_dividends")
            ),
            missing_adjustment_factors=row.get("missing_adjustment_factors"),
        )

    def price(self, basis: str = "raw_observed_price") -> Optional[float]:
        """Price on the requested basis, falling back raw -> split adjusted."""
        value = getattr(self, basis, None)
        if value is not None:
            return value
        if basis != "raw_observed_price":
            return self.raw_observed_price
        return self.split_adjusted_price


@dataclass
class HistoryResult:
    """Aggregated served history for one symbol across all pages."""

    symbol: str
    permno: Optional[int] = None
    release: Optional[str] = None  # input_sha256 reported by the serving layer
    coverage: dict = field(default_factory=dict)
    rows: list[PriceRow] = field(default_factory=list)
    parts: Optional[int] = None
    source: dict = field(default_factory=dict)


def _urllib_transport(url: str, headers: dict) -> tuple[int, str]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


class ServingClient:
    """Read price history from the Zion thin serving layer.

    ``token`` is the Bearer token the serving layer expects. It is a Worker
    secret on the serving side; never hardcode one here. Without a token
    the server answers 401 and this client raises :class:`ServingError`.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        token: Optional[str] = None,
        transport: Optional[Transport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.transport = transport or _urllib_transport

    def _get_page(self, params: dict) -> dict:
        url = self.base_url + HISTORY_PATH + "?" + urllib.parse.urlencode(params)
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        status, body = self.transport(url, headers)
        if status != 200:
            code = None
            try:
                parsed = json.loads(body)
                error = parsed.get("error") if isinstance(parsed, dict) else None
                code = error if isinstance(error, str) else (error or {}).get("code")
            except (ValueError, AttributeError):
                pass
            raise ServingError(
                f"serving layer returned HTTP {status}" + (f" ({code})" if code else ""),
                status=status,
                code=code,
            )
        try:
            page = json.loads(body)
        except ValueError as error:
            raise ServingError(f"serving layer returned invalid JSON: {error}") from error
        if not isinstance(page, dict):
            raise ServingError("serving layer returned a non-object payload")
        return page

    def history(
        self,
        symbol: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 500,
    ) -> HistoryResult:
        """Full served history in the window, following ``next`` pagination."""
        params: dict = {"symbol": symbol.upper(), "limit": str(limit)}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        result = HistoryResult(symbol=params["symbol"])
        for _ in range(MAX_PAGES):
            page = self._get_page(params)
            release = page.get("input_sha256")
            if result.release is not None and release != result.release:
                raise ServingError("serving release changed mid-pagination")
            result.release = release
            if page.get("permno") is not None:
                result.permno = int(page["permno"])
            result.coverage = page.get("coverage") or result.coverage
            result.parts = page.get("parts", result.parts)
            result.source = page.get("source") or result.source
            result.rows.extend(PriceRow.from_json(r) for r in page.get("rows") or [])
            nxt = page.get("next")
            if not nxt:
                return result
            params["part"] = str(nxt["part"])
            params["offset"] = str(nxt["offset"])
            if nxt.get("expected_input_sha256"):
                params["expected_input_sha256"] = nxt["expected_input_sha256"]
        raise ServingError(f"pagination exceeded {MAX_PAGES} pages")

    def latest_on_or_before(self, symbol: str, date: Optional[str] = None) -> PriceRow:
        """Most recent served row on or before ``date`` (latest if omitted)."""
        result = self.history(symbol, date_to=date)
        if not result.rows:
            raise ServingError(
                f"no served price for {symbol.upper()} on or before {date or 'the latest date'}"
            )
        return max(result.rows, key=lambda r: r.date)
