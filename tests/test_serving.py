"""Tests for the Zion serving-layer client, using a fake transport."""

import json
import unittest
import urllib.parse

from options_backtester.serving import ServingClient, ServingError

RELEASE = "f" * 64


def make_row(date, raw, split=None):
    return {
        "date": date,
        "permno": 14593,
        "raw_observed_price": raw,
        "split_adjusted_price": raw if split is None else split,
        "distribution_adjusted_price": raw,
        "source_total_return": 0.01,
        "source_return_ex_ordinary_dividends": 0.009,
        "missing_adjustment_factors": 0,
    }


def make_page(rows, nxt=None, release=RELEASE):
    return {
        "symbol": "AAPL",
        "permno": 14593,
        "input_sha256": release,
        "coverage": {
            "security_id": "CRSP:PERMNO:14593",
            "permno": 14593,
            "date_min": "1980-12-12",
            "date_max": "2025-12-31",
            "rows": 11355,
            "parts": 2,
        },
        "parts": 2,
        "source": {"key": "prices/0.json", "sha256": "a" * 64},
        "rows": rows,
        "next": nxt,
    }


class FakeTransport:
    """Returns queued (status, payload) responses and records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, headers):
        self.requests.append((url, headers))
        status, payload = self.responses.pop(0)
        body = payload if isinstance(payload, str) else json.dumps(payload)
        return status, body


class TestServingClient(unittest.TestCase):
    def test_parses_rows_and_propagates_release(self):
        transport = FakeTransport([(200, make_page([make_row("2025-09-26", 255.46)]))])
        client = ServingClient(token="test-token", transport=transport)
        result = client.history("AAPL", date_to="2025-09-26")
        self.assertEqual(result.release, RELEASE)
        self.assertEqual(result.permno, 14593)
        self.assertEqual(result.coverage["date_max"], "2025-12-31")
        self.assertEqual(len(result.rows), 1)
        row = result.rows[0]
        self.assertEqual(row.date, "2025-09-26")
        self.assertEqual(row.raw_observed_price, 255.46)
        self.assertEqual(row.split_adjusted_price, 255.46)
        self.assertEqual(row.price(), 255.46)

    def test_sends_bearer_token_and_query_params(self):
        transport = FakeTransport([(200, make_page([]))])
        client = ServingClient(token="test-token", transport=transport)
        client.history("aapl", date_from="2025-01-01", date_to="2025-09-26", limit=250)
        url, headers = transport.requests[0]
        self.assertEqual(headers["Authorization"], "Bearer test-token")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        self.assertEqual(query["symbol"], ["AAPL"])
        self.assertEqual(query["date_from"], ["2025-01-01"])
        self.assertEqual(query["date_to"], ["2025-09-26"])
        self.assertEqual(query["limit"], ["250"])

    def test_follows_pagination_within_part(self):
        page1 = make_page(
            [make_row("2025-09-24", 250.0), make_row("2025-09-25", 252.0)],
            nxt={"part": 0, "offset": 2, "expected_input_sha256": RELEASE},
        )
        page2 = make_page([make_row("2025-09-26", 255.46)])
        transport = FakeTransport([(200, page1), (200, page2)])
        client = ServingClient(token="t", transport=transport)
        result = client.history("AAPL")
        self.assertEqual([r.date for r in result.rows],
                         ["2025-09-24", "2025-09-25", "2025-09-26"])
        second = urllib.parse.parse_qs(urllib.parse.urlsplit(transport.requests[1][0]).query)
        self.assertEqual(second["part"], ["0"])
        self.assertEqual(second["offset"], ["2"])
        self.assertEqual(second["expected_input_sha256"], [RELEASE])

    def test_follows_pagination_across_parts(self):
        page1 = make_page([make_row("2025-09-25", 252.0)],
                          nxt={"part": 1, "offset": 0, "expected_input_sha256": RELEASE})
        page2 = make_page([make_row("2025-09-26", 255.46)])
        transport = FakeTransport([(200, page1), (200, page2)])
        client = ServingClient(token="t", transport=transport)
        result = client.history("AAPL")
        self.assertEqual(len(result.rows), 2)
        second = urllib.parse.parse_qs(urllib.parse.urlsplit(transport.requests[1][0]).query)
        self.assertEqual(second["part"], ["1"])
        self.assertEqual(second["offset"], ["0"])

    def test_missing_price_fields_become_none(self):
        row = {"date": "2025-09-26", "permno": 14593, "raw_observed_price": None,
               "split_adjusted_price": 255.46}
        transport = FakeTransport([(200, make_page([row]))])
        client = ServingClient(token="t", transport=transport)
        parsed = client.history("AAPL").rows[0]
        self.assertIsNone(parsed.raw_observed_price)
        # price() falls back to the split-adjusted basis when raw is missing.
        self.assertEqual(parsed.price(), 255.46)

    def test_error_on_non_200(self):
        transport = FakeTransport([(401, {"error": "AUTH_REQUIRED"})])
        client = ServingClient(token="bad", transport=transport)
        with self.assertRaises(ServingError) as ctx:
            client.history("AAPL")
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(ctx.exception.code, "AUTH_REQUIRED")

    def test_error_on_plain_text_non_200(self):
        transport = FakeTransport([(503, "Unauthorized")])
        client = ServingClient(transport=transport)
        with self.assertRaises(ServingError) as ctx:
            client.history("AAPL")
        self.assertEqual(ctx.exception.status, 503)

    def test_release_change_mid_pagination_raises(self):
        page1 = make_page([make_row("2025-09-25", 252.0)],
                          nxt={"part": 1, "offset": 0, "expected_input_sha256": RELEASE})
        page2 = make_page([make_row("2025-09-26", 255.46)], release="e" * 64)
        transport = FakeTransport([(200, page1), (200, page2)])
        client = ServingClient(token="t", transport=transport)
        with self.assertRaises(ServingError):
            client.history("AAPL")

    def test_latest_on_or_before_picks_newest_row(self):
        page = make_page([make_row("2025-09-24", 250.0),
                          make_row("2025-09-26", 255.46),
                          make_row("2025-09-25", 252.0)])
        transport = FakeTransport([(200, page)])
        client = ServingClient(token="t", transport=transport)
        row = client.latest_on_or_before("AAPL", "2025-09-26")
        self.assertEqual(row.date, "2025-09-26")
        self.assertEqual(row.raw_observed_price, 255.46)

    def test_latest_on_or_before_empty_raises(self):
        transport = FakeTransport([(200, make_page([]))])
        client = ServingClient(token="t", transport=transport)
        with self.assertRaises(ServingError):
            client.latest_on_or_before("AAPL", "2025-09-26")


if __name__ == "__main__":
    unittest.main()
