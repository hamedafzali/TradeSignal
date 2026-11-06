import unittest

import numpy as np
import pandas as pd

import edge_stats
import research


class EdgeStatsTests(unittest.TestCase):
    def test_market_of(self):
        self.assertEqual(edge_stats.market_of("SAP.DE"), "eu")
        self.assertEqual(edge_stats.market_of("NOVO-B.CO"), "eu")
        self.assertEqual(edge_stats.market_of("BTC-USD"), "crypto")
        self.assertEqual(edge_stats.market_of("AAPL"), "us")

    def test_clustered_ci_wider_than_naive_when_days_correlated(self):
        rng = np.random.default_rng(0)
        vals, days = [], []
        for d in range(40):            # 40 days, 5 trades/day sharing a day shock
            shock = rng.normal(0, 1)
            for _ in range(5):
                vals.append(shock + rng.normal(0, 0.2))
                days.append(f"d{d}")
        clustered = edge_stats.clustered_mean_ci(vals, days)
        naive = edge_stats.clustered_mean_ci(vals, [str(i) for i in range(len(vals))])
        self.assertGreater(clustered["se"], 1.5 * naive["se"])

    def test_verdicts(self):
        days = [f"d{i % 50}" for i in range(200)]
        self.assertEqual(edge_stats.summarize_r([-0.2] * 100 + [0.0] * 100, days)["verdict"], "NO EDGE")
        self.assertEqual(edge_stats.summarize_r([0.5] * 150 + [0.3] * 50, days)["verdict"], "EDGE")
        self.assertEqual(edge_stats.summarize_r([1.0] * 5, ["a"] * 5)["verdict"], "INSUFFICIENT DATA")


class SimulateTests(unittest.TestCase):
    def _prep(self, highs, lows, closes, atr=1.0):
        n = len(closes)
        t = pd.date_range("2026-05-26 10:05", periods=n, freq="5min", tz="UTC")
        return {"symbol": "AAPL", "t": t, "close": np.array(closes, float),
                "high": np.array(highs, float), "low": np.array(lows, float),
                "atr": np.full(n, atr)}

    def _pad(self, p, extra):
        # extend data past the 24h deadline so the trade can resolve
        n = len(p["close"])
        t = pd.date_range(p["t"][0], periods=n + extra, freq="5min", tz="UTC")
        last = p["close"][-1]
        return {**p, "t": t,
                "close": np.r_[p["close"], np.full(extra, last)],
                "high": np.r_[p["high"], np.full(extra, last)],
                "low": np.r_[p["low"], np.full(extra, last)],
                "atr": np.r_[p["atr"], np.full(extra, 1.0)]}

    def test_buy_tp_first(self):
        p = self._pad(self._prep([100, 101, 103], [100, 99.6, 101], [100, 100.5, 102.6]), 300)
        tr = research.simulate(p, [(0, 1)], sl_mult=1.0, tp_mult=2.5)
        self.assertEqual(tr[0]["outcome"], "tp")
        self.assertAlmostEqual(tr[0]["r"], 2.5)

    def test_sell_sl_first(self):
        p = self._pad(self._prep([100, 101.2], [100, 99.9], [100, 101]), 300)
        tr = research.simulate(p, [(0, -1)], sl_mult=1.0, tp_mult=2.5)
        self.assertEqual(tr[0]["outcome"], "sl")
        self.assertAlmostEqual(tr[0]["r"], -1.0)

    def test_same_bar_both_hit_exits_at_close(self):
        p = self._pad(self._prep([100, 104], [100, 98], [100, 100.4]), 300)
        tr = research.simulate(p, [(0, 1)], sl_mult=1.0, tp_mult=2.5)
        self.assertEqual(tr[0]["outcome"], "both")
        self.assertAlmostEqual(tr[0]["r"], 0.4)

    def test_timeout_at_last_bar_before_deadline(self):
        p = self._pad(self._prep([100, 100.5], [100, 99.5], [100, 100.3]), 300)
        tr = research.simulate(p, [(0, 1)], sl_mult=1.0, tp_mult=2.5)
        self.assertEqual(tr[0]["outcome"], "timeout")
        self.assertAlmostEqual(tr[0]["r"], 0.3)

    def test_unresolvable_trade_dropped(self):
        p = self._prep([100, 100.5], [100, 99.5], [100, 100.3])
        self.assertEqual(research.simulate(p, [(0, 1)], 1.0, 2.5), [])

    def test_cost_in_r(self):
        trades = [{"symbol": "SAP.DE", "t": pd.Timestamp("2026-05-26", tz="UTC"),
                   "dir": 1, "outcome": "tp", "r": 1.0, "sl_pct": 0.5}]
        # EU cost 0.10% with a 0.5% stop = 0.2R
        self.assertAlmostEqual(research.score(trades)["avg_r"], 0.8)


if __name__ == "__main__":
    unittest.main()
