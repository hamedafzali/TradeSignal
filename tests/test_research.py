import unittest

import numpy as np
import pandas as pd

import os
import tempfile

import edge_stats
import research
import research_daily as rd
import strategy_gate as gate


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


class AlpacaParsingTests(unittest.TestCase):
    def test_parse_and_regular_session(self):
        bars = [  # 2026-03-10 (EDT): 13:25Z = 09:25 ET pre-market, 13:30Z = open
            {"t": "2026-03-10T13:25:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 5},
            {"t": "2026-03-10T13:30:00Z", "o": 2, "h": 3, "l": 1.5, "c": 2.5, "v": 10},
            {"t": "2026-03-10T19:55:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 7},
            {"t": "2026-03-10T20:00:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 7},
        ]
        df = research.regular_session(research.parse_alpaca_bars(bars))
        self.assertEqual(len(df), 2)
        self.assertEqual(df["Close"].iloc[0], 2.5)

    def test_hourly_alignment_matches_yfinance(self):
        idx = pd.date_range("2026-03-10 13:30", "2026-03-10 19:55", freq="5min", tz="UTC")
        df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)
        h = research.hourly_from_5m(df)
        self.assertEqual(len(h), 7)  # 09:30 … 15:30 ET
        self.assertEqual(h.index[0].tz_convert("America/New_York").strftime("%H:%M"), "09:30")


class DailySimTests(unittest.TestCase):
    def _x(self, o, h, l, c, atr=1.0):
        idx = pd.date_range("2024-01-01", periods=len(c), freq="B")
        return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                             "atr": atr}, index=idx, dtype=float)

    def test_entry_next_open_and_both_hit_assumes_stop(self):
        x = self._x([100, 100, 100, 100], [100, 100, 105, 100],
                    [100, 100, 97, 100], [100, 100, 100, 100])
        sig = np.array([True, False, False, False])
        tr = rd.sim_bracket("AAPL", x, sig, sl_mult=2.0, tp_mult=3.0, hold=3)
        self.assertEqual(tr[0]["t"], x.index[1])          # entered at t+1 open
        self.assertEqual(tr[0]["how"], "sl")
        self.assertAlmostEqual(tr[0]["ret"], -2.0)

    def test_gap_through_stop_fills_at_open(self):
        x = self._x([100, 100, 95, 95], [100, 100.5, 96, 96],
                    [100, 99.5, 94, 94], [100, 100, 95, 95])
        tr = rd.sim_bracket("AAPL", x, np.array([True, False, False, False]), 2.0, 3.0, 3)
        self.assertEqual(tr[0]["how"], "gap_sl")
        self.assertAlmostEqual(tr[0]["ret"], -5.0)

    def test_excess_subtracts_holding_drift(self):
        t = {"symbol": "AAPL", "t": pd.Timestamp("2024-01-02"), "ret": 1.05, "days": 5}
        s = rd.score_trades([t] * 40, {"AAPL": 0.2})       # 0.05% cost, 5d x 0.2%
        self.assertAlmostEqual(s["net"], 1.0)
        self.assertAlmostEqual(s["excess"], 0.0)


class StatsExtraTests(unittest.TestCase):
    def test_max_drawdown(self):
        self.assertEqual(edge_stats.max_drawdown_r([1, 1, -3, 1, -1, 4]), 3.0)
        self.assertEqual(edge_stats.max_drawdown_r([1, 2, 3]), 0.0)

    def test_deflated_sharpe_penalises_many_trials(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.01, 0.04, 120)
        srs = list(rng.normal(0.0, 0.1, 50))
        few = edge_stats.deflated_sharpe(r, srs, n_trials=2)
        many = edge_stats.deflated_sharpe(r, srs, n_trials=500)
        self.assertGreater(few, many)
        self.assertIsNone(edge_stats.deflated_sharpe(r[:5], srs, 10))

    def test_deflated_sharpe_noise_is_low(self):
        rng = np.random.default_rng(2)
        srs = list(rng.normal(0.0, 0.1, 50))
        self.assertLess(edge_stats.deflated_sharpe(rng.normal(0, 0.04, 120), srs, 100), 0.95)


class GateTests(unittest.TestCase):
    VAL = {"verdict": "DATA VALIDATED", "max_dd": 5.0, "detail": "x"}
    REC_OK = {"n": 40, "avg": 0.1, "ci_hi": 0.3, "dd": 2.0}

    def test_research_to_paper_only_when_validated(self):
        self.assertEqual(gate.decide("swing", "research", self.VAL, {}, 1)["new_status"], "paper")
        for v in ("TRADE-OFF", "NO EDGE", "INSUFFICIENT DATA"):
            self.assertIsNone(gate.decide("swing", "research", {"verdict": v}, {}, 1)["new_status"])
        self.assertIsNone(gate.decide("swing", "research", None, {}, 1)["new_status"])

    def test_paper_never_auto_live(self):
        d = gate.decide("swing", "paper", self.VAL, self.REC_OK, 200)
        self.assertIsNone(d["new_status"])
        self.assertTrue(d["eligible"])

    def test_paper_eligibility_needs_time_sample_and_positive_record(self):
        self.assertFalse(gate.decide("swing", "paper", self.VAL, self.REC_OK, 30)["eligible"])
        self.assertFalse(gate.decide("swing", "paper", self.VAL, {**self.REC_OK, "n": 10}, 200)["eligible"])
        self.assertFalse(gate.decide("swing", "paper", self.VAL, {**self.REC_OK, "avg": -0.01}, 200)["eligible"])
        self.assertFalse(gate.decide("swing", "paper", {"verdict": "TRADE-OFF"}, self.REC_OK, 200)["eligible"])
        self.assertFalse(gate.decide("swing", "paper", self.VAL, {**self.REC_OK, "dd": 8.0}, 200)["eligible"])
        # momentum: 3 months of paper is enough sample
        self.assertTrue(gate.decide("momentum", "paper", self.VAL,
                                    {"n": 3, "avg": 0.2, "ci_hi": 2, "dd": 1}, 95)["eligible"])

    def test_live_demotions(self):
        self.assertIsNone(gate.decide("swing", "live", self.VAL, self.REC_OK, 50)["new_status"])
        self.assertEqual(gate.decide("swing", "live", {"verdict": "NO EDGE"}, self.REC_OK, 50)["new_status"], "paper")
        neg = {"n": 40, "avg": -0.2, "ci_hi": -0.05, "dd": 1.0}
        self.assertEqual(gate.decide("swing", "live", self.VAL, neg, 50)["new_status"], "paper")
        self.assertIsNone(gate.decide("swing", "live", self.VAL, {**neg, "n": 10}, 50)["new_status"])
        self.assertEqual(gate.decide("swing", "live", self.VAL, {**self.REC_OK, "dd": 7.6}, 50)["new_status"], "paper")
        self.assertIsNone(gate.decide("swing", "live", self.VAL, {**self.REC_OK, "dd": 7.4}, 50)["new_status"])

    def test_momentum_paper_returns(self):
        rbs = [{"asof": "2026-01-31", "weights": {"A": 0.5}, "prices": {"A": 100, "B": 50}},
               {"asof": "2026-02-28", "weights": {"B": 0.5}, "prices": {"A": 110, "B": 50}}]
        m = gate.momentum_paper_returns(rbs)
        self.assertEqual(len(m), 1)
        # 0.5 * 10% minus entry cost (0.5 * 0.05%/2), bench = (10% + 0%) / 2
        self.assertAlmostEqual(m[0]["port"], 0.05 - 0.5 * 0.05 / 200)
        self.assertAlmostEqual(m[0]["bench"], 0.05)

    def test_paper_returns_use_bench_symbols(self):
        rbs = [{"asof": "2026-01-31", "weights": {"SHY": 1.0},
                "prices": {"SPY": 100, "SHY": 80}, "info": {"bench_symbols": ["SPY"]}},
               {"asof": "2026-02-28", "weights": {"SHY": 1.0},
                "prices": {"SPY": 90, "SHY": 80}, "info": {"bench_symbols": ["SPY"]}}]
        m = gate.portfolio_paper_returns(rbs)[0]
        self.assertAlmostEqual(m["bench"], -0.10)
        self.assertGreater(m["excess"], 0.09)

    def test_portfolio_record_is_beta_adjusted(self):
        # defensive portfolio (beta 0.4) that earned exactly 0.4 x a +5% month:
        # zero alpha, not a 3% shortfall
        from unittest import mock
        rbs = [{"region": "global", "asof": "2026-01-31", "weights": {"A": 0.4},
                "prices": {"A": 100}},
               {"region": "global", "asof": "2026-02-28", "weights": {"A": 0.4},
                "prices": {"A": 105}}]
        with mock.patch("database.get_paper_rebalances", return_value=rbs):
            raw = gate.portfolio_record("x", None)
            adj = gate.portfolio_record("x", None, {"global": 0.4})
        self.assertAlmostEqual(raw["avg"], 100 * (0.02 - 0.4 * 0.05 / 200 - 0.05), places=3)
        self.assertAlmostEqual(adj["avg"], 100 * (-0.4 * 0.05 / 200), places=3)


class PortfolioStrategyTests(unittest.TestCase):
    def test_lowvol_picks_calmest(self):
        import lowvol
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2020-01-01", periods=300)
        vols = np.linspace(0.005, 0.03, 10)
        ret = pd.DataFrame(rng.normal(0, 1, (300, 10)) * vols,
                           index=idx, columns=[f"S{i}" for i in range(10)])
        close = (1 + ret).cumprod()
        mc = close.resample("ME").last()
        m = mc.index[-1]
        w, info = lowvol.target_weights(mc, ret, pd.Series(dtype=float), m,
                                        {"win": 252, "k": 3, "trend": False})
        self.assertEqual(sorted(info["picks"]), ["S0", "S1", "S2"])
        self.assertAlmostEqual(w.sum(), 1.0)

    def test_incomplete_month_dropped(self):
        import momentum
        i = pd.bdate_range("2026-09-01", "2026-11-06")
        df = pd.DataFrame({"a": 1.0}, index=i)
        self.assertEqual(momentum.drop_incomplete_month(df).index[-1], pd.Timestamp("2026-10-30"))
        i = pd.bdate_range("2026-09-01", "2026-10-30")      # Oct 31 is a Saturday
        df = pd.DataFrame({"a": 1.0}, index=i)
        self.assertEqual(len(momentum.drop_incomplete_month(df)), len(df))

    def test_verdict_published(self):
        from momentum import verdict_published as v
        full = {"n": 240, "alpha_mo": 0.23, "alpha_lo": 0.05}
        oos = {"n": 80, "alpha_mo": 0.26}
        self.assertEqual(v(full, oos, 1.0, 0.99, False), "DATA VALIDATED")
        # survivorship-biased universe can never validate
        self.assertEqual(v(full, oos, 1.0, 0.99, True), "TRADE-OFF")
        # edge that died after publication: recent third negative
        self.assertEqual(v(full, {"n": 80, "alpha_mo": -0.1}, 1.0, 0.99, False), "NO EDGE")
        # CI spans zero / fragile / luck
        self.assertEqual(v({**full, "alpha_lo": -0.01}, oos, 1.0, 0.99, False), "TRADE-OFF")
        self.assertEqual(v(full, oos, 0.5, 0.99, False), "TRADE-OFF")
        self.assertEqual(v(full, oos, 1.0, 0.90, False), "TRADE-OFF")
        self.assertEqual(v({**full, "n": 40}, oos, 1.0, 0.99, False), "INSUFFICIENT DATA")

    def test_trend_alloc_moves_downtrends_to_cash(self):
        import trend_alloc as ta
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        up = np.linspace(100, 150, 24)
        down = np.linspace(150, 100, 24)
        mc = pd.DataFrame({"SPY": up, "EFA": down, "IEF": up, "VNQ": down,
                           "DBC": up, "SHY": np.full(24, 80.0)}, index=idx)
        w, info = ta.target_weights(mc, mc.pct_change(), pd.Series(dtype=float),
                                    idx[-1], {"sma": 10, "top": None})
        self.assertAlmostEqual(w["SPY"], 0.2)
        self.assertAlmostEqual(w["EFA"], 0.0)
        self.assertAlmostEqual(w["SHY"], 0.4)
        self.assertAlmostEqual(w.sum(), 1.0)


class PublishedFilterTests(unittest.TestCase):
    def setUp(self):
        import database
        self.db = database
        self._orig = database.DB_PATH
        self.tmp = tempfile.mkdtemp()
        database.DB_PATH = os.path.join(self.tmp, "t.db")
        database.init_db()

    def tearDown(self):
        self.db.DB_PATH = self._orig

    def test_registry_seed_and_published_flag(self):
        db = self.db
        self.assertEqual(db.get_strategy_status("intraday"), "paper")
        self.assertEqual(db.get_strategy_status("swing"), "paper")
        self.assertEqual(db.get_strategy_status("momentum"), "research")
        self.assertEqual(db.get_strategy_status("lowvol"), "research")
        self.assertEqual(db.get_strategy_status("trend_alloc"), "research")
        self.assertEqual(db.get_strategy_status("etf_mom"), "research")
        self.assertEqual(db.get_strategy_status("unknown"), "paper")
        self.assertFalse(db.is_strategy_live("intraday"))
        base = {"symbol": "AAPL", "action": "BUY", "price": 100, "tp": 102, "sl": 99}
        db.log_signal({**base, "published": False})
        self.assertIsNone(db.get_last_signal_action("AAPL", published_only=True))
        self.assertEqual(db.get_last_signal_action("AAPL"), "BUY")
        db.log_signal(base)
        self.assertEqual(db.get_last_signal_action("AAPL", published_only=True), "BUY")
        self.assertEqual(db.get_stats(published_only=True)["total_signals"], 1)
        self.assertEqual(db.get_stats()["total_signals"], 2)

    def test_status_change_records_event(self):
        db = self.db
        self.assertTrue(db.set_strategy_status("swing", "live", "test", actor="t"))
        self.assertFalse(db.set_strategy_status("swing", "live", "again"))
        self.assertTrue(db.is_strategy_live("swing"))
        ev = db.get_strategy_events(1)[0]
        self.assertEqual((ev["old_status"], ev["new_status"], ev["actor"]), ("paper", "live", "t"))
        with self.assertRaises(ValueError):
            db.set_strategy_status("swing", "bogus", "x")


if __name__ == "__main__":
    unittest.main()
