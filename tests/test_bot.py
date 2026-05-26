import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd

import bot


class ResolveSignalOutcomeTests(unittest.TestCase):
    def _frame(self, rows, freq="5min"):
        idx = pd.date_range("2026-05-26 10:00:00", periods=len(rows), freq=freq, tz="UTC")
        return pd.DataFrame(rows, index=idx)

    @patch("bot.yf.download")
    def test_buy_signal_resolves_correct_when_tp_hit_first(self, download_mock):
        download_mock.return_value = self._frame([
            {"Open": 100, "High": 101, "Low": 99.5, "Close": 100.5, "Volume": 1_000},
            {"Open": 100.5, "High": 106, "Low": 100, "Close": 105.5, "Volume": 1_100},
        ])
        sig = {
            "symbol": "AAPL",
            "action": "BUY",
            "price": 100.0,
            "tp": 105.0,
            "sl": 95.0,
            "sent_at": "2026-05-26T10:00:00",
        }

        resolved = bot._resolve_signal_outcome(sig)

        self.assertIsNotNone(resolved)
        self.assertEqual("correct", resolved["outcome"])
        self.assertEqual(105.0, resolved["price"])
        self.assertEqual("tp_hit", resolved["metadata"]["resolution_reason"])
        self.assertTrue(resolved["metadata"]["touched_tp"])
        self.assertFalse(resolved["metadata"]["touched_sl"])

    @patch("bot.yf.download")
    def test_sell_signal_resolves_incorrect_when_sl_hit_first(self, download_mock):
        download_mock.return_value = self._frame([
            {"Open": 100, "High": 100.5, "Low": 99.0, "Close": 99.5, "Volume": 900},
            {"Open": 99.5, "High": 106.5, "Low": 99.2, "Close": 106.0, "Volume": 1_300},
        ])
        sig = {
            "symbol": "AAPL",
            "action": "SELL",
            "price": 100.0,
            "tp": 95.0,
            "sl": 105.0,
            "sent_at": "2026-05-26T10:00:00",
        }

        resolved = bot._resolve_signal_outcome(sig)

        self.assertIsNotNone(resolved)
        self.assertEqual("incorrect", resolved["outcome"])
        self.assertEqual(105.0, resolved["price"])
        self.assertEqual("sl_hit", resolved["metadata"]["resolution_reason"])

    @patch("bot.datetime")
    @patch("bot.yf.download")
    def test_signal_stays_pending_before_timeout_when_no_level_hit(self, download_mock, datetime_mock):
        download_mock.return_value = self._frame([
            {"Open": 100, "High": 103, "Low": 98, "Close": 101, "Volume": 800},
            {"Open": 101, "High": 104, "Low": 99, "Close": 102, "Volume": 850},
        ])
        real_datetime = __import__("datetime").datetime
        fake_now = real_datetime.fromisoformat("2026-05-26T12:00:00+00:00")
        datetime_mock.fromisoformat.side_effect = real_datetime.fromisoformat
        datetime_mock.now.return_value = fake_now

        sig = {
            "symbol": "AAPL",
            "action": "BUY",
            "price": 100.0,
            "tp": 110.0,
            "sl": 95.0,
            "sent_at": "2026-05-26T10:00:00",
        }

        resolved = bot._resolve_signal_outcome(sig)

        self.assertIsNone(resolved)

    @patch("bot.datetime")
    @patch("bot.yf.download")
    def test_signal_times_out_with_metadata_when_no_level_hit_after_window(self, download_mock, datetime_mock):
        download_mock.return_value = self._frame([
            {"Open": 100, "High": 103, "Low": 98, "Close": 101, "Volume": 800},
            {"Open": 101, "High": 104, "Low": 99, "Close": 102, "Volume": 850},
        ])
        real_datetime = __import__("datetime").datetime
        fake_now = real_datetime.fromisoformat("2026-05-27T12:30:00+00:00")
        datetime_mock.fromisoformat.side_effect = real_datetime.fromisoformat
        datetime_mock.now.return_value = fake_now

        sig = {
            "symbol": "AAPL",
            "action": "BUY",
            "price": 100.0,
            "tp": 110.0,
            "sl": 95.0,
            "sent_at": "2026-05-26T10:00:00",
        }

        resolved = bot._resolve_signal_outcome(sig)

        self.assertIsNotNone(resolved)
        self.assertEqual("neutral", resolved["outcome"])
        self.assertEqual("timeout_no_hit", resolved["metadata"]["resolution_reason"])
        self.assertGreater(resolved["metadata"]["resolution_minutes"], 0)


class StartTrackingTests(unittest.IsolatedAsyncioTestCase):
    @patch("bot.open_position")
    @patch("bot.log_user")
    @patch("bot.get_user_pref", return_value={"lang": "en", "mode": "beginner"})
    async def test_cmd_start_track_link_stores_tp_and_sl(self, _pref_mock, _log_user_mock, open_position_mock):
        reply_text = AsyncMock()
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=123, username="tester"),
            message=SimpleNamespace(reply_text=reply_text),
        )
        context = SimpleNamespace(args=["track_AAPL_100.0_110.0_95.0"])

        await bot.cmd_start(update, context)

        open_position_mock.assert_called_once_with("123", "AAPL", 100.0, tp=110.0, sl=95.0)
        reply_text.assert_awaited()


class EnsureModelsTrainedTests(unittest.IsolatedAsyncioTestCase):
    @patch("bot.get_outcome_training_data")
    async def test_ensure_models_trained_uses_symbol_specific_outcomes(self, training_data_mock):
        model_a = Mock()
        model_a.needs_retrain.return_value = True
        model_a.train.return_value = True
        model_b = Mock()
        model_b.needs_retrain.return_value = True
        model_b.train.return_value = True

        training_data_mock.side_effect = lambda symbol=None: [{"symbol": symbol}]

        with patch.dict(bot.models, {"AAPL": model_a, "TSLA": model_b}, clear=True):
            await bot._ensure_models_trained()

        self.assertEqual(
            [call.kwargs for call in training_data_mock.call_args_list],
            [{"symbol": "AAPL"}, {"symbol": "TSLA"}],
        )
        model_a.train.assert_called_once_with(outcome_data=[{"symbol": "AAPL"}])
        model_b.train.assert_called_once_with(outcome_data=[{"symbol": "TSLA"}])


if __name__ == "__main__":
    unittest.main()
