import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY_SECONDS", "7200"))  # 2-hour fallback
RETRAIN_OUTCOME_THRESHOLD = int(os.getenv("RETRAIN_OUTCOME_THRESHOLD", "3"))
AI_SIGNAL_THRESHOLD = float(os.getenv("AI_SIGNAL_THRESHOLD", "0.65"))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    feat = pd.DataFrame(index=df.index)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(com=13, min_periods=14).mean()
    al = loss.ewm(com=13, min_periods=14).mean()
    feat["rsi"] = 100 - (100 / (1 + ag / al))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = (macd - sig) / close  # normalized

    # EMA ratio
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    feat["ema_ratio"] = ema9 / ema21 - 1

    # Price momentum
    for p in [3, 5, 10, 20]:
        feat[f"roc_{p}"] = close.pct_change(p)

    # Bollinger Band position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_pos"] = (close - sma20) / (2 * std20 + 1e-9)

    # Volume ratio
    vol_ma = volume.rolling(20).mean()
    feat["vol_ratio"] = volume / (vol_ma + 1e-9)

    # ATR (normalized)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    feat["atr_pct"] = tr.rolling(14).mean() / (close + 1e-9)

    return feat


def build_labels(df: pd.DataFrame, forward: int = 3, threshold: float = 0.003) -> pd.Series:
    close = df["Close"].squeeze()
    ret = close.shift(-forward) / close - 1
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[ret > threshold] = 1
    labels[ret < -threshold] = -1
    return labels


class StockModel:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.clf_buy = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
        self.clf_sell = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.trained = False
        self.trained_at = 0.0
        self.train_samples = 0
        self.outcome_samples = 0
        self.outcome_count_at_train = 0  # resolved outcome count when last trained
        self._path = os.path.join(CACHE_DIR, f"{symbol}.pkl")
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "rb") as f:
                d = pickle.load(f)
            self.__dict__.update(d)
            logger.info(f"[ML] Loaded cached model for {self.symbol}")
        except Exception:
            pass

    def _save(self):
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(self._path, "wb") as f:
            pickle.dump(data, f)

    def needs_retrain(self, current_outcome_count: int = 0) -> bool:
        if not self.trained:
            return True
        new_outcomes = current_outcome_count - self.outcome_count_at_train
        if new_outcomes >= RETRAIN_OUTCOME_THRESHOLD:
            return True
        return (time.time() - self.trained_at) > RETRAIN_EVERY

    def train(self, outcome_data: list[dict] | None = None, _log: bool = True) -> bool:
        """
        Train on 60 days of hourly data, then blend in real outcome samples.
        outcome_data: list of {"features": {col: val, ...}, "label": int}
        """
        logger.info(f"[ML] Training {self.symbol}...")
        try:
            df = yf.download(self.symbol, period="60d", interval="1h", progress=False, auto_adjust=True)
            if len(df) < 200:
                logger.warning(f"[ML] Too little data for {self.symbol}")
                return False

            features = build_features(df)
            labels = build_labels(df)
            combined = pd.concat([features, labels.rename("label")], axis=1).dropna()

            if len(combined) < 100:
                return False

            self.feature_cols = [c for c in combined.columns if c != "label"]
            X = combined[self.feature_cols].values
            y = combined["label"].values

            # Blend in confirmed real outcomes (repeat each 3x to give them weight)
            if outcome_data:
                extra_rows, extra_labels = [], []
                for item in outcome_data:
                    row = [item["features"].get(col, 0.0) for col in self.feature_cols]
                    extra_rows.extend([row] * 3)
                    extra_labels.extend([item["label"]] * 3)
                if extra_rows:
                    X = np.vstack([X, extra_rows])
                    y = np.concatenate([y, extra_labels])
                    logger.info(f"[ML] {self.symbol}: blended {len(outcome_data)} real outcomes")

            X_scaled = self.scaler.fit_transform(X)
            self.clf_buy.fit(X_scaled, (y == 1).astype(int))
            self.clf_sell.fit(X_scaled, (y == -1).astype(int))

            self.trained = True
            self.trained_at = time.time()
            self.train_samples = len(X)
            self.outcome_samples = len(outcome_data) if outcome_data else 0
            # Snapshot current resolved outcome count so needs_retrain() can
            # detect new outcomes arriving after this train run
            try:
                from database import get_outcome_count
                self.outcome_count_at_train = get_outcome_count(self.symbol)
            except Exception:
                self.outcome_count_at_train = self.outcome_samples
            self._save()

            # Persist to DB for dashboard reporting (skipped when caller handles logging)
            if _log:
                try:
                    from database import log_training
                    trigger = "outcomes" if (outcome_data and len(outcome_data) >= 3) else "time"
                    log_training(self.symbol, len(X), self.outcome_samples, trigger=trigger)
                except Exception:
                    pass

            logger.info(f"[ML] {self.symbol} trained on {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"[ML] Train error for {self.symbol}: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> dict:
        """Returns buy_prob, sell_prob, ai_signal for the latest candle."""
        default = {"buy_prob": None, "sell_prob": None, "ai_signal": None}
        if not self.trained or not self.feature_cols:
            return default
        try:
            features = build_features(df)
            row = features[self.feature_cols].dropna().iloc[-1:]
            if row.empty:
                return default

            X = self.scaler.transform(row.values)
            buy_prob = float(self.clf_buy.predict_proba(X)[0][1])
            sell_prob = float(self.clf_sell.predict_proba(X)[0][1])

            ai_signal = None
            if buy_prob > AI_SIGNAL_THRESHOLD:
                ai_signal = "BUY"
            elif sell_prob > AI_SIGNAL_THRESHOLD:
                ai_signal = "SELL"

            return {"buy_prob": buy_prob, "sell_prob": sell_prob, "ai_signal": ai_signal}
        except Exception as e:
            logger.error(f"[ML] Predict error for {self.symbol}: {e}")
            return default
