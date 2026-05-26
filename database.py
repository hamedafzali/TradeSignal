import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "trading.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                chat_id    TEXT PRIMARY KEY,
                username   TEXT,
                first_seen TEXT,
                last_seen  TEXT
            );

            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT NOT NULL,
                action        TEXT NOT NULL,
                strength      TEXT,
                price         REAL,
                tp            REAL,
                sl            REAL,
                tp_pct        REAL,
                sl_pct        REAL,
                rr            REAL,
                rsi           REAL,
                trend         TEXT,
                quality       INTEGER DEFAULT 0,
                ai_confidence REAL,
                vol_spike     INTEGER DEFAULT 0,
                reasons       TEXT,
                features      TEXT,
                sent_at       TEXT,
                outcome       TEXT DEFAULT 'pending',
                outcome_price REAL,
                outcome_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                entry_price REAL NOT NULL,
                tp          REAL,
                sl          REAL,
                opened_at   TEXT NOT NULL,
                closed_at   TEXT,
                exit_price  REAL,
                pnl_pct     REAL,
                status      TEXT DEFAULT 'open'
            );

            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id TEXT NOT NULL,
                symbol  TEXT NOT NULL,
                PRIMARY KEY (user_id, symbol)
            );

            CREATE TABLE IF NOT EXISTS symbols (
                symbol     TEXT PRIMARY KEY,
                active     INTEGER DEFAULT 1,
                added_at   TEXT NOT NULL,
                added_by   TEXT DEFAULT 'admin'
            );

            CREATE TABLE IF NOT EXISTS ml_training_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol       TEXT NOT NULL,
                trained_at   TEXT NOT NULL,
                train_samples INTEGER,
                outcome_samples INTEGER DEFAULT 0
            );
        """)
        # Migrate existing positions table if tp/sl columns missing
        cols = [r[1] for r in conn.execute("PRAGMA table_info(positions)").fetchall()]
        if "tp" not in cols:
            conn.execute("ALTER TABLE positions ADD COLUMN tp REAL")
        if "sl" not in cols:
            conn.execute("ALTER TABLE positions ADD COLUMN sl REAL")


# ── Users ─────────────────────────────────────────────────────────────────────

def log_user(chat_id: str | int, username: str = "") -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute("""
            INSERT INTO users (chat_id, username, first_seen, last_seen)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET last_seen = ?, username = ?
        """, (str(chat_id), username, now, now, now, username))


def get_all_user_ids() -> list[str]:
    with _conn() as conn:
        rows = conn.execute("SELECT chat_id FROM users").fetchall()
        return [r["chat_id"] for r in rows]


# ── Signals ───────────────────────────────────────────────────────────────────

def log_signal(sig: dict, features: dict | None = None) -> int:
    with _conn() as conn:
        cur = conn.execute("""
            INSERT INTO signals
                (symbol, action, strength, price, tp, sl, tp_pct, sl_pct, rr,
                 rsi, trend, quality, ai_confidence, vol_spike, reasons, features, sent_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sig["symbol"], sig["action"], sig.get("strength", "RULE"),
            sig.get("price"), sig.get("tp"), sig.get("sl"),
            sig.get("tp_pct"), sig.get("sl_pct"), sig.get("rr"),
            sig.get("rsi"), sig.get("trend"), sig.get("quality", 0),
            sig.get("ai_confidence"), int(sig.get("vol_spike", False)),
            json.dumps(sig.get("reasons", [])),
            json.dumps(features or {}),
            datetime.utcnow().isoformat(),
        ))
        return cur.lastrowid


def get_last_signal_action(symbol: str, within_hours: int = 6) -> str | None:
    """Persistent deduplication — survives restarts."""
    cutoff = (datetime.utcnow() - timedelta(hours=within_hours)).isoformat()
    with _conn() as conn:
        row = conn.execute("""
            SELECT action FROM signals
            WHERE symbol = ? AND sent_at >= ?
            ORDER BY sent_at DESC LIMIT 1
        """, (symbol, cutoff)).fetchone()
        return row["action"] if row else None


def get_pending_outcomes(older_than_hours: int = 24) -> list[dict]:
    cutoff = (datetime.utcnow() - timedelta(hours=older_than_hours)).isoformat()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM signals WHERE outcome = 'pending' AND sent_at <= ? AND tp IS NOT NULL",
            (cutoff,)
        ).fetchall()
        return [dict(r) for r in rows]


def update_outcome(signal_id: int, outcome: str, outcome_price: float) -> None:
    with _conn() as conn:
        conn.execute("""
            UPDATE signals SET outcome = ?, outcome_price = ?, outcome_at = ?
            WHERE id = ?
        """, (outcome, outcome_price, datetime.utcnow().isoformat(), signal_id))


def get_outcome_training_data() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT features, action, outcome FROM signals
            WHERE outcome IN ('correct', 'incorrect') AND features IS NOT NULL
        """).fetchall()
    result = []
    for r in rows:
        feats = json.loads(r["features"] or "{}")
        if not feats:
            continue
        label = (1 if r["action"] == "BUY" else -1) if r["outcome"] == "correct" else (
            -1 if r["action"] == "BUY" else 1
        )
        result.append({"features": feats, "label": label})
    return result


# ── Positions (per-user) ──────────────────────────────────────────────────────

def open_position(user_id: str | int, symbol: str, entry_price: float,
                  tp: float | None = None, sl: float | None = None) -> None:
    with _conn() as conn:
        conn.execute("""
            UPDATE positions SET status = 'replaced', closed_at = ?
            WHERE user_id = ? AND symbol = ? AND status = 'open'
        """, (datetime.utcnow().isoformat(), str(user_id), symbol))
        conn.execute("""
            INSERT INTO positions (user_id, symbol, entry_price, tp, sl, opened_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(user_id), symbol, entry_price, tp, sl, datetime.utcnow().isoformat()))


def close_position(user_id: str | int, symbol: str, exit_price: float) -> dict | None:
    with _conn() as conn:
        row = conn.execute("""
            SELECT * FROM positions
            WHERE user_id = ? AND symbol = ? AND status = 'open'
            ORDER BY opened_at DESC LIMIT 1
        """, (str(user_id), symbol)).fetchone()
        if not row:
            return None
        pnl = (exit_price - row["entry_price"]) / row["entry_price"] * 100
        conn.execute("""
            UPDATE positions
            SET status = 'closed', closed_at = ?, exit_price = ?, pnl_pct = ?
            WHERE id = ?
        """, (datetime.utcnow().isoformat(), exit_price, pnl, row["id"]))
        return {"symbol": symbol, "entry": row["entry_price"], "exit": exit_price, "pnl_pct": pnl}


def get_user_open_positions(user_id: str | int) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM positions WHERE user_id = ? AND status = 'open'
            ORDER BY opened_at DESC
        """, (str(user_id),)).fetchall()
        return [dict(r) for r in rows]


def get_all_open_positions() -> list[dict]:
    """All open positions across all users — used by TP/SL monitor job."""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM positions WHERE status = 'open'
            ORDER BY opened_at DESC
        """).fetchall()
        return [dict(r) for r in rows]


def get_users_with_open_position(symbol: str) -> list[str]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT user_id FROM positions
            WHERE symbol = ? AND status = 'open'
        """, (symbol,)).fetchall()
        return [r["user_id"] for r in rows]


def get_user_pnl(user_id: str | int) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT symbol, entry_price, exit_price, pnl_pct, opened_at, closed_at
            FROM positions WHERE user_id = ? AND status = 'closed'
            ORDER BY closed_at DESC LIMIT 20
        """, (str(user_id),)).fetchall()
        return [dict(r) for r in rows]


# ── Subscriptions ─────────────────────────────────────────────────────────────

def subscribe_user(user_id: str | int, symbols: list[str]) -> None:
    with _conn() as conn:
        for sym in symbols:
            conn.execute("""
                INSERT OR IGNORE INTO subscriptions (user_id, symbol) VALUES (?, ?)
            """, (str(user_id), sym.upper()))


def unsubscribe_user(user_id: str | int, symbol: str | None = None) -> None:
    with _conn() as conn:
        if symbol:
            conn.execute("DELETE FROM subscriptions WHERE user_id = ? AND symbol = ?",
                         (str(user_id), symbol.upper()))
        else:
            conn.execute("DELETE FROM subscriptions WHERE user_id = ?", (str(user_id),))


def get_user_subscriptions(user_id: str | int) -> list[str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT symbol FROM subscriptions WHERE user_id = ? ORDER BY symbol",
            (str(user_id),)
        ).fetchall()
        return [r["symbol"] for r in rows]


def get_subscribers_for_symbol(symbol: str) -> list[str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT user_id FROM subscriptions WHERE symbol = ?", (symbol.upper(),)
        ).fetchall()
        return [r["user_id"] for r in rows]


def get_all_subscriber_ids() -> list[str]:
    with _conn() as conn:
        rows = conn.execute("SELECT DISTINCT user_id FROM subscriptions").fetchall()
        return [r["user_id"] for r in rows]


# ── Dashboard stats ───────────────────────────────────────────────────────────

def get_stats() -> dict:
    with _conn() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        today = datetime.utcnow().date().isoformat()
        today_signals = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE sent_at LIKE ?", (f"{today}%",)
        ).fetchone()[0]
        resolved = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome != 'pending'"
        ).fetchone()[0]
        correct = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome = 'correct'"
        ).fetchone()[0]
        accuracy = round(correct / resolved * 100, 1) if resolved > 0 else 0

        symbol_rows = conn.execute("""
            SELECT symbol,
                   COUNT(*) AS total,
                   SUM(CASE WHEN outcome='correct'   THEN 1 ELSE 0 END) AS correct,
                   SUM(CASE WHEN outcome='incorrect' THEN 1 ELSE 0 END) AS incorrect,
                   SUM(CASE WHEN outcome!='pending'  THEN 1 ELSE 0 END) AS resolved
            FROM signals GROUP BY symbol
        """).fetchall()
        per_symbol = {}
        for r in symbol_rows:
            acc = round(r["correct"] / r["resolved"] * 100, 1) if r["resolved"] > 0 else 0
            per_symbol[r["symbol"]] = {
                "total": r["total"], "correct": r["correct"],
                "incorrect": r["incorrect"], "accuracy": acc,
            }

        daily_rows = conn.execute("""
            SELECT DATE(sent_at) AS day, COUNT(*) AS count
            FROM signals WHERE sent_at >= DATE('now', '-14 days')
            GROUP BY day ORDER BY day
        """).fetchall()

        buy_sell = conn.execute(
            "SELECT action, COUNT(*) AS cnt FROM signals GROUP BY action"
        ).fetchall()

        top_pnl = conn.execute("""
            SELECT u.username, u.chat_id,
                   COUNT(p.id) AS trades,
                   AVG(p.pnl_pct) AS avg_pnl,
                   SUM(CASE WHEN p.pnl_pct > 0 THEN 1 ELSE 0 END) AS wins
            FROM positions p JOIN users u ON p.user_id = u.chat_id
            WHERE p.status = 'closed'
            GROUP BY p.user_id ORDER BY avg_pnl DESC LIMIT 10
        """).fetchall()

        return {
            "total_users": total_users,
            "total_signals": total_signals,
            "today_signals": today_signals,
            "accuracy": accuracy,
            "correct": correct,
            "resolved": resolved,
            "per_symbol": per_symbol,
            "daily": [{"day": r["day"], "count": r["count"]} for r in daily_rows],
            "buy_sell": {r["action"]: r["cnt"] for r in buy_sell},
            "top_pnl": [dict(r) for r in top_pnl],
        }


def get_weekly_stats() -> dict:
    """Signal stats for the past 7 days — used in weekly recap."""
    cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT symbol, action,
                   COUNT(*) AS total,
                   SUM(CASE WHEN outcome='correct'   THEN 1 ELSE 0 END) AS correct,
                   SUM(CASE WHEN outcome='incorrect' THEN 1 ELSE 0 END) AS incorrect,
                   SUM(CASE WHEN outcome!='pending'  THEN 1 ELSE 0 END) AS resolved
            FROM signals WHERE sent_at >= ?
            GROUP BY symbol, action
        """, (cutoff,)).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE sent_at >= ?", (cutoff,)
        ).fetchone()[0]
        resolved_total = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome != 'pending' AND sent_at >= ?", (cutoff,)
        ).fetchone()[0]
        correct_total = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome = 'correct' AND sent_at >= ?", (cutoff,)
        ).fetchone()[0]
    accuracy = round(correct_total / resolved_total * 100, 1) if resolved_total > 0 else 0
    per_symbol = {}
    for r in rows:
        sym = r["symbol"]
        if sym not in per_symbol:
            per_symbol[sym] = {"total": 0, "correct": 0, "resolved": 0}
        per_symbol[sym]["total"] += r["total"]
        per_symbol[sym]["correct"] += r["correct"]
        per_symbol[sym]["resolved"] += r["resolved"]
    return {
        "total": total,
        "resolved": resolved_total,
        "correct": correct_total,
        "accuracy": accuracy,
        "per_symbol": per_symbol,
    }


# ── Symbol management ─────────────────────────────────────────────────────────

def seed_symbols(symbols: list[str]) -> None:
    """Insert symbols from env on first run — skip if already in DB."""
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        for sym in symbols:
            conn.execute("""
                INSERT OR IGNORE INTO symbols (symbol, active, added_at, added_by)
                VALUES (?, 1, ?, 'env')
            """, (sym.upper(), now))


def get_active_symbols() -> list[str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT symbol FROM symbols WHERE active = 1 ORDER BY added_at"
        ).fetchall()
        return [r["symbol"] for r in rows]


def add_symbol(symbol: str, added_by: str = "admin") -> bool:
    """Returns True if newly added, False if already exists."""
    sym = symbol.upper()
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT active FROM symbols WHERE symbol = ?", (sym,)
        ).fetchone()
        if existing:
            if not existing["active"]:
                conn.execute(
                    "UPDATE symbols SET active = 1, added_at = ? WHERE symbol = ?",
                    (now, sym)
                )
                return True
            return False
        conn.execute(
            "INSERT INTO symbols (symbol, active, added_at, added_by) VALUES (?, 1, ?, ?)",
            (sym, now, added_by)
        )
        return True


def remove_symbol(symbol: str) -> bool:
    """Soft-delete (sets active=0). Returns True if found."""
    sym = symbol.upper()
    with _conn() as conn:
        row = conn.execute(
            "SELECT active FROM symbols WHERE symbol = ?", (sym,)
        ).fetchone()
        if not row or not row["active"]:
            return False
        conn.execute("UPDATE symbols SET active = 0 WHERE symbol = ?", (sym,))
        return True


def get_all_symbols_with_status() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT symbol, active, added_at, added_by FROM symbols ORDER BY added_at"
        ).fetchall()
        return [dict(r) for r in rows]


# ── ML training log ───────────────────────────────────────────────────────────

def log_training(symbol: str, train_samples: int, outcome_samples: int = 0) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO ml_training_log (symbol, trained_at, train_samples, outcome_samples)
            VALUES (?, ?, ?, ?)
        """, (symbol, datetime.utcnow().isoformat(), train_samples, outcome_samples))


def get_ml_accuracy_stats() -> dict:
    """
    Returns per-symbol ML accuracy derived from resolved signals that had
    an AI confidence value, plus the training log.
    """
    with _conn() as conn:
        # Per-symbol AI accuracy: signals where AI had a view and we know the outcome
        rows = conn.execute("""
            SELECT symbol, action, ai_confidence, outcome, strength
            FROM signals
            WHERE outcome IN ('correct', 'incorrect')
              AND ai_confidence IS NOT NULL
            ORDER BY sent_at
        """).fetchall()

        # Training log — last entry per symbol
        train_rows = conn.execute("""
            SELECT symbol, trained_at, train_samples, outcome_samples
            FROM ml_training_log
            WHERE id IN (
                SELECT MAX(id) FROM ml_training_log GROUP BY symbol
            )
        """).fetchall()
        training = {r["symbol"]: dict(r) for r in train_rows}

        # Total training runs per symbol
        run_counts = conn.execute("""
            SELECT symbol, COUNT(*) AS runs FROM ml_training_log GROUP BY symbol
        """).fetchall()
        run_map = {r["symbol"]: r["runs"] for r in run_counts}

        # Confidence buckets: [0.65-0.70), [0.70-0.80), [0.80-0.90), [0.90-1.0]
        buckets_def = [(0.65, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
        bucket_labels = ["65–70%", "70–80%", "80–90%", "90%+"]

        per_symbol: dict[str, dict] = {}
        buckets_global: dict[str, dict] = {
            lbl: {"total": 0, "correct": 0} for lbl in bucket_labels
        }

        for r in rows:
            sym = r["symbol"]
            if sym not in per_symbol:
                per_symbol[sym] = {
                    "ai_total": 0, "ai_correct": 0,
                    "strong_total": 0, "strong_correct": 0,
                    "rule_total": 0, "rule_correct": 0,
                }
            d = per_symbol[sym]
            is_correct = r["outcome"] == "correct"

            conf = r["ai_confidence"]
            if conf is not None and conf >= 0.65:
                d["ai_total"] += 1
                if is_correct:
                    d["ai_correct"] += 1

            strength = r["strength"]
            if strength == "STRONG":
                d["strong_total"] += 1
                if is_correct:
                    d["strong_correct"] += 1
            elif strength == "RULE":
                d["rule_total"] += 1
                if is_correct:
                    d["rule_correct"] += 1

            # Confidence bucket
            if conf is not None:
                for (lo, hi), lbl in zip(buckets_def, bucket_labels):
                    if lo <= conf < hi:
                        buckets_global[lbl]["total"] += 1
                        if is_correct:
                            buckets_global[lbl]["correct"] += 1
                        break

        # Calculate accuracy percentages
        for sym, d in per_symbol.items():
            d["ai_accuracy"] = (
                round(d["ai_correct"] / d["ai_total"] * 100, 1)
                if d["ai_total"] > 0 else None
            )
            d["strong_accuracy"] = (
                round(d["strong_correct"] / d["strong_total"] * 100, 1)
                if d["strong_total"] > 0 else None
            )
            d["rule_accuracy"] = (
                round(d["rule_correct"] / d["rule_total"] * 100, 1)
                if d["rule_total"] > 0 else None
            )
            d["training"] = training.get(sym)
            d["train_runs"] = run_map.get(sym, 0)

        for lbl, b in buckets_global.items():
            b["accuracy"] = (
                round(b["correct"] / b["total"] * 100, 1) if b["total"] > 0 else None
            )

        return {
            "per_symbol": per_symbol,
            "confidence_buckets": [
                {"label": lbl, **buckets_global[lbl]} for lbl in bucket_labels
            ],
        }


def get_recent_signals(limit: int = 50) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT id, symbol, action, strength, price, rsi, ai_confidence,
                   reasons, sent_at, outcome, outcome_price, outcome_at
            FROM signals ORDER BY sent_at DESC LIMIT ?
        """, (limit,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["reasons"] = json.loads(d["reasons"] or "[]")
            result.append(d)
        return result
