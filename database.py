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
                outcome_at    TEXT,
                resolution_reason TEXT,
                touched_tp    INTEGER DEFAULT 0,
                touched_sl    INTEGER DEFAULT 0,
                resolution_minutes REAL,
                max_favorable_pct REAL,
                max_adverse_pct REAL
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

            CREATE TABLE IF NOT EXISTS user_preferences (
                chat_id TEXT PRIMARY KEY,
                lang    TEXT DEFAULT 'en',
                mode    TEXT DEFAULT 'beginner'
            );

            CREATE TABLE IF NOT EXISTS ml_training_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT NOT NULL,
                trained_at      TEXT NOT NULL,
                train_samples   INTEGER,
                outcome_samples INTEGER DEFAULT 0,
                trigger         TEXT DEFAULT 'time',
                win_rate        REAL,
                correct_count   INTEGER DEFAULT 0,
                incorrect_count INTEGER DEFAULT 0,
                neutral_count   INTEGER DEFAULT 0,
                job_id          INTEGER
            );

            CREATE TABLE IF NOT EXISTS ml_training_jobs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type        TEXT NOT NULL,
                status          TEXT DEFAULT 'running',
                started_at      TEXT NOT NULL,
                completed_at    TEXT,
                total_symbols   INTEGER DEFAULT 0,
                done_symbols    INTEGER DEFAULT 0,
                current_symbol  TEXT,
                result_summary  TEXT,
                note            TEXT
            );

            CREATE TABLE IF NOT EXISTS ml_cycle_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT NOT NULL,
                checked_at    TEXT NOT NULL,
                retrain_needed INTEGER DEFAULT 0,
                retrained     INTEGER DEFAULT 0,
                resolved_outcomes INTEGER DEFAULT 0,
                new_outcomes  INTEGER DEFAULT 0,
                note          TEXT
            );

            CREATE TABLE IF NOT EXISTS action_requests (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                action        TEXT NOT NULL,
                symbol        TEXT,
                requested_at  TEXT NOT NULL,
                status        TEXT DEFAULT 'pending',
                requested_by  TEXT DEFAULT 'dashboard',
                result_note   TEXT,
                completed_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS app_settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sentiment_cache (
                symbol       TEXT PRIMARY KEY,
                score        REAL DEFAULT 0.0,
                label        TEXT DEFAULT 'neutral',
                source       TEXT DEFAULT 'disabled',
                headlines    TEXT DEFAULT '[]',
                computed_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_wf_results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT NOT NULL,
                tested_at     TEXT NOT NULL,
                train_bars    INTEGER,
                test_bars     INTEGER,
                rule_signals  INTEGER,
                rule_win_rate REAL,
                ai_signals    INTEGER,
                ai_win_rate   REAL,
                grade         TEXT
            );

            CREATE TABLE IF NOT EXISTS scan_log (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at              TEXT NOT NULL,
                symbols_total           INTEGER DEFAULT 0,
                symbols_scanned         INTEGER DEFAULT 0,
                symbols_skipped_market  INTEGER DEFAULT 0,
                symbols_skipped_brake   INTEGER DEFAULT 0,
                signals_generated       INTEGER DEFAULT 0,
                scan_duration_ms        REAL,
                regime                  TEXT,
                perf_brake_engaged      INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS pipeline_trace (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id       INTEGER,
                symbol          TEXT NOT NULL,
                scanned_at      TEXT NOT NULL,
                data_ok         INTEGER DEFAULT 1,
                features_count  INTEGER DEFAULT 0,
                rule_signal     TEXT,
                ml_buy_prob     REAL,
                ml_sell_prob    REAL,
                meta_conf       REAL,
                sentiment_score REAL,
                sentiment_adj   REAL,
                suppressed      INTEGER DEFAULT 0,
                suppress_reason TEXT,
                final_action    TEXT
            );
        """)
        # Migrate existing positions table if tp/sl columns missing
        cols = [r[1] for r in conn.execute("PRAGMA table_info(positions)").fetchall()]
        if "tp" not in cols:
            conn.execute("ALTER TABLE positions ADD COLUMN tp REAL")
        if "sl" not in cols:
            conn.execute("ALTER TABLE positions ADD COLUMN sl REAL")
        signal_cols = [r[1] for r in conn.execute("PRAGMA table_info(signals)").fetchall()]
        signal_migrations = {
            "resolution_reason": "TEXT",
            "touched_tp": "INTEGER DEFAULT 0",
            "touched_sl": "INTEGER DEFAULT 0",
            "resolution_minutes": "REAL",
            "max_favorable_pct": "REAL",
            "max_adverse_pct": "REAL",
            "suggested_size_pct": "REAL",
            "regime": "TEXT DEFAULT 'unknown'",
        }
        for col, sql_type in signal_migrations.items():
            if col not in signal_cols:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {sql_type}")
        train_cols = [r[1] for r in conn.execute("PRAGMA table_info(ml_training_log)").fetchall()]
        train_migrations = {
            "trigger": "TEXT DEFAULT 'time'",
            "win_rate": "REAL",
            "correct_count": "INTEGER DEFAULT 0",
            "incorrect_count": "INTEGER DEFAULT 0",
            "neutral_count": "INTEGER DEFAULT 0",
            "job_id": "INTEGER",
        }
        for col, sql_type in train_migrations.items():
            if col not in train_cols:
                conn.execute(f"ALTER TABLE ml_training_log ADD COLUMN {col} {sql_type}")
        # Add market column to symbols if missing
        sym_cols = [r[1] for r in conn.execute("PRAGMA table_info(symbols)").fetchall()]
        if "market" not in sym_cols:
            conn.execute("ALTER TABLE symbols ADD COLUMN market TEXT")


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
                 rsi, trend, quality, ai_confidence, vol_spike, reasons, features,
                 sent_at, suggested_size_pct, regime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sig["symbol"], sig["action"], sig.get("strength", "RULE"),
            sig.get("price"), sig.get("tp"), sig.get("sl"),
            sig.get("tp_pct"), sig.get("sl_pct"), sig.get("rr"),
            sig.get("rsi"), sig.get("trend"), sig.get("quality", 0),
            sig.get("ai_confidence"), int(sig.get("vol_spike", False)),
            json.dumps(sig.get("reasons", [])),
            json.dumps(features or {}),
            datetime.utcnow().isoformat(),
            sig.get("suggested_size_pct"),
            sig.get("regime", "unknown"),
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


def get_latest_signal(symbol: str, action: str | None = None) -> dict | None:
    """Most recent signal for a symbol, optionally filtered by action."""
    with _conn() as conn:
        if action:
            row = conn.execute("""
                SELECT * FROM signals
                WHERE symbol = ? AND action = ?
                ORDER BY sent_at DESC LIMIT 1
            """, (symbol, action)).fetchone()
        else:
            row = conn.execute("""
                SELECT * FROM signals
                WHERE symbol = ?
                ORDER BY sent_at DESC LIMIT 1
            """, (symbol,)).fetchone()
        return dict(row) if row else None


def get_pending_outcomes(older_than_hours: int = 24) -> list[dict]:
    cutoff = (datetime.utcnow() - timedelta(hours=older_than_hours)).isoformat()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM signals WHERE outcome = 'pending' AND sent_at <= ? AND tp IS NOT NULL",
            (cutoff,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_pending_outcomes_recent(min_age_hours: float = 1.0) -> list[dict]:
    """Pending signals old enough to have a meaningful outcome (TP or SL reached)."""
    cutoff = (datetime.utcnow() - timedelta(hours=min_age_hours)).isoformat()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM signals WHERE outcome = 'pending' AND sent_at <= ? AND tp IS NOT NULL AND sl IS NOT NULL",
            (cutoff,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_outcome_count(symbol: str | None = None) -> int:
    """Count of resolved (non-pending) outcomes, optionally filtered by symbol."""
    with _conn() as conn:
        if symbol:
            row = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE outcome != 'pending' AND symbol = ?",
                (symbol,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE outcome != 'pending'"
            ).fetchone()
        return row[0]


def update_outcome(signal_id: int, outcome: str, outcome_price: float,
                   metadata: dict | None = None) -> None:
    metadata = metadata or {}
    with _conn() as conn:
        conn.execute("""
            UPDATE signals
            SET outcome = ?, outcome_price = ?, outcome_at = ?,
                resolution_reason = ?, touched_tp = ?, touched_sl = ?,
                resolution_minutes = ?, max_favorable_pct = ?, max_adverse_pct = ?
            WHERE id = ?
        """, (
            outcome,
            outcome_price,
            datetime.utcnow().isoformat(),
            metadata.get("resolution_reason"),
            int(bool(metadata.get("touched_tp", False))),
            int(bool(metadata.get("touched_sl", False))),
            metadata.get("resolution_minutes"),
            metadata.get("max_favorable_pct"),
            metadata.get("max_adverse_pct"),
            signal_id,
        ))


def get_outcome_training_data(symbol: str | None = None) -> list[dict]:
    with _conn() as conn:
        if symbol:
            rows = conn.execute("""
                SELECT features, action, outcome FROM signals
                WHERE outcome IN ('correct', 'incorrect') AND features IS NOT NULL
                AND symbol = ?
            """, (symbol,)).fetchall()
        else:
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


def get_symbol_outcome_profile(symbol: str, lookback_days: int = 60) -> dict:
    """Recent resolved-outcome aggregates used for TP/SL tuning."""
    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT outcome, resolution_reason, max_favorable_pct, max_adverse_pct
            FROM signals
            WHERE symbol = ? AND outcome IN ('correct', 'incorrect', 'neutral')
              AND outcome_at IS NOT NULL AND outcome_at >= ?
            ORDER BY outcome_at DESC
        """, (symbol, cutoff)).fetchall()
    data = [dict(r) for r in rows]
    if not data:
        return {
            "symbol": symbol,
            "sample_size": 0,
            "avg_favorable_pct": None,
            "avg_adverse_pct": None,
            "timeout_rate": 0.0,
            "ambiguous_rate": 0.0,
        }
    mfe = [r["max_favorable_pct"] for r in data if r["max_favorable_pct"] is not None]
    mae = [abs(r["max_adverse_pct"]) for r in data if r["max_adverse_pct"] is not None]
    timeout_count = sum(1 for r in data if r.get("resolution_reason") == "timeout_no_hit")
    ambiguous_count = sum(1 for r in data if r.get("resolution_reason") == "both_hit_same_candle")
    total = len(data)
    return {
        "symbol": symbol,
        "sample_size": total,
        "avg_favorable_pct": round(sum(mfe) / len(mfe), 2) if mfe else None,
        "avg_adverse_pct": round(sum(mae) / len(mae), 2) if mae else None,
        "timeout_rate": round(timeout_count / total, 3) if total else 0.0,
        "ambiguous_rate": round(ambiguous_count / total, 3) if total else 0.0,
    }


def create_action_request(action: str, symbol: str | None = None,
                          requested_by: str = "dashboard") -> int:
    with _conn() as conn:
        cur = conn.execute("""
            INSERT INTO action_requests (action, symbol, requested_at, requested_by)
            VALUES (?, ?, ?, ?)
        """, (
            action,
            symbol.upper() if symbol else None,
            datetime.utcnow().isoformat(),
            requested_by,
        ))
        return cur.lastrowid


def get_pending_action_requests(limit: int = 20) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM action_requests
            WHERE status = 'pending'
            ORDER BY requested_at ASC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def complete_action_request(request_id: int, status: str, result_note: str = "") -> None:
    with _conn() as conn:
        conn.execute("""
            UPDATE action_requests
            SET status = ?, result_note = ?, completed_at = ?
            WHERE id = ?
        """, (status, result_note, datetime.utcnow().isoformat(), request_id))


def get_recent_action_requests(limit: int = 25) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM action_requests
            ORDER BY requested_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_ops_snapshot() -> dict:
    with _conn() as conn:
        total_pending_actions = conn.execute(
            "SELECT COUNT(*) FROM action_requests WHERE status = 'pending'"
        ).fetchone()[0]
        last_action = conn.execute("""
            SELECT action, symbol, status, requested_at, completed_at, result_note
            FROM action_requests ORDER BY id DESC LIMIT 1
        """).fetchone()
        last_signal = conn.execute("""
            SELECT symbol, action, sent_at, strength
            FROM signals ORDER BY sent_at DESC LIMIT 1
        """).fetchone()
        last_outcome = conn.execute("""
            SELECT symbol, outcome, outcome_at, resolution_reason
            FROM signals
            WHERE outcome != 'pending'
            ORDER BY outcome_at DESC LIMIT 1
        """).fetchone()
        last_cycle = conn.execute("""
            SELECT symbol, checked_at, retrained, note
            FROM ml_cycle_log ORDER BY checked_at DESC LIMIT 1
        """).fetchone()
        last_train = conn.execute("""
            SELECT symbol, trained_at, outcome_samples
            FROM ml_training_log ORDER BY trained_at DESC LIMIT 1
        """).fetchone()
        pending_outcomes = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome = 'pending'"
        ).fetchone()[0]
        active_symbols = get_active_symbols()
    return {
        "active_symbols": active_symbols,
        "active_symbol_count": len(active_symbols),
        "pending_actions": total_pending_actions,
        "pending_outcomes": pending_outcomes,
        "last_action": dict(last_action) if last_action else None,
        "last_signal": dict(last_signal) if last_signal else None,
        "last_outcome": dict(last_outcome) if last_outcome else None,
        "last_cycle": dict(last_cycle) if last_cycle else None,
        "last_train": dict(last_train) if last_train else None,
        "recent_actions": get_recent_action_requests(12),
    }


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
        correct = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome = 'correct'"
        ).fetchone()[0]
        incorrect = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome = 'incorrect'"
        ).fetchone()[0]
        resolved = correct + incorrect  # exclude neutral from accuracy denominator
        accuracy = round(correct / resolved * 100, 1) if resolved > 0 else 0

        symbol_rows = conn.execute("""
            SELECT symbol,
                   COUNT(*) AS total,
                   SUM(CASE WHEN outcome='correct'   THEN 1 ELSE 0 END) AS correct,
                   SUM(CASE WHEN outcome='incorrect' THEN 1 ELSE 0 END) AS incorrect,
                   SUM(CASE WHEN outcome='neutral'   THEN 1 ELSE 0 END) AS neutral
            FROM signals
            WHERE symbol IN (SELECT symbol FROM symbols WHERE active=1)
            GROUP BY symbol
        """).fetchall()
        per_symbol = {}
        for r in symbol_rows:
            decisive = r["correct"] + r["incorrect"]
            acc = round(r["correct"] / decisive * 100, 1) if decisive > 0 else 0
            per_symbol[r["symbol"]] = {
                "total": r["total"], "correct": r["correct"],
                "incorrect": r["incorrect"], "neutral": r["neutral"],
                "accuracy": acc, "resolved": decisive,
            }

        daily_rows = conn.execute("""
            SELECT DATE(sent_at) AS day, COUNT(*) AS count
            FROM signals
            WHERE sent_at >= DATE('now', '-14 days')
              AND symbol IN (SELECT symbol FROM symbols WHERE active=1)
            GROUP BY day ORDER BY day
        """).fetchall()

        buy_sell = conn.execute("""
            SELECT action, COUNT(*) AS cnt FROM signals
            WHERE symbol IN (SELECT symbol FROM symbols WHERE active=1)
            GROUP BY action
        """).fetchall()

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
                   SUM(CASE WHEN outcome='neutral'   THEN 1 ELSE 0 END) AS neutral
            FROM signals
            WHERE sent_at >= ?
              AND symbol IN (SELECT symbol FROM symbols WHERE active=1)
            GROUP BY symbol, action
        """, (cutoff,)).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE sent_at >= ?", (cutoff,)
        ).fetchone()[0]
        correct_total = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome = 'correct' AND sent_at >= ?", (cutoff,)
        ).fetchone()[0]
        incorrect_total = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome = 'incorrect' AND sent_at >= ?", (cutoff,)
        ).fetchone()[0]
        resolved_total = correct_total + incorrect_total
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


# ── User preferences ─────────────────────────────────────────────────────────

def get_user_pref(chat_id: str | int) -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT lang, mode FROM user_preferences WHERE chat_id = ?",
            (str(chat_id),)
        ).fetchone()
        return {"lang": row["lang"], "mode": row["mode"]} if row else {"lang": "en", "mode": "beginner"}


def set_user_pref(chat_id: str | int, lang: str | None = None, mode: str | None = None) -> None:
    current = get_user_pref(chat_id)
    new_lang = lang or current["lang"]
    new_mode = mode or current["mode"]
    with _conn() as conn:
        conn.execute("""
            INSERT INTO user_preferences (chat_id, lang, mode) VALUES (?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET lang = ?, mode = ?
        """, (str(chat_id), new_lang, new_mode, new_lang, new_mode))


# ── Symbol management ─────────────────────────────────────────────────────────

def seed_symbols(symbols: list[str]) -> None:
    """Insert symbols from env only on the very first run (empty table)."""
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        if count > 0:
            return  # already configured — never override user's symbol choices
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
    """Returns True if newly added (or re-activated), False if already active."""
    sym = symbol.upper()
    now = datetime.utcnow().isoformat()
    from signals import detect_market
    market = detect_market(sym)
    with _conn() as conn:
        existing = conn.execute(
            "SELECT active FROM symbols WHERE symbol = ?", (sym,)
        ).fetchone()
        if existing:
            if not existing["active"]:
                conn.execute(
                    "UPDATE symbols SET active = 1, market = ?, added_at = ? WHERE symbol = ?",
                    (market, now, sym)
                )
                return True
            return False
        conn.execute(
            "INSERT INTO symbols (symbol, active, market, added_at, added_by) VALUES (?, 1, ?, ?, ?)",
            (sym, market, now, added_by)
        )
        return True


def remove_symbol(symbol: str) -> bool:
    """Soft-delete (sets active=0) and clears sentiment cache. Returns True if found."""
    sym = symbol.upper()
    with _conn() as conn:
        row = conn.execute(
            "SELECT symbol FROM symbols WHERE symbol = ?", (sym,)
        ).fetchone()
        if not row:
            return False
        conn.execute("UPDATE symbols SET active = 0 WHERE symbol = ?", (sym,))
        conn.execute("DELETE FROM sentiment_cache WHERE symbol = ?", (sym,))
        return True


def delete_symbol(symbol: str) -> bool:
    """Hard-delete symbol from symbols table. Signal history is preserved. Returns True if found."""
    sym = symbol.upper()
    with _conn() as conn:
        row = conn.execute(
            "SELECT symbol FROM symbols WHERE symbol = ?", (sym,)
        ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM symbols WHERE symbol = ?", (sym,))
        conn.execute("DELETE FROM sentiment_cache WHERE symbol = ?", (sym,))
        return True


def get_all_symbols_with_status() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT symbol, active, added_at, added_by, market FROM symbols ORDER BY added_at"
        ).fetchall()
        return [dict(r) for r in rows]


def get_symbol_market(symbol: str) -> str | None:
    """Return stored market override for symbol, or None if not set."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT market FROM symbols WHERE symbol = ?", (symbol.upper(),)
        ).fetchone()
        return row["market"] if row else None


def set_symbol_market(symbol: str, market: str) -> None:
    """Set market type for a symbol: 'us', 'xetra', or 'crypto'."""
    with _conn() as conn:
        conn.execute(
            "UPDATE symbols SET market = ? WHERE symbol = ?",
            (market or None, symbol.upper())
        )


# ── ML training log ───────────────────────────────────────────────────────────

def log_training(symbol: str, train_samples: int, outcome_samples: int = 0,
                 trigger: str = "time", win_rate: float | None = None,
                 correct_count: int = 0, incorrect_count: int = 0,
                 neutral_count: int = 0, job_id: int | None = None) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO ml_training_log
                (symbol, trained_at, train_samples, outcome_samples,
                 trigger, win_rate, correct_count, incorrect_count, neutral_count, job_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, datetime.utcnow().isoformat(), train_samples, outcome_samples,
              trigger, win_rate, correct_count, incorrect_count, neutral_count, job_id))


def start_training_job(job_type: str, total_symbols: int) -> int:
    with _conn() as conn:
        cur = conn.execute("""
            INSERT INTO ml_training_jobs (job_type, status, started_at, total_symbols)
            VALUES (?, 'running', ?, ?)
        """, (job_type, datetime.utcnow().isoformat(), total_symbols))
        return cur.lastrowid


def update_training_job(job_id: int, done_symbols: int,
                        current_symbol: str | None = None) -> None:
    with _conn() as conn:
        conn.execute("""
            UPDATE ml_training_jobs
            SET done_symbols = ?, current_symbol = ?
            WHERE id = ?
        """, (done_symbols, current_symbol, job_id))


def finish_training_job(job_id: int, status: str = "done",
                        result_summary: str | None = None, note: str = "") -> None:
    with _conn() as conn:
        conn.execute("""
            UPDATE ml_training_jobs
            SET status = ?, completed_at = ?, result_summary = ?, note = ?,
                current_symbol = NULL
            WHERE id = ?
        """, (status, datetime.utcnow().isoformat(), result_summary, note, job_id))


def get_running_training_job() -> dict | None:
    with _conn() as conn:
        row = conn.execute("""
            SELECT * FROM ml_training_jobs WHERE status = 'running'
            ORDER BY started_at DESC LIMIT 1
        """).fetchone()
        return dict(row) if row else None


def get_recent_training_jobs(limit: int = 10) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM ml_training_jobs ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def log_learning_cycle(symbol: str, retrain_needed: bool, retrained: bool,
                       resolved_outcomes: int, new_outcomes: int, note: str = "") -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO ml_cycle_log
                (symbol, checked_at, retrain_needed, retrained, resolved_outcomes, new_outcomes, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            datetime.utcnow().isoformat(),
            int(retrain_needed),
            int(retrained),
            resolved_outcomes,
            new_outcomes,
            note,
        ))


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
              AND symbol IN (SELECT symbol FROM symbols WHERE active=1)
            ORDER BY sent_at
        """).fetchall()

        # Training log — last bootstrap entry per symbol (for win_rate) + last overall entry
        train_rows = conn.execute("""
            SELECT symbol, trained_at, train_samples, outcome_samples,
                   win_rate, correct_count, incorrect_count, trigger
            FROM ml_training_log
            WHERE id IN (
                SELECT MAX(id) FROM ml_training_log GROUP BY symbol
            )
        """).fetchall()
        bootstrap_rows = conn.execute("""
            SELECT symbol, win_rate, correct_count, incorrect_count, train_samples
            FROM ml_training_log
            WHERE trigger = 'bootstrap'
              AND id IN (
                SELECT MAX(id) FROM ml_training_log
                WHERE trigger = 'bootstrap' GROUP BY symbol
              )
        """).fetchall()
        bootstrap_stats = {r["symbol"]: dict(r) for r in bootstrap_rows}
        training = {r["symbol"]: dict(r) for r in train_rows}
        active_symbols = set(get_active_symbols())
        # Only show active symbols — inactive ones are excluded from all views
        all_symbols = active_symbols

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
            bs = bootstrap_stats.get(sym)
            d["bootstrap_win_rate"] = bs["win_rate"] if bs else None
            d["bootstrap_correct"] = bs["correct_count"] if bs else None
            d["bootstrap_incorrect"] = bs["incorrect_count"] if bs else None
            d["bootstrap_samples"] = bs["train_samples"] if bs else None

        for sym in sorted(all_symbols):
            if sym not in per_symbol:
                bs = bootstrap_stats.get(sym)
                per_symbol[sym] = {
                    "ai_total": 0, "ai_correct": 0,
                    "strong_total": 0, "strong_correct": 0,
                    "rule_total": 0, "rule_correct": 0,
                    "ai_accuracy": None,
                    "strong_accuracy": None,
                    "rule_accuracy": None,
                    "training": training.get(sym),
                    "train_runs": run_map.get(sym, 0),
                    "bootstrap_win_rate": bs["win_rate"] if bs else None,
                    "bootstrap_correct": bs["correct_count"] if bs else None,
                    "bootstrap_incorrect": bs["incorrect_count"] if bs else None,
                    "bootstrap_samples": bs["train_samples"] if bs else None,
                }

        for lbl, b in buckets_global.items():
            b["accuracy"] = (
                round(b["correct"] / b["total"] * 100, 1) if b["total"] > 0 else None
            )

        trained_model_count = sum(1 for sym in active_symbols if training.get(sym))
        ai_totals = [d["ai_total"] for d in per_symbol.values()]
        ai_acc_values = [d["ai_accuracy"] for d in per_symbol.values()
                         if d["ai_accuracy"] is not None and d["ai_total"] >= 5]
        return {
            "per_symbol": per_symbol,
            "confidence_buckets": [
                {"label": lbl, **buckets_global[lbl]} for lbl in bucket_labels
            ],
            "summary": {
                "active_symbol_count": len(active_symbols),
                "trained_model_count": trained_model_count,
                "ai_resolved_signals": sum(ai_totals),
                "avg_ai_accuracy": round(sum(ai_acc_values) / len(ai_acc_values), 1) if ai_acc_values else None,
            },
        }


def get_ml_activity_log(limit: int = 40, symbol: str | None = None,
                        days: int | None = None) -> dict:
    """
    Returns:
    - training_events: recent entries from ml_training_log
    - recent_outcomes: last N resolved signals (for outcome feed)
    - symbol_health: per-symbol accuracy trend and model age
    """
    filters = []
    params: list = []
    cycle_filters = []
    cycle_params: list = []
    outcome_filters = []
    outcome_params: list = []
    if symbol:
        filters.append("symbol = ?")
        params.append(symbol.upper())
        cycle_filters.append("symbol = ?")
        cycle_params.append(symbol.upper())
        outcome_filters.append("symbol = ?")
        outcome_params.append(symbol.upper())
    if days and days > 0:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        filters.append("trained_at >= ?")
        params.append(cutoff)
        cycle_filters.append("checked_at >= ?")
        cycle_params.append(cutoff)
        outcome_filters.append("outcome_at >= ?")
        outcome_params.append(cutoff)
    train_where = f"WHERE {' AND '.join(filters)}" if filters else ""

    with _conn() as conn:
        train_rows = conn.execute("""
            SELECT symbol, trained_at, train_samples, outcome_samples
            FROM ml_training_log
            {train_where}
            ORDER BY id DESC LIMIT ?
        """.format(train_where=train_where), (*params, limit)).fetchall()

        cycle_rows = conn.execute("""
            SELECT symbol, checked_at, retrain_needed, retrained,
                   resolved_outcomes, new_outcomes, note
            FROM ml_cycle_log
            {cycle_where}
            ORDER BY id DESC LIMIT ?
        """.format(
            cycle_where=("WHERE " + " AND ".join(cycle_filters)) if cycle_filters else ""
        ), (*cycle_params, limit)).fetchall()

        outcome_rows = conn.execute("""
            SELECT symbol, action, strength, price, outcome, outcome_at,
                   ai_confidence, reasons, sent_at, resolution_reason,
                   touched_tp, touched_sl, resolution_minutes,
                   max_favorable_pct, max_adverse_pct
            FROM signals
            WHERE outcome IN ('correct', 'incorrect', 'neutral')
              AND symbol IN (SELECT symbol FROM symbols WHERE active=1)
            {extra_where}
            ORDER BY outcome_at DESC LIMIT 60
        """.format(
            extra_where=(" AND " + " AND ".join(outcome_filters)) if outcome_filters else ""
        ), outcome_params).fetchall()

        # Per-symbol: last 10 resolved outcomes for accuracy trend
        sym_rows = conn.execute("""
            SELECT symbol, outcome, resolution_reason, touched_tp, touched_sl,
                   resolution_minutes, max_favorable_pct, max_adverse_pct
            FROM signals
            WHERE outcome IN ('correct', 'incorrect')
            {extra_where}
            ORDER BY outcome_at DESC
        """.format(
            extra_where=(" AND " + " AND ".join(outcome_filters)) if outcome_filters else ""
        ), outcome_params).fetchall()

        # Pending count per symbol
        pending_rows = conn.execute("""
            SELECT symbol, COUNT(*) AS cnt FROM signals
            WHERE outcome = 'pending'
            {extra_where}
            GROUP BY symbol
        """.format(
            extra_where=(" AND symbol = ?" if symbol else "")
        ), ([symbol.upper()] if symbol else [])).fetchall()

        # Last training per symbol
        last_train = conn.execute("""
            SELECT symbol, MAX(trained_at) AS last_trained,
                   MAX(outcome_samples) AS last_outcome_samples
            FROM ml_training_log
            {train_where}
            GROUP BY symbol
        """.format(train_where=train_where), params).fetchall()

        last_cycle = conn.execute("""
            SELECT symbol, MAX(checked_at) AS last_checked
            FROM ml_cycle_log
            {cycle_where}
            GROUP BY symbol
        """.format(
            cycle_where=("WHERE " + " AND ".join(cycle_filters)) if cycle_filters else ""
        ), cycle_params).fetchall()

        available_symbols_rows = conn.execute("""
            SELECT DISTINCT symbol FROM signals ORDER BY symbol
        """).fetchall()

    # Build per-symbol health
    sym_outcomes: dict[str, list[dict]] = {}
    for r in sym_rows:
        sym_outcomes.setdefault(r["symbol"], []).append(dict(r))

    pending_map = {r["symbol"]: r["cnt"] for r in pending_rows}
    last_train_map = {r["symbol"]: dict(r) for r in last_train}
    last_cycle_map = {r["symbol"]: dict(r) for r in last_cycle}
    all_active_symbols = set(get_active_symbols())

    symbol_health = {}
    resolution_reason_counts: dict[str, int] = {}
    # Only include active symbols — filter out inactive from all merged sets
    all_symbols = sorted((set(sym_outcomes.keys()) | set(last_train_map.keys()) | set(last_cycle_map.keys()) | all_active_symbols) & all_active_symbols)
    for sym in all_symbols:
        outcomes = sym_outcomes.get(sym, [])
        last10 = outcomes[:10]
        correct = sum(1 for o in last10 if o["outcome"] == "correct")
        recent_acc = round(correct / len(last10) * 100, 1) if last10 else 0.0
        all_correct = sum(1 for o in outcomes if o["outcome"] == "correct")
        all_acc = round(all_correct / len(outcomes) * 100, 1) if outcomes else 0.0
        trend = "improving" if len(last10) >= 5 and recent_acc > all_acc else (
            "declining" if len(last10) >= 5 and recent_acc < all_acc - 5 else "stable"
        )
        lt = last_train_map.get(sym, {})
        lc = last_cycle_map.get(sym, {})
        recent_neutral_like = sum(
            1 for o in last10 if o.get("resolution_reason") in ("both_hit_same_candle", "timeout_no_hit")
        )
        ambiguous_count = sum(1 for o in outcomes if o.get("resolution_reason") == "both_hit_same_candle")
        timeout_count = sum(1 for o in outcomes if o.get("resolution_reason") == "timeout_no_hit")
        resolution_samples = [o["resolution_minutes"] for o in outcomes if o.get("resolution_minutes") is not None]
        mfe_samples = [o["max_favorable_pct"] for o in outcomes if o.get("max_favorable_pct") is not None]
        mae_samples = [o["max_adverse_pct"] for o in outcomes if o.get("max_adverse_pct") is not None]
        for o in outcomes:
            reason = o.get("resolution_reason") or "unknown"
            resolution_reason_counts[reason] = resolution_reason_counts.get(reason, 0) + 1
        symbol_health[sym] = {
            "recent_acc": recent_acc,
            "all_acc": all_acc,
            "recent_count": len(last10),
            "total_outcomes": len(outcomes),
            "recent_successes": sum(1 for o in last10 if o["outcome"] == "correct"),
            "recent_failures": sum(1 for o in last10 if o["outcome"] == "incorrect"),
            "trend": trend,
            "pending": pending_map.get(sym, 0),
            "last_trained": lt.get("last_trained"),
            "last_outcome_samples": lt.get("last_outcome_samples", 0),
            "last_checked": lc.get("last_checked"),
            "ambiguous_count": ambiguous_count,
            "timeout_count": timeout_count,
            "recent_unclear": recent_neutral_like,
            "avg_resolution_minutes": (
                round(sum(resolution_samples) / len(resolution_samples), 1)
                if resolution_samples else None
            ),
            "avg_favorable_pct": (
                round(sum(mfe_samples) / len(mfe_samples), 2)
                if mfe_samples else None
            ),
            "avg_adverse_pct": (
                round(sum(mae_samples) / len(mae_samples), 2)
                if mae_samples else None
            ),
        }

    resolution_reason_breakdown = [
        {"reason": reason, "count": count}
        for reason, count in sorted(
            resolution_reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    resolution_time_by_symbol = [
        {"symbol": sym, "avg_resolution_minutes": data["avg_resolution_minutes"]}
        for sym, data in sorted(symbol_health.items())
        if data.get("avg_resolution_minutes") is not None
    ]

    return {
        "training_events": [dict(r) for r in train_rows],
        "cycle_events": [dict(r) for r in cycle_rows],
        "recent_outcomes": [
            {**dict(r), "reasons": json.loads(r["reasons"] or "[]")}
            for r in outcome_rows
        ],
        "symbol_health": symbol_health,
        "resolution_reason_breakdown": resolution_reason_breakdown,
        "resolution_time_by_symbol": resolution_time_by_symbol,
        "mfe_mae_by_symbol": [
            {
                "symbol": sym,
                "avg_favorable_pct": data["avg_favorable_pct"],
                "avg_adverse_pct": abs(data["avg_adverse_pct"]) if data["avg_adverse_pct"] is not None else None,
            }
            for sym, data in sorted(symbol_health.items())
            if data.get("avg_favorable_pct") is not None or data.get("avg_adverse_pct") is not None
        ],
        "available_symbols": [r["symbol"] for r in available_symbols_rows],
        "filters": {"symbol": symbol.upper() if symbol else "", "days": days or 0},
        "running_job": get_running_training_job(),
        "recent_jobs": get_recent_training_jobs(5),
    }


# ── App settings ──────────────────────────────────────────────────────────────

_SETTING_DEFAULTS = {
    "sentiment_provider": "disabled",          # disabled / local_finbert / gemini / claude
    "sentiment_local_url": "http://finbert:5001",
    "gemini_api_key": "",
    "claude_api_key": "",
    "sentiment_suppress_threshold": "0.35",
    "news_provider": "disabled",               # disabled / finnhub
    "finnhub_api_key": "",
    "news_lookback_hours": "6",
    "outcome_notify_admin": "false",           # true / false
    "display_currency": "USD",                 # USD / EUR / GBP / CHF
    "default_market": "us",                    # us / xetra
}


def get_setting(key: str, default: str = "") -> str:
    fallback = _SETTING_DEFAULTS.get(key, default)
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else fallback


def set_setting(key: str, value: str) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute("""
            INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
        """, (key, value, now, value, now))


def get_all_settings() -> dict[str, str]:
    with _conn() as conn:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        stored = {r["key"]: r["value"] for r in rows}
    return {k: stored.get(k, v) for k, v in _SETTING_DEFAULTS.items()}


# ── Sentiment cache ───────────────────────────────────────────────────────────

def set_sentiment_cache(symbol: str, result: dict) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute("""
            INSERT INTO sentiment_cache (symbol, score, label, source, headlines, computed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                score = ?, label = ?, source = ?, headlines = ?, computed_at = ?
        """, (
            symbol.upper(),
            result.get("score", 0.0),
            result.get("label", "neutral"),
            result.get("source", "disabled"),
            json.dumps(result.get("headlines", [])),
            now,
            result.get("score", 0.0),
            result.get("label", "neutral"),
            result.get("source", "disabled"),
            json.dumps(result.get("headlines", [])),
            now,
        ))


def get_sentiment_cache(symbol: str, max_age_minutes: int = 30) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(minutes=max_age_minutes)).isoformat()
    with _conn() as conn:
        row = conn.execute("""
            SELECT score, label, source, headlines, computed_at
            FROM sentiment_cache
            WHERE symbol = ? AND computed_at >= ?
        """, (symbol.upper(), cutoff)).fetchone()
    if not row:
        return None
    return {
        "score": row["score"],
        "label": row["label"],
        "source": row["source"],
        "headlines": json.loads(row["headlines"] or "[]"),
        "computed_at": row["computed_at"],
    }


def get_all_sentiment_cache() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT symbol, score, label, source, computed_at
            FROM sentiment_cache ORDER BY symbol
        """).fetchall()
    return [dict(r) for r in rows]


def get_accuracy_by_direction() -> dict:
    """Per-symbol, per-direction (BUY/SELL) accuracy for resolved signals."""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT symbol, action,
                   COUNT(*) AS total,
                   SUM(CASE WHEN outcome='correct'   THEN 1 ELSE 0 END) AS correct,
                   SUM(CASE WHEN outcome='incorrect' THEN 1 ELSE 0 END) AS incorrect
            FROM signals
            WHERE outcome IN ('correct', 'incorrect')
              AND symbol IN (SELECT symbol FROM symbols WHERE active=1)
            GROUP BY symbol, action
        """).fetchall()
    result: dict[str, dict] = {}
    for r in rows:
        sym = r["symbol"]
        decisive = r["correct"] + r["incorrect"]
        result.setdefault(sym, {})[r["action"]] = {
            "total": r["total"],
            "correct": r["correct"],
            "incorrect": r["incorrect"],
            "accuracy": round(r["correct"] / decisive * 100, 1) if decisive > 0 else None,
        }
    return result


def log_scan(scanned_at: str, symbols_total: int, symbols_scanned: int,
             symbols_skipped_market: int, symbols_skipped_brake: int,
             signals_generated: int, scan_duration_ms: float,
             regime: str | None = None, perf_brake_engaged: bool = False) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO scan_log
                (scanned_at, symbols_total, symbols_scanned, symbols_skipped_market,
                 symbols_skipped_brake, signals_generated, scan_duration_ms,
                 regime, perf_brake_engaged)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (scanned_at, symbols_total, symbols_scanned, symbols_skipped_market,
              symbols_skipped_brake, signals_generated, scan_duration_ms,
              regime, int(perf_brake_engaged)))


def get_recent_scans(limit: int = 20) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM scan_log ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_last_scan() -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM scan_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def log_pipeline_trace(symbol: str, scanned_at: str, signal_id: int | None = None,
                       data_ok: bool = True, features_count: int = 0,
                       rule_signal: str | None = None, ml_buy_prob: float | None = None,
                       ml_sell_prob: float | None = None, meta_conf: float | None = None,
                       sentiment_score: float | None = None, sentiment_adj: float | None = None,
                       suppressed: bool = False, suppress_reason: str | None = None,
                       final_action: str | None = None) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO pipeline_trace
                (signal_id, symbol, scanned_at, data_ok, features_count,
                 rule_signal, ml_buy_prob, ml_sell_prob, meta_conf,
                 sentiment_score, sentiment_adj, suppressed, suppress_reason, final_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (signal_id, symbol.upper(), scanned_at, int(data_ok), features_count,
              rule_signal, ml_buy_prob, ml_sell_prob, meta_conf,
              sentiment_score, sentiment_adj, int(suppressed), suppress_reason, final_action))


def get_pipeline_trace(symbol: str, limit: int = 10) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM pipeline_trace WHERE symbol = ?
            ORDER BY id DESC LIMIT ?
        """, (symbol.upper(), limit)).fetchall()
        return [dict(r) for r in rows]


def get_event_timeline(limit: int = 100, event_types: list[str] | None = None) -> list[dict]:
    """
    Unified chronological event feed across all system activity.
    event_types filter: 'signal', 'outcome', 'train', 'cycle', 'action', 'scan'
    """
    all_types = event_types or ['signal', 'outcome', 'train', 'cycle', 'action', 'scan']
    events: list[dict] = []

    with _conn() as conn:
        if 'signal' in all_types:
            rows = conn.execute("""
                SELECT 'signal' AS type, sent_at AS ts,
                       symbol, action, strength, price, ai_confidence, outcome, regime
                FROM signals ORDER BY sent_at DESC LIMIT ?
            """, (limit,)).fetchall()
            events.extend(dict(r) for r in rows)

        if 'outcome' in all_types:
            rows = conn.execute("""
                SELECT 'outcome' AS type, outcome_at AS ts,
                       symbol, action, outcome, resolution_reason,
                       max_favorable_pct, max_adverse_pct, resolution_minutes
                FROM signals
                WHERE outcome != 'pending' AND outcome_at IS NOT NULL
                ORDER BY outcome_at DESC LIMIT ?
            """, (limit,)).fetchall()
            events.extend(dict(r) for r in rows)

        if 'train' in all_types:
            rows = conn.execute("""
                SELECT 'train' AS type, trained_at AS ts,
                       symbol, train_samples, outcome_samples, trigger, win_rate,
                       correct_count, incorrect_count
                FROM ml_training_log ORDER BY trained_at DESC LIMIT ?
            """, (limit,)).fetchall()
            events.extend(dict(r) for r in rows)

        if 'cycle' in all_types:
            rows = conn.execute("""
                SELECT 'cycle' AS type, checked_at AS ts,
                       symbol, retrain_needed, retrained,
                       resolved_outcomes, new_outcomes, note
                FROM ml_cycle_log ORDER BY checked_at DESC LIMIT ?
            """, (limit,)).fetchall()
            events.extend(dict(r) for r in rows)

        if 'action' in all_types:
            rows = conn.execute("""
                SELECT 'action' AS type, requested_at AS ts,
                       action, symbol, status, result_note, requested_by
                FROM action_requests ORDER BY requested_at DESC LIMIT ?
            """, (limit,)).fetchall()
            events.extend(dict(r) for r in rows)

        if 'scan' in all_types:
            rows = conn.execute("""
                SELECT 'scan' AS type, scanned_at AS ts,
                       symbols_total, symbols_scanned, symbols_skipped_market,
                       signals_generated, scan_duration_ms, regime, perf_brake_engaged
                FROM scan_log ORDER BY scanned_at DESC LIMIT ?
            """, (limit,)).fetchall()
            events.extend(dict(r) for r in rows)

    # Sort all events by timestamp descending, return top `limit`
    def _ts(e: dict) -> str:
        return e.get("ts") or ""
    events.sort(key=_ts, reverse=True)
    return events[:limit]


def log_wf_result(symbol: str, result: dict) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO ml_wf_results
                (symbol, tested_at, train_bars, test_bars,
                 rule_signals, rule_win_rate, ai_signals, ai_win_rate, grade)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            datetime.utcnow().isoformat(),
            result.get("train_bars"),
            result.get("test_bars"),
            result.get("rule_signals"),
            result.get("rule_win_rate"),
            result.get("ai_signals"),
            result.get("ai_win_rate"),
            result.get("grade"),
        ))


def get_wf_results() -> list[dict]:
    """Most recent walk-forward result per symbol."""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM ml_wf_results
            WHERE id IN (
                SELECT MAX(id) FROM ml_wf_results GROUP BY symbol
            )
            ORDER BY tested_at DESC
        """).fetchall()
        return [dict(r) for r in rows]


def get_recent_signals(limit: int = 50, include_all_fields: bool = False) -> list[dict]:
    if include_all_fields:
        cols = "*"
    else:
        cols = ("id, symbol, action, strength, price, rsi, ai_confidence, "
                "reasons, sent_at, outcome, outcome_price, outcome_at, "
                "tp, sl, tp_pct, sl_pct, rr, trend, quality, vol_spike, "
                "regime, features, suggested_size_pct, resolution_reason, "
                "resolution_minutes, max_favorable_pct, max_adverse_pct")
    with _conn() as conn:
        rows = conn.execute(f"""
            SELECT {cols}
            FROM signals
            WHERE symbol IN (SELECT symbol FROM symbols WHERE active=1)
            ORDER BY sent_at DESC LIMIT ?
        """, (limit,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["reasons"] = json.loads(d["reasons"] or "[]")
            if "features" in d:
                try:
                    d["features"] = json.loads(d["features"] or "{}")
                except Exception:
                    d["features"] = {}
            result.append(d)
        return result


def get_rolling_performance(days: int = 30, window: int = 20) -> dict:
    """
    Rolling win rate over time — used for equity curve and performance trend charts.

    Returns:
      - daily: [{date, win_rate_7d, win_rate_30d, correct, incorrect}] — one row per day
      - by_strength: {STRONG/RULE/AI: {total, correct, win_rate}} over the last `days`
      - by_symbol: {symbol: {total, correct, win_rate}} over the last `days`
      - rolling_20: [bool] — last 20 resolved outcomes for performance brake display
    """
    with _conn() as conn:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # All resolved outcomes ordered by outcome_at
        all_rows = conn.execute("""
            SELECT DATE(outcome_at) AS day, outcome, strength, symbol
            FROM signals
            WHERE outcome IN ('correct', 'incorrect')
              AND symbol IN (SELECT symbol FROM symbols WHERE active=1)
            ORDER BY outcome_at ASC
        """).fetchall()

        # Rolling 7-day and 30-day win rate per calendar day
        from collections import deque
        daily_map: dict[str, dict] = {}
        buf_7: deque = deque()
        buf_30: deque = deque()

        for r in all_rows:
            day = r["day"]
            is_correct = r["outcome"] == "correct"
            buf_7.append((day, is_correct))
            buf_30.append((day, is_correct))
            # Trim to 7-day window
            while buf_7 and buf_7[0][0] < (
                datetime.fromisoformat(day) - timedelta(days=7)
            ).date().isoformat():
                buf_7.popleft()
            while buf_30 and buf_30[0][0] < (
                datetime.fromisoformat(day) - timedelta(days=30)
            ).date().isoformat():
                buf_30.popleft()

            if day not in daily_map:
                daily_map[day] = {"correct": 0, "incorrect": 0}
            daily_map[day]["correct" if is_correct else "incorrect"] += 1

            wr7  = round(sum(1 for _, v in buf_7  if v) / max(1, len(buf_7))  * 100, 1)
            wr30 = round(sum(1 for _, v in buf_30 if v) / max(1, len(buf_30)) * 100, 1)
            daily_map[day]["win_rate_7d"]  = wr7
            daily_map[day]["win_rate_30d"] = wr30

        daily = [{"date": d, **v} for d, v in sorted(daily_map.items())]

        # By strength (last `days`)
        strength_rows = conn.execute("""
            SELECT strength,
                   COUNT(*) AS total,
                   SUM(CASE WHEN outcome='correct' THEN 1 ELSE 0 END) AS correct
            FROM signals
            WHERE outcome IN ('correct','incorrect')
              AND outcome_at >= ?
              AND symbol IN (SELECT symbol FROM symbols WHERE active=1)
            GROUP BY strength
        """, (cutoff,)).fetchall()
        by_strength = {}
        for r in strength_rows:
            s = r["strength"] or "RULE"
            wr = round(r["correct"] / r["total"] * 100, 1) if r["total"] else 0
            by_strength[s] = {"total": r["total"], "correct": r["correct"], "win_rate": wr}

        # By symbol (last `days`)
        sym_rows = conn.execute("""
            SELECT symbol,
                   COUNT(*) AS total,
                   SUM(CASE WHEN outcome='correct' THEN 1 ELSE 0 END) AS correct
            FROM signals
            WHERE outcome IN ('correct','incorrect')
              AND outcome_at >= ?
              AND symbol IN (SELECT symbol FROM symbols WHERE active=1)
            GROUP BY symbol
        """, (cutoff,)).fetchall()
        by_symbol = {}
        for r in sym_rows:
            wr = round(r["correct"] / r["total"] * 100, 1) if r["total"] else 0
            by_symbol[r["symbol"]] = {"total": r["total"], "correct": r["correct"], "win_rate": wr}

        # Last `window` outcomes for performance brake status
        recent = conn.execute("""
            SELECT outcome FROM signals
            WHERE outcome IN ('correct','incorrect')
              AND symbol IN (SELECT symbol FROM symbols WHERE active=1)
            ORDER BY outcome_at DESC LIMIT ?
        """, (window,)).fetchall()
        rolling = [r["outcome"] == "correct" for r in reversed(recent)]

    return {
        "daily": daily,
        "by_strength": by_strength,
        "by_symbol": by_symbol,
        "rolling_20": rolling,
    }


def get_equity_curve(trade_size: float = 1000.0) -> dict:
    """
    Simulated equity curve treating every resolved signal as `trade_size` USD.

    Win:  +trade_size * tp_pct / 100
    Loss: -trade_size * sl_pct / 100
    Neutral: 0

    Returns:
      - curve: [{date, equity, drawdown_pct}]  — cumulative P&L over time
      - peak: float — all-time equity peak
      - current: float — current equity
      - max_drawdown_pct: float — worst peak-to-trough drawdown
      - total_trades: int
      - win_rate: float
    """
    with _conn() as conn:
        rows = conn.execute("""
            SELECT outcome, tp_pct, sl_pct, outcome_at, strength, symbol
            FROM signals
            WHERE outcome IN ('correct', 'incorrect', 'neutral')
              AND outcome_at IS NOT NULL
              AND symbol IN (SELECT symbol FROM symbols WHERE active=1)
            ORDER BY outcome_at ASC
        """).fetchall()

    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    correct = 0
    total   = 0
    curve   = []

    for r in rows:
        tp_pct = r["tp_pct"] or 0.0
        sl_pct = r["sl_pct"] or 0.0

        if r["outcome"] == "correct":
            pnl = trade_size * tp_pct / 100
            correct += 1
            total += 1
        elif r["outcome"] == "incorrect":
            pnl = -(trade_size * sl_pct / 100)
            total += 1
        else:
            pnl = 0.0

        equity += pnl
        if equity > peak:
            peak = equity
        dd = round((peak - equity) / max(1.0, peak + trade_size) * 100, 2) if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

        curve.append({
            "date": r["outcome_at"][:10],
            "equity": round(equity, 2),
            "drawdown_pct": dd,
            "strength": r["strength"],
            "symbol": r["symbol"],
            "outcome": r["outcome"],
        })

    return {
        "curve": curve,
        "peak": round(peak, 2),
        "current": round(equity, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": total,
        "win_rate": round(correct / total * 100, 1) if total else 0,
    }


def get_regime_attribution(days: int = 90) -> list[dict]:
    """Win rate and signal count per market regime over the last N days."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT regime,
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome='correct' THEN 1 ELSE 0 END) as correct
            FROM signals
            WHERE sent_at >= ? AND outcome != 'pending' AND regime IS NOT NULL
            GROUP BY regime
            ORDER BY total DESC
        """, (cutoff,)).fetchall()
    return [
        {
            "regime": r["regime"],
            "total": r["total"],
            "correct": r["correct"],
            "win_rate": round(r["correct"] / r["total"] * 100, 1) if r["total"] else 0,
        }
        for r in rows
    ]
