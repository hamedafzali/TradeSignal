"""
Bilingual (English / Persian) message templates.
Supports two modes: 'beginner' and 'expert'.

Usage:
    from lang import t, signal_msg, tp_sl_alert, sell_alert, welcome_msg
    text = t("status_no_positions", lang="fa")
    msg  = signal_msg(sig, lang="fa", mode="beginner")
"""

from __future__ import annotations


# ── Generic strings ────────────────────────────────────────────────────────────

_S: dict[str, dict[str, str]] = {
    # Settings menu
    "settings_title": {
        "en": "⚙️ *Settings*\nChoose your language and experience level:",
        "fa": "⚙️ *تنظیمات*\nزبان و سطح تجربه خود را انتخاب کنید:",
    },
    "settings_saved_lang_en": {
        "en": "✅ Language set to *English*",
        "fa": "✅ زبان به *انگلیسی* تغییر یافت",
    },
    "settings_saved_lang_fa": {
        "en": "✅ Language set to *Persian* 🇮🇷",
        "fa": "✅ زبان به *فارسی* تغییر یافت 🇮🇷",
    },
    "settings_saved_beginner": {
        "en": "✅ Mode set to *Beginner* — simple, easy-to-read messages",
        "fa": "✅ حالت *مبتدی* انتخاب شد — پیام‌های ساده و قابل فهم",
    },
    "settings_saved_expert": {
        "en": "✅ Mode set to *Expert* — full technical details",
        "fa": "✅ حالت *حرفه‌ای* انتخاب شد — تمام جزئیات فنی",
    },
    "btn_lang_en":      {"en": "🇬🇧 English",  "fa": "🇬🇧 انگلیسی"},
    "btn_lang_fa":      {"en": "🇮🇷 Persian",  "fa": "🇮🇷 فارسی"},
    "btn_beginner":     {"en": "📗 Beginner",   "fa": "📗 مبتدی"},
    "btn_expert":       {"en": "📘 Expert",      "fa": "📘 حرفه‌ای"},

    # Start / welcome
    "welcome": {
        "en": (
            "📈 *Trading Signals Bot*\n\n"
            "We scan the market every few minutes and send *BUY / SELL* signals "
            "with entry price, target, and stop loss.\n\n"
            "Tap *Track this trade* on any BUY signal to get automatic alerts "
            "when your target or stop is reached.\n\n"
            "Use /settings to choose your language and experience level.\n\n"
            "Buttons below or /menu to navigate."
        ),
        "fa": (
            "📈 *ربات سیگنال معاملاتی*\n\n"
            "ما هر چند دقیقه بازار را اسکن می‌کنیم و سیگنال‌های *خرید / فروش* "
            "با قیمت ورود، هدف قیمتی و حد ضرر ارسال می‌کنیم.\n\n"
            "روی *Track this trade* در هر سیگنال خرید بزنید تا هنگام رسیدن به هدف "
            "یا فعال شدن حد ضرر، آلرت خودکار دریافت کنید.\n\n"
            "از /settings برای انتخاب زبان و سطح تجربه استفاده کنید.\n\n"
            "از دکمه‌های پایین یا /menu برای ناوبری استفاده کنید."
        ),
    },
    "position_tracked": {
        "en": (
            "✅ *{symbol}* position tracked at `${price:.2f}`\n\n"
            "You'll get a private alert when:\n"
            "  • 🎯 Target price is reached\n"
            "  • 🛑 Stop loss is triggered\n"
            "  • 🔔 A SELL signal fires\n\n"
            "Use /status to see open positions."
        ),
        "fa": (
            "✅ معامله *{symbol}* در قیمت `${price:.2f}` ثبت شد\n\n"
            "در موارد زیر آلرت خصوصی دریافت می‌کنید:\n"
            "  • 🎯 رسیدن به قیمت هدف\n"
            "  • 🛑 فعال شدن حد ضرر\n"
            "  • 🔔 صدور سیگنال فروش\n\n"
            "برای مشاهده معاملات باز از /status استفاده کنید."
        ),
    },

    # Status
    "status_no_positions": {
        "en": "You have no open positions.",
        "fa": "شما هیچ معامله باز فعالی ندارید.",
    },
    "status_header": {
        "en": "*Your open positions:*",
        "fa": "*معاملات باز شما:*",
    },

    # PnL
    "pnl_empty": {
        "en": "No closed trades yet.\n\nTrack a BUY signal from the channel to get started.",
        "fa": "هنوز هیچ معامله بسته‌ای ندارید.\n\nیک سیگنال خرید از کانال را دنبال کنید تا شروع کنید.",
    },
    "pnl_header": {
        "en": "📊 *Your Trade History*\n{n} trades  ·  {wins} wins  ·  Avg `{sign}{avg:.2f}%`",
        "fa": "📊 *تاریخچه معاملات شما*\n{n} معامله  ·  {wins} برنده  ·  میانگین `{sign}{avg:.2f}٪`",
    },

    # TP/SL auto-close
    "tp_hit": {
        "en": (
            "🎯 *{symbol} — Target Reached!*\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "Entry:  `${entry:.2f}`\n"
            "Exit:   `${exit:.2f}`\n"
            "Profit: `+{pnl:.2f}%` 🟢\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "_Position closed automatically._"
        ),
        "fa": (
            "🎯 *{symbol} — به هدف رسید!*\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "قیمت ورود:  `${entry:.2f}`\n"
            "قیمت خروج:  `${exit:.2f}`\n"
            "سود:        `+{pnl:.2f}٪` 🟢\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "_معامله به صورت خودکار بسته شد._"
        ),
    },
    "sl_hit": {
        "en": (
            "🛑 *{symbol} — Stop Loss Triggered*\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "Entry:  `${entry:.2f}`\n"
            "Exit:   `${exit:.2f}`\n"
            "Loss:   `{pnl:.2f}%` 🔴\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "_Position closed to protect your capital._"
        ),
        "fa": (
            "🛑 *{symbol} — حد ضرر فعال شد*\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "قیمت ورود:  `${entry:.2f}`\n"
            "قیمت خروج:  `${exit:.2f}`\n"
            "ضرر:        `{pnl:.2f}٪` 🔴\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "_معامله برای حفاظت از سرمایه بسته شد._"
        ),
    },

    # Private SELL alert
    "sell_alert_header": {
        "en": "🔔 *SELL Signal — {symbol}* (your position)",
        "fa": "🔔 *سیگنال فروش — {symbol}* (معامله شما)",
    },
    "sell_alert_body": {
        "en": (
            "Your entry:    `${entry:.2f}`\n"
            "Current price: `${current:.2f}`\n"
            "{pnl_line}\n"
            "Signal target: `${tp:.2f}`\n"
            "Signal stop:   `${sl:.2f}`\n\n"
            "_Consider closing your position now._"
        ),
        "fa": (
            "قیمت ورود شما:   `${entry:.2f}`\n"
            "قیمت فعلی:       `${current:.2f}`\n"
            "{pnl_line}\n"
            "هدف سیگنال:     `${tp:.2f}`\n"
            "حد ضرر سیگنال:  `${sl:.2f}`\n\n"
            "_در نظر بگیرید که معامله خود را ببندید._"
        ),
    },
    "unrealized_pnl": {
        "en": "{emoji} Unrealized P&L: `{sign}{pct:.2f}%`",
        "fa": "{emoji} سود/زیان باز: `{sign}{pct:.2f}٪`",
    },
    "btn_close_position": {
        "en": "✅ Close my position",
        "fa": "✅ بستن معامله",
    },

    # Cancel command
    "cancel_usage": {
        "en": "Usage: /cancel SYMBOL  (e.g. /cancel AAPL)",
        "fa": "نحوه استفاده: /cancel SYMBOL  (مثال: /cancel AAPL)",
    },
    "cancel_closed": {
        "en": "{emoji} *{symbol}* closed manually\nP&L: `{sign}{pnl:.2f}%`",
        "fa": "{emoji} *{symbol}* به صورت دستی بسته شد\nسود/زیان: `{sign}{pnl:.2f}٪`",
    },
    "cancel_not_found": {
        "en": "No open position for {symbol}.",
        "fa": "هیچ معامله بازی برای {symbol} پیدا نشد.",
    },

    # Subscribe
    "subscribe_current": {
        "en": "📬 *Your subscriptions:* {subs}\n\nTo add: /subscribe AAPL BTC-USD\nTo remove all: /unsubscribe\n\nAvailable: {all}",
        "fa": "📬 *اشتراک‌های شما:* {subs}\n\nبرای افزودن: /subscribe AAPL BTC-USD\nبرای حذف همه: /unsubscribe\n\nموجود: {all}",
    },
    "subscribe_prompt": {
        "en": "Subscribe to get DM alerts when signals fire:\n/subscribe AAPL TSLA BTC-USD\n\nAvailable: {all}",
        "fa": "برای دریافت آلرت خصوصی هنگام صدور سیگنال اشتراک بگیرید:\n/subscribe AAPL TSLA BTC-USD\n\nموجود: {all}",
    },
    "subscribe_added": {
        "en": "✅ Subscribed to: {added}\n\nYou'll get DM alerts when signals fire for these symbols.",
        "fa": "✅ اشتراک ثبت شد: {added}\n\nهنگام صدور سیگنال برای این نمادها آلرت خصوصی دریافت می‌کنید.",
    },
    "unsubscribed": {
        "en": "Unsubscribed from all symbols.",
        "fa": "اشتراک همه نمادها لغو شد.",
    },
    "unsubscribed_sym": {
        "en": "Unsubscribed from `{symbol}`.",
        "fa": "اشتراک `{symbol}` لغو شد.",
    },

    # Close callback
    "close_confirmed": {
        "en": "{emoji} *{symbol}* closed\nEntry `${entry:.2f}` → Exit `${exit:.2f}`\nP&L: `{sign}{pnl:.2f}%`",
        "fa": "{emoji} *{symbol}* بسته شد\nورود `${entry:.2f}` → خروج `${exit:.2f}`\nسود/زیان: `{sign}{pnl:.2f}٪`",
    },
    "close_not_found": {
        "en": "No open position found for {symbol}.",
        "fa": "معامله باز برای {symbol} پیدا نشد.",
    },

    # Stats command
    "stats_header": {
        "en": "📊 *Signal Performance*",
        "fa": "📊 *عملکرد سیگنال‌ها*",
    },
    "stats_body": {
        "en": "Subscribers: `{users}`\nTotal signals: `{total}`\nToday: `{today}`\nAccuracy: `{acc}`\n\nPer symbol:\n{sym_lines}",
        "fa": "مشترکان: `{users}`\nکل سیگنال‌ها: `{total}`\nامروز: `{today}`\nدقت: `{acc}`\n\nبه تفکیک نماد:\n{sym_lines}",
    },
    "stats_accuracy": {
        "en": "{pct}%  ({correct}/{resolved} resolved)",
        "fa": "{pct}٪  ({correct}/{resolved} بررسی‌شده)",
    },
    "stats_no_data": {
        "en": "N/A — no resolved signals yet",
        "fa": "موجود نیست — هنوز سیگنالی بررسی نشده",
    },

    # test command DM preview
    "test_channel_ok": {
        "en": "✅ Test signal sent to `{channel}`\n\n📱 *Preview of your DM format (below):*",
        "fa": "✅ سیگنال تست به `{channel}` ارسال شد\n\n📱 *پیش‌نمایش فرمت پیام شما (پایین):*",
    },
    "test_channel_fail": {
        "en": "❌ Failed to post to channel `{channel}`\n\nError: `{error}`\n\nMake sure:\n1. Channel exists\n2. Bot is Admin\n3. Bot has 'Post Messages' permission",
        "fa": "❌ ارسال به کانال `{channel}` ناموفق بود\n\nخطا: `{error}`\n\nبررسی کنید:\n۱. کانال وجود داشته باشد\n۲. ربات ادمین کانال باشد\n۳. ربات اجازه ارسال پیام داشته باشد",
    },

    # Misc
    "scanning": {
        "en": "🔍 Scanning…",
        "fa": "🔍 در حال اسکن…",
    },
    "backtest_running": {
        "en": "⏳ Running {years}-year backtest for *{symbol}*…",
        "fa": "⏳ در حال اجرای بک‌تست {years} ساله برای *{symbol}*…",
    },
    "backtest_usage": {
        "en": "Usage: /backtest SYMBOL [years]\n\nExample: /backtest AAPL 2\n\nAvailable: {all}",
        "fa": "نحوه استفاده: /backtest SYMBOL [سال‌ها]\n\nمثال: /backtest AAPL 2\n\nموجود: {all}",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    """Return a translated string, optionally format with kwargs."""
    template = _S.get(key, {}).get(lang) or _S.get(key, {}).get("en", key)
    return template.format(**kwargs) if kwargs else template


# ── Signal messages ────────────────────────────────────────────────────────────

def _trend_label(trend: str, action: str, lang: str) -> str:
    labels = {
        "en": {"up": "📈 Uptrend", "down": "📉 Downtrend", "sideways": "➡️ Sideways", "unknown": "— Unknown"},
        "fa": {"up": "📈 صعودی", "down": "📉 نزولی", "sideways": "➡️ خنثی", "unknown": "— نامشخص"},
    }
    label = labels.get(lang, labels["en"]).get(trend, "—")
    if (action == "BUY" and trend == "down") or (action == "SELL" and trend == "up"):
        warning = "  ⚠️ Counter-trend" if lang == "en" else "  ⚠️ خلاف روند"
        label += warning
    return label


def _quality_label(quality: int, lang: str) -> str:
    if lang == "fa":
        labels = {1: "ضعیف", 2: "متوسط", 3: "خوب", 4: "خیلی خوب", 5: "عالی"}
    else:
        labels = {1: "Weak", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"}
    return labels.get(quality, "—")


def _rr_label(rr: float, lang: str) -> str:
    """Plain-language risk/reward description for beginners."""
    if rr >= 2.0:
        return ("Great — gain 2× what you risk" if lang == "en"
                else "عالی — ۲ برابر ریسک سود می‌کنید")
    if rr >= 1.5:
        return ("Good — gain 1.5× what you risk" if lang == "en"
                else "خوب — ۱.۵ برابر ریسک سود می‌کنید")
    return ("Fair — gain ~equal to what you risk" if lang == "en"
            else "متوسط — سود برابر با ریسک شما")


def signal_msg(sig: dict, lang: str = "en", mode: str = "beginner") -> str:
    """
    Format a signal message for private DMs.
    Channel always uses expert English; this is for subscriptions/tracked trades.
    """
    action = sig["action"]
    symbol = sig["symbol"]
    price  = sig["price"]
    tp     = sig.get("tp", 0)
    sl     = sig.get("sl", 0)
    tp_pct = sig.get("tp_pct", 0)
    sl_pct = sig.get("sl_pct", 0)
    rr     = sig.get("rr", 0)
    rsi    = sig.get("rsi", 0)
    trend  = sig.get("trend", "unknown")
    quality = sig.get("quality", 0)
    stars   = sig.get("stars", "")
    ai_conf = sig.get("ai_confidence")
    strength = sig.get("strength", "RULE")
    mtf     = sig.get("mtf_confirmed", False)

    is_buy = action == "BUY"
    action_emoji = ("🟢" if is_buy else "🔴") if strength != "STRONG" else ("🔥" if is_buy else "💀")

    if mode == "beginner":
        return _signal_beginner(sig, lang, action_emoji)
    else:
        return _signal_expert(sig, lang, action_emoji)


def _signal_beginner(sig: dict, lang: str, action_emoji: str) -> str:
    action   = sig["action"]
    symbol   = sig["symbol"]
    price    = sig["price"]
    tp       = sig.get("tp", 0)
    sl       = sig.get("sl", 0)
    tp_pct   = sig.get("tp_pct", 0)
    sl_pct   = sig.get("sl_pct", 0)
    rr       = sig.get("rr", 0)
    quality  = sig.get("quality", 0)
    stars    = sig.get("stars", "")
    strength = sig.get("strength", "RULE")
    mtf      = sig.get("mtf_confirmed", False)

    ql = _quality_label(quality, lang)
    rrl = _rr_label(rr, lang)

    tp_abs = abs(tp_pct)
    sl_abs = abs(sl_pct)

    if lang == "fa":
        action_word = "خرید" if action == "BUY" else "فروش"
        entry_label = "قیمت خرید" if action == "BUY" else "قیمت فروش"
        strength_tip = {
            "STRONG": "هوش مصنوعی و تحلیل تکنیکال هر دو این سیگنال را تأیید کردند.",
            "AI": "هوش مصنوعی این الگو را شناسایی کرد.",
            "RULE": "تحلیل تکنیکال این سیگنال را تأیید کرد.",
        }.get(strength, "")
        mtf_line = "\n✅ تأیید چند تایم‌فریمی (۱ ساعته + ۵ دقیقه‌ای)" if mtf else ""
        return (
            f"{action_emoji} *سیگنال {action_word} — {symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"💰 {entry_label}:    `${price:.2f}`\n"
            f"🎯 هدف قیمتی:    `${tp:.2f}`  *(سود +{tp_abs:.1f}٪)*\n"
            f"🛑 حد ضرر:       `${sl:.2f}`  *(ضرر {sl_abs:.1f}٪)*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"⚖️ نسبت سود به ریسک: {rrl}\n"
            f"⭐ قدرت سیگنال: {stars} *{ql}*\n"
            f"{mtf_line}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"💡 _{strength_tip}_"
        )
    else:
        action_word = "BUY" if action == "BUY" else "SELL"
        entry_label = "Buy at:" if action == "BUY" else "Sell at:"
        strength_tip = {
            "STRONG": "Both AI and technical rules agree on this signal.",
            "AI": "AI model detected this pattern with high confidence.",
            "RULE": "Technical indicators confirmed this signal.",
        }.get(strength, "")
        mtf_line = "\n✅ Confirmed on both 1h and 5m chart" if mtf else ""
        return (
            f"{action_emoji} *{action_word} Signal — {symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"💰 {entry_label}    `${price:.2f}`\n"
            f"🎯 Target:    `${tp:.2f}`  *(profit +{tp_abs:.1f}%)*\n"
            f"🛑 Stop loss: `${sl:.2f}`  *(max loss {sl_abs:.1f}%)*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"⚖️ Risk/Reward: {rrl}\n"
            f"⭐ Signal strength: {stars} *{ql}*\n"
            f"{mtf_line}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"💡 _{strength_tip}_"
        )


def _signal_expert(sig: dict, lang: str, action_emoji: str) -> str:
    action   = sig["action"]
    symbol   = sig["symbol"]
    price    = sig["price"]
    tp       = sig.get("tp", 0)
    sl       = sig.get("sl", 0)
    tp_pct   = sig.get("tp_pct", 0)
    sl_pct   = sig.get("sl_pct", 0)
    rr       = sig.get("rr", 0)
    rsi      = sig.get("rsi", 0)
    trend    = sig.get("trend", "unknown")
    quality  = sig.get("quality", 0)
    stars    = sig.get("stars", "")
    ai_conf  = sig.get("ai_confidence")
    strength = sig.get("strength", "RULE")
    vol_spike = sig.get("vol_spike", False)
    mtf      = sig.get("mtf_confirmed", False)

    reasons  = sig.get("reasons", [])
    ind_parts = []
    if any("RSI" in r for r in reasons):
        ind_parts.append(f"RSI {rsi:.0f}")
    if any("MACD" in r for r in reasons):
        ind_parts.append("MACD " + ("↑" if action == "BUY" else "↓"))
    if any("EMA" in r for r in reasons):
        ind_parts.append("EMA " + ("↑" if action == "BUY" else "↓"))
    if vol_spike:
        ind_parts.append("Vol 🔥")
    ind_line = "  ·  ".join(ind_parts) if ind_parts else "AI pattern"

    tp_sign  = "+" if tp_pct >= 0 else ""
    sl_sign  = "+" if sl_pct >= 0 else ""
    trend_lbl = _trend_label(trend, action, lang)

    if lang == "fa":
        action_word = "خرید" if action == "BUY" else "فروش"
        str_labels = {"STRONG": "قوانین + هوش مصنوعی ✦", "AI": "فقط هوش مصنوعی", "RULE": "تکنیکال"}
        ai_line = f"\nهوش مصنوعی: `{ai_conf*100:.0f}٪` اطمینان" if ai_conf else ""
        mtf_line = "\n✅ تأیید چند تایم‌فریمی (۱ساعته+۵دقیقه)" if mtf else ""
        return (
            f"{action_emoji} *{action_word} — ${symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"ورود:         `${price:.2f}`\n"
            f"🎯 هدف:      `${tp:.2f}`  (`{tp_sign}{tp_pct:.1f}٪`)\n"
            f"🛑 حد ضرر:  `${sl:.2f}`  (`{sl_sign}{sl_pct:.1f}٪`)\n"
            f"⚖️ R/R:      `1 : {rr}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"اندیکاتورها: {ind_line}\n"
            f"روند: {trend_lbl}\n"
            f"نوع: _{str_labels.get(strength, strength)}_"
            f"{ai_line}"
            f"{mtf_line}\n"
            f"کیفیت: {stars}  `({quality}/5)`\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )
    else:
        str_labels = {"STRONG": "Rules + AI ✦", "AI": "AI only", "RULE": "Technical"}
        ai_line = f"\nAI: `{ai_conf*100:.0f}%` confident" if ai_conf else ""
        mtf_line = "\n✅ Multi-timeframe confirmed (1h+5m)" if mtf else ""
        return (
            f"{action_emoji} *{action} — ${symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Entry:       `${price:.2f}`\n"
            f"🎯 Target:  `${tp:.2f}`  (`{tp_sign}{tp_pct:.1f}%`)\n"
            f"🛑 Stop:    `${sl:.2f}`  (`{sl_sign}{sl_pct:.1f}%`)\n"
            f"⚖️  R/R:     `1 : {rr}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Indicators: {ind_line}\n"
            f"Trend: {trend_lbl}\n"
            f"Type: _{str_labels.get(strength, strength)}_"
            f"{ai_line}"
            f"{mtf_line}\n"
            f"Quality: {stars}  `({quality}/5)`\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )


def tp_sl_alert(symbol: str, entry: float, exit_price: float, pnl: float,
                hit_tp: bool, lang: str = "en") -> str:
    key = "tp_hit" if hit_tp else "sl_hit"
    return t(key, lang=lang, symbol=symbol, entry=entry, exit=exit_price, pnl=abs(pnl))


def sell_alert(symbol: str, entry: float, current: float, tp: float, sl: float,
               lang: str = "en") -> str:
    pct = (current - entry) / entry * 100
    sign = "+" if pct >= 0 else ""
    emoji = "🟢" if pct >= 0 else "🔴"
    pnl_line = t("unrealized_pnl", lang=lang, emoji=emoji, sign=sign, pct=pct)
    header = t("sell_alert_header", lang=lang, symbol=symbol)
    body   = t("sell_alert_body",   lang=lang,
               entry=entry, current=current, pnl_line=pnl_line, tp=tp, sl=sl)
    return f"🔔 {header}\n━━━━━━━━━━━━━━━━━━━\n{body}"
