#!/usr/bin/env python3
"""
monitor.py — VIX/vStoxx spread monitor for week-1 entry conditions.

Strategy (always-on, no regime filter):
  - Check each trading day; signal is relevant only in week 1 (days 1–7).
  - Skip entry if vStoxx − VIX > 10  (EU crisis / dislocation filter).
  - Position: Short 1× VIX futures, Long ~8× vStoxx futures (dollar-neutral).

Data sources:
  VIX    → Yahoo Finance    (^VIX)   — via yfinance
  vStoxx → Investing.com   (id:1498) — via curl_cffi (bundled with yfinance)
"""

import os
import sys
import time
import logging
from datetime import date, datetime

import yfinance as yf
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Strategy parameters ────────────────────────────────────────────────────────
EU_CRISIS_THRESHOLD         = 10.0   # Skip entry if vStoxx − VIX exceeds this
VSTOXX_PER_VIX              = 8      # ~dollar-neutral ratio: short 1 VIX, long 8 vStoxx
WEEK_ONE_MAX_DAY            = 7      # Days 1–7 of the month define "week 1"
CORRELATION_ALERT_THRESHOLD = 30.0   # % rise in BOTH VIX and vStoxx over 5 trading days

# ── Network settings ──────────────────────────────────────────────────────────
RETRIES     = 3
RETRY_DELAY = 5   # seconds (multiplied by attempt number)


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_vix_series(period: str = "5d") -> "pd.Series":
    """Download VIX close-price series from Yahoo Finance (^VIX)."""
    ticker = "^VIX"
    for attempt in range(1, RETRIES + 1):
        try:
            logging.info("Fetching %s (attempt %d)…", ticker, attempt)
            df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            if df.empty:
                raise RuntimeError("No data returned for ^VIX")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if close.empty:
                raise RuntimeError("'Close' column empty for ^VIX")
            return close
        except Exception as exc:
            logging.warning("Attempt %d failed for ^VIX: %s", attempt, exc)
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def fetch_vix() -> float:
    """Fetch VIX closing price from Yahoo Finance (^VIX)."""
    close = _fetch_vix_series(period="5d")
    val = float(close.iloc[-1])
    logging.info("  VIX = %.4f  (date: %s)", val, close.index[-1].date())
    return val


def fetch_vix_5d_pct() -> tuple[float, float]:
    """
    Fetch VIX history and return (pct_5d_change, today_value).
    pct_5d_change = (today − 5_trading_days_ago) / 5_trading_days_ago * 100
    """
    close = _fetch_vix_series(period="1mo")
    if len(close) < 6:
        raise RuntimeError(f"Insufficient ^VIX history ({len(close)} rows, need ≥ 6)")
    today_val     = float(close.iloc[-1])
    five_days_ago = float(close.iloc[-6])
    pct = (today_val - five_days_ago) / five_days_ago * 100
    logging.info("  VIX 5-day: %.4f → %.4f  (%+.2f%%)", five_days_ago, today_val, pct)
    return pct, today_val


def _fetch_vstoxx_rows() -> list:
    """
    Fetch raw vStoxx data rows from Investing.com.

    Yahoo Finance does not carry the EURO STOXX 50 Volatility Index.
    We use curl_cffi (already installed as a yfinance dependency) to fetch
    from Investing.com's internal chart API.

    Endpoint : api.investing.com/api/financialdata/1498/historical/chart/
    Instrument ID 1498 = VSTOXX (EURO STOXX 50 Volatility Index)
    Data format: JSON array of [timestamp_ms, open, high, low, close, ...]
    """
    from curl_cffi import requests as cffi_req

    url = (
        "https://api.investing.com/api/financialdata/1498/historical/chart/"
        "?period=P1M&interval=P1D&pointscount=60"
    )

    for attempt in range(1, RETRIES + 1):
        try:
            logging.info("Fetching vStoxx from investing.com (attempt %d)…", attempt)
            r = cffi_req.get(url, impersonate="chrome110", timeout=20)
            r.raise_for_status()
            rows = r.json().get("data", [])
            if not rows:
                raise RuntimeError("Empty data array from investing.com")
            return rows
        except Exception as exc:
            logging.warning("Attempt %d failed for vStoxx: %s", attempt, exc)
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def fetch_vstoxx() -> float:
    """
    Fetch the latest VSTOXX closing price from Investing.com.

    Rows are sorted oldest → newest; last row = most recent session.
    Each row: [timestamp_ms, open, high, low, close, ...]
    """
    from datetime import timezone

    rows   = _fetch_vstoxx_rows()
    latest = rows[-1]
    ts_ms  = latest[0]
    close  = float(latest[4])
    date_str = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    logging.info("  vStoxx = %.4f  (date: %s)", close, date_str)
    return close


def fetch_vstoxx_5d_pct() -> tuple[float, float]:
    """
    Fetch vStoxx history and return (pct_5d_change, today_value).
    pct_5d_change = (today − 5_trading_days_ago) / 5_trading_days_ago * 100
    """
    rows = _fetch_vstoxx_rows()
    if len(rows) < 6:
        raise RuntimeError(f"Insufficient vStoxx history ({len(rows)} rows, need ≥ 6)")
    today_val     = float(rows[-1][4])
    five_days_ago = float(rows[-6][4])
    pct = (today_val - five_days_ago) / five_days_ago * 100
    logging.info("  vStoxx 5-day: %.4f → %.4f  (%+.2f%%)", five_days_ago, today_val, pct)
    return pct, today_val


# ─────────────────────────────────────────────────────────────────────────────
# Entry-condition logic
# ─────────────────────────────────────────────────────────────────────────────

def is_week_one(today: date | None = None) -> bool:
    """Return True when today falls in week 1 of the month (day 1–7)."""
    if today is None:
        today = date.today()
    return today.day <= WEEK_ONE_MAX_DAY


def evaluate(
    vix: float,
    vstoxx: float,
    pct_change_vix: float = 0.0,
    pct_change_vstoxx: float = 0.0,
    today: date | None = None,
) -> dict:
    """
    Evaluate all entry conditions and return a result dict:
      spread             : vStoxx − VIX
      eu_crisis          : True if spread > EU_CRISIS_THRESHOLD
      week_one           : True if today is in week 1 of the month
      enter              : True iff week_one AND NOT eu_crisis
      pct_change_vix     : VIX 5-trading-day % change
      pct_change_vstoxx  : vStoxx 5-trading-day % change
      correlation_alert  : True if BOTH pct changes > CORRELATION_ALERT_THRESHOLD
    """
    spread    = vstoxx - vix
    eu_crisis = spread > EU_CRISIS_THRESHOLD
    week_one  = is_week_one(today)
    enter     = week_one and not eu_crisis
    correlation_alert = (
        pct_change_vix    > CORRELATION_ALERT_THRESHOLD
        and pct_change_vstoxx > CORRELATION_ALERT_THRESHOLD
    )

    return {
        "spread":            spread,
        "eu_crisis":         eu_crisis,
        "week_one":          week_one,
        "enter":             enter,
        "pct_change_vix":    pct_change_vix,
        "pct_change_vstoxx": pct_change_vstoxx,
        "correlation_alert": correlation_alert,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Message formatting
# ─────────────────────────────────────────────────────────────────────────────

def build_message(vix: float, vstoxx: float, result: dict) -> str:
    today_str         = date.today().strftime("%A, %d %b %Y")
    spread            = result["spread"]
    pct_change_vix    = result["pct_change_vix"]
    pct_change_vstoxx = result["pct_change_vstoxx"]
    correlation_alert = result["correlation_alert"]

    lines = [
        f"📊 <b>VIX / vStoxx Monitor</b>",
        f"📅 {today_str}",
        "",
        f"  VIX    (^VIX):  <b>{vix:.2f}</b>",
        f"  vStoxx (^V2TX): <b>{vstoxx:.2f}</b>",
        f"  Spread (vStoxx − VIX): <b>{spread:+.2f}</b>",
        "",
    ]

    # Week-1 status
    if result["week_one"]:
        lines.append("📅 Week 1 of month: ✅ YES — entry window open")
    else:
        lines.append("📅 Week 1 of month: ⏳ NO  — wait for next week 1")

    # EU-crisis filter
    if result["eu_crisis"]:
        lines.append(
            f"⚠️  EU Crisis filter: 🔴 TRIGGERED  "
            f"(spread {spread:+.2f} > {EU_CRISIS_THRESHOLD:.0f})"
        )
    else:
        lines.append(
            f"🛡️  EU Crisis filter: ✅ CLEAR  "
            f"(spread {spread:+.2f} ≤ {EU_CRISIS_THRESHOLD:.0f})"
        )

    # Correlation risk
    if correlation_alert:
        lines.append(
            f"⚡ Corr. Risk:    🔴 ALERT — "
            f"VIX +{pct_change_vix:.1f}% & vStoxx +{pct_change_vstoxx:.1f}% in 5 days"
        )
    else:
        lines.append(
            f"⚡ Corr. Risk:    ✅ CLEAR  "
            f"(VIX {pct_change_vix:+.1f}% / vStoxx {pct_change_vstoxx:+.1f}% over 5 days)"
        )

    lines.append("")

    # ── Final verdict ──────────────────────────────────────────────────────
    # Correlation spike overrides all other signals
    if correlation_alert:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🔴 <b>FLATTEN POSITION</b> — Correlation spike detected",
            "   Both VIX and vStoxx rose >30% in 5 days",
            "   Global correlated melt-up — spread thesis breaks down",
            "   Exit spread immediately, re-assess when volatility stabilises",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    elif result["enter"]:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🟢 <b>ENTER TRADE</b>",
            f"   • Short  <b>1×</b>  VIX futures   (^VIX)",
            f"   • Long   <b>{VSTOXX_PER_VIX}×</b>  vStoxx futures (^V2TX)",
            "   • Dollar-neutral position",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    elif not result["week_one"]:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "⏳ <b>NO SIGNAL</b> — not in entry week",
            "   Wait for week 1 of next month",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    else:  # week_one but eu_crisis
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🔴 <b>SKIP ENTRY</b> — EU crisis filter active",
            f"   Spread ({spread:+.2f}) exceeds threshold ({EU_CRISIS_THRESHOLD:.0f})",
            "   Monitor daily; re-assess when spread normalises",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Telegram delivery
# ─────────────────────────────────────────────────────────────────────────────

def send_telegram(bot_token: str, chat_id: str, text: str) -> dict:
    url     = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(url, data=payload, timeout=15)
            r.raise_for_status()
            logging.info("Telegram: sent OK (HTTP %s)", r.status_code)
            return r.json()
        except Exception as exc:
            logging.warning("Attempt %d: Telegram send failed: %s", attempt, exc)
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id   = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logging.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
        sys.exit(2)

    try:
        # Fetch current spot values
        vix    = fetch_vix()
        vstoxx = fetch_vstoxx()

        # Fetch 5-trading-day % changes for correlation risk
        pct_change_vix,    _ = fetch_vix_5d_pct()
        pct_change_vstoxx, _ = fetch_vstoxx_5d_pct()

        result  = evaluate(vix, vstoxx, pct_change_vix, pct_change_vstoxx)
        message = build_message(vix, vstoxx, result)

        logging.info("\n%s", message)
        send_telegram(bot_token, chat_id, message)
        logging.info("Done.")

    except Exception as exc:
        logging.exception("Unhandled error in monitor")
        try:
            send_telegram(bot_token, chat_id, f"⚠️ VIX/vStoxx monitor error: {exc}")
        except Exception:
            logging.exception("Also failed to send error notification to Telegram")
        sys.exit(1)


if __name__ == "__main__":
    main()
