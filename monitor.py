#!/usr/bin/env python3
"""
monitor.py — VIX/vStoxx spread monitor for week-1 entry conditions.

Strategy (always-on, no regime filter):
  - Check each trading day; signal is relevant only in week 1 (days 1–7).
  - Skip entry if vStoxx − VIX > 10  (EU crisis / dislocation filter).
  - Position: Short 1× VIX futures, Long ~8× vStoxx futures (dollar-neutral).

Data sources:
  VIX    → Yahoo Finance  (^VIX)
  vStoxx → Stooq.com      (^VSTOXX) — Yahoo Finance does not carry this index
"""

import io
import os
import sys
import time
import logging
from datetime import date, timedelta

import yfinance as yf
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Strategy parameters ────────────────────────────────────────────────────────
EU_CRISIS_THRESHOLD = 10.0   # Skip entry if vStoxx − VIX exceeds this
VSTOXX_PER_VIX     = 8       # ~dollar-neutral ratio: short 1 VIX, long 8 vStoxx
WEEK_ONE_MAX_DAY   = 7       # Days 1–7 of the month define "week 1"

# ── Network settings ──────────────────────────────────────────────────────────
RETRIES     = 3
RETRY_DELAY = 5   # seconds (multiplied by attempt number)


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_vix() -> float:
    """Fetch VIX closing price from Yahoo Finance (^VIX)."""
    ticker = "^VIX"
    for attempt in range(1, RETRIES + 1):
        try:
            logging.info("Fetching %s (attempt %d)…", ticker, attempt)
            df = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
            if df.empty:
                raise RuntimeError("No data returned for ^VIX")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if close.empty:
                raise RuntimeError("'Close' column empty for ^VIX")
            val = float(close.iloc[-1])
            logging.info("  VIX = %.4f  (date: %s)", val, close.index[-1].date())
            return val
        except Exception as exc:
            logging.warning("Attempt %d failed for ^VIX: %s", attempt, exc)
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def fetch_vstoxx() -> float:
    """
    Fetch VSTOXX closing price from stooq.com.

    Yahoo Finance does not carry the EURO STOXX 50 Volatility Index (^V2TX).
    Stooq.com provides it under the symbol ^vstoxx and returns a plain CSV
    — no API key required.
    """
    today = date.today()
    start = (today - timedelta(days=10)).strftime("%Y%m%d")
    end   = today.strftime("%Y%m%d")
    url   = f"https://stooq.com/q/d/l/?s=^vstoxx&d1={start}&d2={end}&i=d"

    for attempt in range(1, RETRIES + 1):
        try:
            logging.info("Fetching vStoxx from stooq (attempt %d)…", attempt)
            r = requests.get(url, timeout=15)
            r.raise_for_status()

            df = pd.read_csv(io.StringIO(r.text))
            if df.empty or "Close" not in df.columns:
                raise RuntimeError("Unexpected stooq response format")

            df = df[df["Close"] > 0].dropna(subset=["Close"])
            if df.empty:
                raise RuntimeError("No valid vStoxx prices returned from stooq")

            val      = float(df["Close"].iloc[-1])
            date_str = df["Date"].iloc[-1]
            logging.info("  vStoxx = %.4f  (date: %s)", val, date_str)
            return val

        except Exception as exc:
            logging.warning("Attempt %d failed for vStoxx (stooq): %s", attempt, exc)
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


# ─────────────────────────────────────────────────────────────────────────────
# Entry-condition logic
# ─────────────────────────────────────────────────────────────────────────────

def is_week_one(today: date | None = None) -> bool:
    """Return True when today falls in week 1 of the month (day 1–7)."""
    if today is None:
        today = date.today()
    return today.day <= WEEK_ONE_MAX_DAY


def evaluate(vix: float, vstoxx: float, today: date | None = None) -> dict:
    """
    Evaluate all entry conditions and return a result dict:
      spread     : vStoxx − VIX
      eu_crisis  : True if spread > EU_CRISIS_THRESHOLD
      week_one   : True if today is in week 1 of the month
      enter      : True iff week_one AND NOT eu_crisis
    """
    spread    = vstoxx - vix
    eu_crisis = spread > EU_CRISIS_THRESHOLD
    week_one  = is_week_one(today)
    enter     = week_one and not eu_crisis

    return {
        "spread":    spread,
        "eu_crisis": eu_crisis,
        "week_one":  week_one,
        "enter":     enter,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Message formatting
# ─────────────────────────────────────────────────────────────────────────────

def build_message(vix: float, vstoxx: float, result: dict) -> str:
    today_str = date.today().strftime("%A, %d %b %Y")
    spread    = result["spread"]

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

    lines.append("")

    # ── Final verdict ──────────────────────────────────────────────────────
    if result["enter"]:
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
        vix     = fetch_vix()
        vstoxx  = fetch_vstoxx()
        result  = evaluate(vix, vstoxx)
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
