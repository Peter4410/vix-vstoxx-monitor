#!/usr/bin/env python3
"""
monitor.py — VIX/vStoxx spread monitor for week-1 entry conditions.

Strategy (always-on, no regime filter):
  - Check each trading day; signal is relevant only in week 1 (days 1–7).
  - Skip entry if vStoxx − VIX spread triggers EU crisis filter (hysteresis band).
  - Position: Short 1× VIX futures, Long ~8× vStoxx futures (dollar-neutral).

Changes vs v1:
  [EU Crisis]  Threshold tightened from 10pt to 7pt with a 2-pt hysteresis band.
               Enter crisis mode when spread > 7.0; exit only when spread < 5.0.
               State (eu_crisis_active) is persisted in state.json.
  [HAR-RV]     Heterogeneous Autoregressive Realized Volatility model added as a
               confirming signal. Uses three lagged vol averages (1-day, 5-day,
               22-day) to forecast whether VIX / vStoxx is mean-reverting or
               accelerating. Green-light if mean-reverting; caution if accelerating.

Data sources:
  VIX    → Yahoo Finance    (^VIX)   — via yfinance
  vStoxx → Investing.com   (id:1498) — via curl_cffi (bundled with yfinance)
"""

import json
import os
import sys
import time
import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import yfinance as yf
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Strategy parameters ────────────────────────────────────────────────────────
EU_CRISIS_ENTER             = 7.0    # Enter crisis mode when vStoxx−VIX spread > this
EU_CRISIS_EXIT              = 5.0    # Exit crisis mode when spread drops below this
VSTOXX_PER_VIX              = 8      # ~dollar-neutral ratio: short 1 VIX, long 8 vStoxx
WEEK_ONE_MAX_DAY            = 7      # Days 1–7 of the month define "week 1"
CORRELATION_ALERT_THRESHOLD = 30.0   # % rise in BOTH VIX and vStoxx over 5 trading days

# ── HAR-RV parameters ─────────────────────────────────────────────────────────
HAR_MIN_ROWS = 100   # Minimum trading days of history needed to fit the model

# ── Network settings ──────────────────────────────────────────────────────────
RETRIES     = 3
RETRY_DELAY = 5   # seconds (multiplied by attempt number)

# ── State file ─────────────────────────────────────────────────────────────────
STATE_FILE = Path(__file__).parent / "state.json"

DEFAULT_STATE: dict = {
    "eu_crisis_active": False,   # hysteresis: True once spread > 7, until spread < 5
    "last_run":         None,
}


# ─────────────────────────────────────────────────────────────────────────────
# State management
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return {**DEFAULT_STATE, **json.load(f)}
    logging.info("No state.json found — using defaults (first run).")
    return DEFAULT_STATE.copy()


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    logging.info("State saved: %s", state)


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_vix_series(period: str = "5d") -> pd.Series:
    """Download VIX close-price series from Yahoo Finance (^VIX)."""
    ticker = "^VIX"
    for attempt in range(1, RETRIES + 1):
        try:
            logging.info("Fetching %s period=%s (attempt %d)…", ticker, period, attempt)
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
    """Return (pct_5d_change, today_value) for VIX."""
    close = _fetch_vix_series(period="1mo")
    if len(close) < 6:
        raise RuntimeError(f"Insufficient ^VIX history ({len(close)} rows, need ≥ 6)")
    today_val     = float(close.iloc[-1])
    five_days_ago = float(close.iloc[-6])
    pct = (today_val - five_days_ago) / five_days_ago * 100
    logging.info("  VIX 5-day: %.4f → %.4f  (%+.2f%%)", five_days_ago, today_val, pct)
    return pct, today_val


def fetch_vix_history() -> pd.Series:
    """Fetch 2-year VIX history for HAR-RV model fitting."""
    return _fetch_vix_series(period="2y")


def _fetch_vstoxx_rows(period: str = "P1M", pointscount: int = 60) -> list:
    """
    Fetch raw vStoxx data rows from Investing.com.
    Instrument ID 1498 = VSTOXX. Each row: [timestamp_ms, open, high, low, close, ...]
    """
    from curl_cffi import requests as cffi_req

    url = (
        "https://api.investing.com/api/financialdata/1498/historical/chart/"
        f"?period={period}&interval=P1D&pointscount={pointscount}"
    )

    for attempt in range(1, RETRIES + 1):
        try:
            logging.info("Fetching vStoxx (period=%s, pointscount=%d, attempt %d)…", period, pointscount, attempt)
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
    """Fetch the latest VSTOXX closing price from Investing.com."""
    from datetime import timezone
    rows   = _fetch_vstoxx_rows(period="P1M", pointscount=60)
    latest = rows[-1]
    close  = float(latest[4])
    date_str = datetime.fromtimestamp(latest[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    logging.info("  vStoxx = %.4f  (date: %s)", close, date_str)
    return close


def fetch_vstoxx_5d_pct() -> tuple[float, float]:
    """Return (pct_5d_change, today_value) for vStoxx."""
    rows = _fetch_vstoxx_rows(period="P1M", pointscount=60)
    if len(rows) < 6:
        raise RuntimeError(f"Insufficient vStoxx history ({len(rows)} rows, need ≥ 6)")
    today_val     = float(rows[-1][4])
    five_days_ago = float(rows[-6][4])
    pct = (today_val - five_days_ago) / five_days_ago * 100
    logging.info("  vStoxx 5-day: %.4f → %.4f  (%+.2f%%)", five_days_ago, today_val, pct)
    return pct, today_val


def fetch_vstoxx_history() -> pd.Series:
    """
    Fetch as much vStoxx history as investing.com will return for HAR-RV fitting.
    Tries progressively shorter periods until one succeeds.
    HAR model needs ≥ 100 rows; returns whatever is available (model skips if too short).
    """
    for period, points in [("P1Y", 365), ("P6M", 180), ("P3M", 90), ("P1M", 60)]:
        try:
            rows = _fetch_vstoxx_rows(period=period, pointscount=points)
            dates = [datetime.fromtimestamp(r[0] / 1000).date() for r in rows]
            vals  = [float(r[4]) for r in rows]
            series = pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
            logging.info("  vStoxx history: %d rows (period=%s)", len(series), period)
            return series
        except Exception as exc:
            logging.warning("vStoxx history period=%s failed: %s — trying shorter period…", period, exc)
    # Final fallback: return empty series (HAR will be skipped gracefully)
    logging.warning("vStoxx history: all periods failed — HAR-RV will be skipped.")
    return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# HAR-RV confirming signal
# ─────────────────────────────────────────────────────────────────────────────

def compute_har_forecast(series: pd.Series, label: str = "") -> dict | None:
    """
    Fit a HAR (Heterogeneous Autoregressive) model on a vol level series
    and return a one-step-ahead forecast.

    Model:  V_{t+1} = β0 + β1·V_d + β2·V_w + β3·V_m + ε
      V_d = level at time t           (1-day lag)
      V_w = mean of t−4 … t           (5-day average)
      V_m = mean of t−21 … t          (22-day average)

    Coefficients estimated by OLS over all available history.
    Returns None if insufficient history (< HAR_MIN_ROWS rows).
    """
    if len(series) < HAR_MIN_ROWS:
        logging.warning(
            "HAR-RV [%s]: only %d rows, need ≥ %d — skipping.",
            label, len(series), HAR_MIN_ROWS,
        )
        return None

    values = series.values.astype(float)
    n      = len(values)

    X_rows, y_vals = [], []
    for i in range(22, n - 1):          # 22-day lookback; predict value at i+1
        v_d = values[i]
        v_w = values[i - 4 : i + 1].mean()
        v_m = values[i - 21: i + 1].mean()
        X_rows.append([1.0, v_d, v_w, v_m])
        y_vals.append(values[i + 1])

    X = np.array(X_rows)
    y = np.array(y_vals)

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    current  = float(values[-1])
    v_w_now  = float(values[-5:].mean())
    v_m_now  = float(values[-22:].mean())
    forecast = float(np.dot(beta, [1.0, current, v_w_now, v_m_now]))

    direction = "mean_reverting" if forecast < current else "accelerating"
    logging.info(
        "  HAR-RV [%s]: current=%.2f  forecast=%.2f  → %s",
        label, current, forecast, direction,
    )

    return {
        "current":   current,
        "forecast":  forecast,
        "direction": direction,
        "beta":      beta.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EU crisis hysteresis
# ─────────────────────────────────────────────────────────────────────────────

def update_eu_crisis(spread: float, currently_active: bool) -> bool:
    """
    Apply hysteresis to the EU crisis flag.
      Enter crisis mode:  spread > EU_CRISIS_ENTER (7.0)
      Exit  crisis mode:  spread < EU_CRISIS_EXIT  (5.0)
      In the 5–7 band the flag stays unchanged (sticky).
    """
    if currently_active:
        return spread >= EU_CRISIS_EXIT     # stay active until spread < 5
    else:
        return spread > EU_CRISIS_ENTER     # activate once spread exceeds 7


# ─────────────────────────────────────────────────────────────────────────────
# Entry-condition logic
# ─────────────────────────────────────────────────────────────────────────────

def is_week_one(today: date | None = None) -> bool:
    """Return True when today falls in week 1 of the month (day 1–7)."""
    if today is None:
        today = date.today()
    return today.day <= WEEK_ONE_MAX_DAY


def evaluate(
    vix:               float,
    vstoxx:            float,
    eu_crisis_active:  bool,
    pct_change_vix:    float = 0.0,
    pct_change_vstoxx: float = 0.0,
    har_vix:           dict | None = None,
    har_vstoxx:        dict | None = None,
    today:             date | None = None,
) -> dict:
    """Evaluate all entry conditions and return a result dict."""
    spread    = vstoxx - vix
    week_one  = is_week_one(today)
    enter     = week_one and not eu_crisis_active

    correlation_alert = (
        pct_change_vix    > CORRELATION_ALERT_THRESHOLD
        and pct_change_vstoxx > CORRELATION_ALERT_THRESHOLD
    )

    # HAR confirming signal: yellow if EITHER forecast is accelerating
    har_green = not (
        (har_vix    and har_vix["direction"]    == "accelerating") or
        (har_vstoxx and har_vstoxx["direction"] == "accelerating")
    )

    return {
        "spread":            spread,
        "eu_crisis":         eu_crisis_active,
        "week_one":          week_one,
        "enter":             enter,
        "pct_change_vix":    pct_change_vix,
        "pct_change_vstoxx": pct_change_vstoxx,
        "correlation_alert": correlation_alert,
        "har_vix":           har_vix,
        "har_vstoxx":        har_vstoxx,
        "har_green":         har_green,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Message formatting
# ─────────────────────────────────────────────────────────────────────────────

def _har_line(label: str, har: dict | None) -> str:
    if har is None:
        return f"  {label}: ⚪ insufficient history"
    arrow     = "↓" if har["direction"] == "mean_reverting" else "↑"
    icon      = "✅" if har["direction"] == "mean_reverting" else "⚠️"
    direction = "Mean-reverting" if har["direction"] == "mean_reverting" else "Accelerating"
    return (
        f"  {label}: {icon} {direction} {arrow}  "
        f"({har['current']:.2f} → <b>{har['forecast']:.2f}</b>)"
    )


def build_message(vix: float, vstoxx: float, result: dict) -> str:
    today_str         = date.today().strftime("%A, %d %b %Y")
    spread            = result["spread"]
    pct_change_vix    = result["pct_change_vix"]
    pct_change_vstoxx = result["pct_change_vstoxx"]
    correlation_alert = result["correlation_alert"]
    eu_crisis         = result["eu_crisis"]
    har_green         = result["har_green"]

    lines = [
        "📊 <b>VIX / vStoxx Monitor</b>",
        f"📅 {today_str}",
        "",
        f"  VIX    (^VIX):  <b>{vix:.2f}</b>",
        f"  vStoxx (^V2TX): <b>{vstoxx:.2f}</b>",
        f"  Spread (vStoxx − VIX): <b>{spread:+.2f}</b>",
        "",
    ]

    # ── Week-1 window ──────────────────────────────────────────────────────
    if result["week_one"]:
        lines.append("📅 Week 1 of month: ✅ YES — entry window open")
    else:
        lines.append("📅 Week 1 of month: ⏳ NO  — wait for next week 1")

    # ── EU crisis (with hysteresis band annotation) ────────────────────────
    if eu_crisis:
        lines.append(
            f"⚠️  EU Crisis filter: 🔴 ACTIVE  "
            f"(spread {spread:+.2f} — clears when &lt; {EU_CRISIS_EXIT:.0f})"
        )
    elif spread > EU_CRISIS_EXIT:
        lines.append(
            f"🛡️  EU Crisis filter: 🟡 WATCH  "
            f"(spread {spread:+.2f} — trigger at {EU_CRISIS_ENTER:.0f})"
        )
    else:
        lines.append(
            f"🛡️  EU Crisis filter: ✅ CLEAR  "
            f"(spread {spread:+.2f} &lt; {EU_CRISIS_ENTER:.0f})"
        )

    # ── Correlation risk ───────────────────────────────────────────────────
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

    # ── HAR-RV confirming signal ───────────────────────────────────────────
    lines += [
        "",
        "📈 <b>HAR-RV Forecast (vol regime):</b>",
        _har_line("VIX   ", result["har_vix"]),
        _har_line("vStoxx", result["har_vstoxx"]),
    ]
    if har_green:
        lines.append("   → ✅ Both mean-reverting — vol regime supports entry")
    else:
        lines.append("   → ⚠️  Accelerating vol — consider reduced size")

    lines.append("")

    # ── Final verdict ──────────────────────────────────────────────────────
    if correlation_alert:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🔴 <b>FLATTEN POSITION</b> — Correlation spike detected",
            "   Both VIX and vStoxx rose >30% in 5 days",
            "   Spread thesis breaks down — exit immediately",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    elif result["enter"] and har_green:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🟢 <b>ENTER TRADE</b>  (all signals clear)",
            f"   • Short  <b>1×</b>  VIX futures   (^VIX)",
            f"   • Long   <b>{VSTOXX_PER_VIX}×</b>  vStoxx futures (^V2TX)",
            "   • Dollar-neutral position",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    elif result["enter"] and not har_green:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🟡 <b>ENTER — REDUCED SIZE</b>",
            "   Week 1 open & EU filter clear,",
            "   but HAR-RV shows accelerating vol.",
            f"   • Short  <b>1×</b>  VIX futures   (^VIX)",
            f"   • Long   <b>{VSTOXX_PER_VIX}×</b>  vStoxx futures (^V2TX)",
            "   ⚠️  Consider half-size position",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    elif not result["week_one"]:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "⏳ <b>NO SIGNAL</b> — not in entry week",
            "   Wait for week 1 of next month",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    else:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🔴 <b>SKIP ENTRY</b> — EU crisis filter active",
            f"   Spread ({spread:+.2f}) &gt; trigger ({EU_CRISIS_ENTER:.0f})",
            f"   Clears when spread &lt; {EU_CRISIS_EXIT:.0f}",
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

    state = load_state()
    logging.info("Loaded state: %s", state)

    try:
        # ── Spot values ───────────────────────────────────────────────────────
        vix    = fetch_vix()
        vstoxx = fetch_vstoxx()

        # ── 5-day % changes (correlation risk) ────────────────────────────────
        pct_change_vix,    _ = fetch_vix_5d_pct()
        pct_change_vstoxx, _ = fetch_vstoxx_5d_pct()

        # ── HAR-RV forecasts ──────────────────────────────────────────────────
        logging.info("Fitting HAR-RV models…")
        har_vix    = compute_har_forecast(fetch_vix_history(),    label="VIX")
        har_vstoxx = compute_har_forecast(fetch_vstoxx_history(), label="vStoxx")

        # ── EU crisis with hysteresis ─────────────────────────────────────────
        spread           = vstoxx - vix
        eu_crisis_active = update_eu_crisis(spread, state["eu_crisis_active"])

        if eu_crisis_active != state["eu_crisis_active"]:
            logging.info(
                "EU crisis state changed: %s → %s  (spread %.2f)",
                state["eu_crisis_active"], eu_crisis_active, spread,
            )

        # ── Evaluate & send ───────────────────────────────────────────────────
        result  = evaluate(
            vix, vstoxx, eu_crisis_active,
            pct_change_vix, pct_change_vstoxx,
            har_vix, har_vstoxx,
        )
        message = build_message(vix, vstoxx, result)

        logging.info("\n%s", message)
        send_telegram(bot_token, chat_id, message)

        # ── Persist state ─────────────────────────────────────────────────────
        state["eu_crisis_active"] = eu_crisis_active
        state["last_run"]         = str(date.today())
        save_state(state)

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
