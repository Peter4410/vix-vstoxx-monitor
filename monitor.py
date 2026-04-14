#!/usr/bin/env python3
"""
monitor.py — VIX/vStoxx spread monitor for week-1 entry conditions.

Strategy (always-on, no regime filter):
  - Check each trading day; signal is relevant only in week 1 (days 1–7).
  - Skip entry if vStoxx − VIX spread triggers EU crisis filter (hysteresis band).
  - Skip entry if MOVE/VIX ratio exceeds cross-asset stress threshold.
  - Position: Short 1× VIX futures, Long ~8× vStoxx futures (dollar-neutral).

Changes vs v1:
  [EU Crisis]  Threshold tightened from 10pt to 7pt with a 2-pt hysteresis band.
               Enter crisis mode when spread > 7.0; exit only when spread < 5.0.
               State (eu_crisis_active) is persisted in state.json.
  [HAR-RV]     Heterogeneous Autoregressive Realized Volatility model added as a
               confirming signal. Uses three lagged vol averages (1-day, 5-day,
               22-day) to forecast whether VIX / vStoxx is mean-reverting or
               accelerating. Green-light if mean-reverting; caution if accelerating.

Changes vs v2:
  [MOVE Filter]  Skip new entries when MOVE/VIX > 1.5 (bond-vol stress signal).
  [Time Stop]    If spread hasn't compressed ≥1.5 pts in 45 days → reduce to 50%.
  [Roll Alert]   Alert 5–7 days before VIX futures monthly expiry (Wed before 3rd Fri).
  [Dollar Drift] Alert when long/short dollar exposure drifts >15% from entry.
  [Trade State]  Trade entry/exit now tracked in state.json.

Changes vs v3:
  [Time Stop v2] Time-stop is now PERSISTENT — every daily message shows "⚠️ REDUCED 50%"
                 while the flag is active. Clears automatically once the spread compresses
                 ≥1.5 pts from entry (restores full size with a recovery notification).
  [Kelly Sizing] Kelly Criterion framework added: computes optimal position size from
                 rolling historical P&L using K = μ/σ². Recommends VIX & vStoxx contract
                 counts anchored to backtested edge magnitude and variance. Recommended
                 contract count is tracked in state.json and shown in every daily message.

Data sources:
  VIX    → Yahoo Finance    (^VIX)   — via yfinance
  vStoxx → Yahoo Finance    (^V2TX)  — via yfinance
  MOVE   → Yahoo Finance    (^MOVE)  — via yfinance
"""

import json
import os
import sys
import time
import logging
from datetime import date, datetime, timedelta
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

# ── MOVE cross-asset stress filter ────────────────────────────────────────────
MOVE_VIX_THRESHOLD    = 1.5    # Skip entry when MOVE/VIX > this

# ── Time stop ─────────────────────────────────────────────────────────────────
TIME_STOP_DAYS        = 45     # Days in trade before time-stop check activates
TIME_STOP_COMPRESSION = 1.5    # Minimum spread compression (pts) required to avoid stop

# ── Dollar neutrality ─────────────────────────────────────────────────────────
DOLLAR_DRIFT_ALERT    = 0.15   # Rebalance alert when drift exceeds 15%

# ── VIX futures roll window ───────────────────────────────────────────────────
ROLL_ALERT_MIN_DAYS   = 5      # Alert this many days before VIX expiry
ROLL_ALERT_MAX_DAYS   = 7      # ...to this many days before expiry

# ── Kelly position sizing ──────────────────────────────────────────────────────
KELLY_BASE_CAPITAL   = 30_000  # Capital allocated to this strategy ($)
KELLY_MIN_HISTORY    = 60      # Minimum overlapping trading days to compute Kelly
KELLY_MAX_MULTIPLIER = 2.0     # Maximum Kelly multiplier (cap leverage)
KELLY_ANNUALIZE      = 252     # Trading days per year

# ── HAR-RV parameters ─────────────────────────────────────────────────────────
HAR_MIN_ROWS = 100   # Minimum trading days of history needed to fit the model

# ── Network settings ──────────────────────────────────────────────────────────
RETRIES     = 3
RETRY_DELAY = 5   # seconds (multiplied by attempt number)

# ── State file ─────────────────────────────────────────────────────────────────
STATE_FILE = Path(__file__).parent / "state.json"

DEFAULT_STATE: dict = {
    "eu_crisis_active":    False,   # hysteresis: True once spread > 7, until spread < 5
    "last_run":            None,
    # Trade tracking
    "trade_active":        False,   # True after ENTER signal, until correlation exit
    "trade_entry_date":    None,    # ISO date string of entry
    "trade_entry_spread":  None,    # vStoxx − VIX spread at entry
    "trade_entry_vix":     None,    # VIX level at entry (dollar-drift reference)
    "trade_entry_vstoxx":  None,    # vStoxx level at entry (dollar-drift reference)
    "time_stop_alerted":   False,   # True once the one-time initial time-stop alert has fired
    "time_stop_active":    False,   # True while position is PERSISTENTLY at 50% (clears on compression)
    "roll_alerted_expiry": None,    # ISO date of the expiry we already alerted for
    # Kelly-based position tracking
    "recommended_vix_contracts":    0,     # Kelly-recommended VIX contracts (short)
    "recommended_vstoxx_contracts": 0,     # Kelly-recommended vStoxx contracts (long)
    "kelly_fraction":               None,  # Last computed Kelly fraction (float)
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


def fetch_move() -> float | None:
    """
    Fetch the ICE BofA MOVE Index (^MOVE) from Yahoo Finance.
    Returns None if data is unavailable (MOVE filter is skipped gracefully).
    """
    ticker = "^MOVE"
    for attempt in range(1, RETRIES + 1):
        try:
            logging.info("Fetching %s (attempt %d)…", ticker, attempt)
            df = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
            if df.empty:
                raise RuntimeError("No data returned for ^MOVE")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if close.empty:
                raise RuntimeError("'Close' column empty for ^MOVE")
            val = float(close.iloc[-1])
            logging.info("  MOVE = %.4f  (date: %s)", val, close.index[-1].date())
            return val
        except Exception as exc:
            logging.warning("Attempt %d failed for ^MOVE: %s", attempt, exc)
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    logging.warning("MOVE fetch failed after %d attempts — MOVE filter skipped.", RETRIES)
    return None


def _fetch_vstoxx_series(period: str = "5d") -> pd.Series:
    """Download VSTOXX (^V2TX) close-price series from Yahoo Finance."""
    ticker = "^V2TX"
    for attempt in range(1, RETRIES + 1):
        try:
            logging.info("Fetching %s period=%s (attempt %d)…", ticker, period, attempt)
            df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            if df.empty:
                raise RuntimeError("No data returned for ^V2TX")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if close.empty:
                raise RuntimeError("'Close' column empty for ^V2TX")
            return close
        except Exception as exc:
            logging.warning("Attempt %d failed for ^V2TX: %s", attempt, exc)
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def fetch_vstoxx() -> float:
    """Fetch the latest VSTOXX closing price from Yahoo Finance (^V2TX)."""
    close = _fetch_vstoxx_series(period="5d")
    val = float(close.iloc[-1])
    logging.info("  vStoxx = %.4f  (date: %s)", val, close.index[-1].date())
    return val


def fetch_vstoxx_5d_pct() -> tuple[float, float]:
    """Return (pct_5d_change, today_value) for vStoxx."""
    close = _fetch_vstoxx_series(period="1mo")
    if len(close) < 6:
        raise RuntimeError(f"Insufficient ^V2TX history ({len(close)} rows, need ≥ 6)")
    today_val     = float(close.iloc[-1])
    five_days_ago = float(close.iloc[-6])
    pct = (today_val - five_days_ago) / five_days_ago * 100
    logging.info("  vStoxx 5-day: %.4f → %.4f  (%+.2f%%)", five_days_ago, today_val, pct)
    return pct, today_val


def fetch_vstoxx_history() -> pd.Series:
    """
    Fetch vStoxx (^V2TX) history from Yahoo Finance for HAR-RV fitting.
    HAR model needs ≥ 100 rows; returns whatever is available (model skips if too short).
    """
    for period in ["2y", "1y", "6mo", "3mo"]:
        try:
            series = _fetch_vstoxx_series(period=period)
            logging.info("  vStoxx history: %d rows (period=%s)", len(series), period)
            return series
        except Exception as exc:
            logging.warning("vStoxx history period=%s failed: %s — trying shorter period…", period, exc)
    logging.warning("vStoxx history: all periods failed — HAR-RV will be skipped.")
    return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Kelly Criterion position sizing
# ─────────────────────────────────────────────────────────────────────────────

def compute_kelly_sizing(
    vix_history:    pd.Series,
    vstoxx_history: pd.Series,
    vix:            float,
    capital:        float = KELLY_BASE_CAPITAL,
) -> dict | None:
    """
    Compute Kelly-optimal position size from historical spread P&L.

    Model: Short 1 VIX futures (×$1,000/pt) + Long 8 vStoxx futures (×$100/pt).
    Daily P&L = −ΔVIX × 1,000  +  ΔvStoxx × 8 × 100  (FX ignored for sizing).

    Continuous Kelly criterion for normally distributed returns:
        K = μ / σ²
    where μ = mean daily P&L relative to capital,
          σ² = variance of daily P&L relative to capital.

    Kelly fraction is clamped to [0, KELLY_MAX_MULTIPLIER].
    Returns None if fewer than KELLY_MIN_HISTORY overlapping days are available.
    """
    aligned = pd.DataFrame(
        {"vix": vix_history, "vstoxx": vstoxx_history}
    ).dropna()

    if len(aligned) < KELLY_MIN_HISTORY:
        logging.warning(
            "Kelly: only %d overlapping rows (need ≥ %d) — skipped.",
            len(aligned), KELLY_MIN_HISTORY,
        )
        return None

    delta_vix    = aligned["vix"].diff().dropna()
    delta_vstoxx = aligned["vstoxx"].diff().dropna()
    common_idx   = delta_vix.index.intersection(delta_vstoxx.index)

    # Dollar P&L per day for the base 1×VIX / 8×vStoxx position
    daily_pnl = (
        -delta_vix[common_idx]    * 1_000       # short VIX leg
        + delta_vstoxx[common_idx] * 8 * 100    # long vStoxx leg
    )
    daily_returns = daily_pnl / capital          # dimensionless return

    mu     = float(daily_returns.mean())
    sigma2 = float(daily_returns.var())

    if sigma2 <= 0:
        logging.warning("Kelly: zero return variance — skipped.")
        return None

    raw_kelly      = mu / sigma2
    kelly_fraction = float(np.clip(raw_kelly, 0.0, KELLY_MAX_MULTIPLIER))

    # Translate Kelly fraction → integer contract recommendation
    target_notional      = kelly_fraction * capital
    kelly_vix            = max(1, round(target_notional / (vix * 1_000)))
    kelly_vstoxx         = max(1, round(kelly_vix * VSTOXX_PER_VIX))

    # Annualised statistics for display
    ann_return = mu     * KELLY_ANNUALIZE
    ann_vol    = float(daily_returns.std()) * np.sqrt(KELLY_ANNUALIZE)
    sharpe     = (ann_return / ann_vol) if ann_vol > 0 else 0.0

    # Regime label (informational — user applies discretion)
    if vix < 15:
        regime, regime_mult = "Optimal",  1.5
    elif vix < 22:
        regime, regime_mult = "Normal",   1.0
    elif vix < 30:
        regime, regime_mult = "Elevated", 0.7
    else:
        regime, regime_mult = "High Vol", 0.7

    logging.info(
        "  Kelly: fraction=%.3f (raw=%.3f)  VIX=%d×  vStoxx=%d×  "
        "ann_return=%+.1f%%  ann_vol=%.1f%%  Sharpe=%.2f  N=%d days",
        kelly_fraction, raw_kelly, kelly_vix, kelly_vstoxx,
        ann_return * 100, ann_vol * 100, sharpe, len(daily_returns),
    )

    return {
        "kelly_fraction":          kelly_fraction,
        "raw_kelly":               raw_kelly,
        "kelly_vix_contracts":     kelly_vix,
        "kelly_vstoxx_contracts":  kelly_vstoxx,
        "regime":                  regime,
        "regime_multiplier":       regime_mult,
        "ann_return":              ann_return,
        "ann_vol":                 ann_vol,
        "sharpe":                  sharpe,
        "sample_days":             len(daily_returns),
        "daily_mean_pnl":          float(daily_pnl.mean()),
    }


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
# Risk management helpers
# ─────────────────────────────────────────────────────────────────────────────

def next_vix_expiry(today: date) -> date:
    """
    Return the next upcoming VIX futures expiry date.
    VIX futures expire on the Wednesday before the 3rd Friday of each month.
    If this month's expiry has already passed, returns next month's.
    """
    def _expiry_for_month(year: int, month: int) -> date:
        first_day = date(year, month, 1)
        # Days until first Friday (Friday = weekday 4)
        first_friday_offset = (4 - first_day.weekday()) % 7
        third_friday = first_day + timedelta(days=first_friday_offset + 14)
        return third_friday - timedelta(days=2)   # Wednesday before 3rd Friday

    exp = _expiry_for_month(today.year, today.month)
    if exp <= today:
        next_month = today.month + 1 if today.month < 12 else 1
        next_year  = today.year if today.month < 12 else today.year + 1
        exp = _expiry_for_month(next_year, next_month)
    return exp


def check_time_stop(
    state: dict, spread: float, today: date
) -> tuple[bool, bool, bool, int, float]:
    """
    Evaluate the 45-day time stop.

    Returns (should_fire, is_active, should_clear, days_elapsed, compression_pts).
      should_fire   — True once when conditions are first met (one-time initial alert).
      is_active     — True on every run while time_stop_active is set in state.
      should_clear  — True when time_stop was active but spread has now compressed ≥1.5 pts.
      days_elapsed  — Calendar days since trade entry.
      compression   — entry_spread − current_spread (positive = narrowed = good).
    """
    if not state.get("trade_active"):
        return False, False, False, 0, 0.0

    entry_date_str = state.get("trade_entry_date")
    entry_spread   = state.get("trade_entry_spread")
    if not entry_date_str or entry_spread is None:
        return False, False, False, 0, 0.0

    entry_date   = date.fromisoformat(entry_date_str)
    days_elapsed = (today - entry_date).days
    compression  = float(entry_spread) - spread   # positive = spread narrowed

    already_alerted  = state.get("time_stop_alerted", False)
    currently_active = state.get("time_stop_active",  False)

    # Fire the one-time alert when threshold is first breached
    should_fire = (
        not already_alerted
        and days_elapsed >= TIME_STOP_DAYS
        and compression  < TIME_STOP_COMPRESSION
    )

    # Clear the persistent flag once the spread has finally compressed enough
    should_clear = (
        currently_active
        and compression >= TIME_STOP_COMPRESSION
    )

    return should_fire, currently_active, should_clear, days_elapsed, compression


def check_dollar_drift(state: dict, vix: float, vstoxx: float) -> tuple[bool, float, float, float]:
    """
    Returns (alert, drift_fraction, entry_ratio, current_ratio).
    Drift = |current_ratio / entry_ratio − 1| where ratio = vstoxx / vix.
    Returns (False, 0.0, 0.0, current_ratio) if no active trade or missing entry data.
    """
    current_ratio = vstoxx / vix
    if not state.get("trade_active"):
        return False, 0.0, current_ratio, current_ratio

    entry_vix    = state.get("trade_entry_vix")
    entry_vstoxx = state.get("trade_entry_vstoxx")
    if not entry_vix or not entry_vstoxx:
        return False, 0.0, current_ratio, current_ratio

    entry_ratio = float(entry_vstoxx) / float(entry_vix)
    drift       = abs(current_ratio / entry_ratio - 1)
    alert       = drift > DOLLAR_DRIFT_ALERT
    logging.info(
        "  Dollar drift: entry_ratio=%.4f  current_ratio=%.4f  drift=%.1f%%",
        entry_ratio, current_ratio, drift * 100,
    )
    return alert, drift, entry_ratio, current_ratio


def check_roll_alert(state: dict, today: date) -> tuple[bool, date, int]:
    """
    Returns (should_alert, expiry_date, days_to_expiry).
    should_alert is True when 5–7 days from expiry and not yet alerted for that expiry.
    """
    expiry          = next_vix_expiry(today)
    days_to_expiry  = (expiry - today).days
    already_alerted = (state.get("roll_alerted_expiry") == str(expiry))
    should_alert    = (
        ROLL_ALERT_MIN_DAYS <= days_to_expiry <= ROLL_ALERT_MAX_DAYS
        and not already_alerted
    )
    return should_alert, expiry, days_to_expiry


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
    move:              float | None = None,
    today:             date | None = None,
) -> dict:
    """Evaluate all entry conditions and return a result dict."""
    spread   = vstoxx - vix
    week_one = is_week_one(today)
    enter    = week_one and not eu_crisis_active

    correlation_alert = (
        pct_change_vix    > CORRELATION_ALERT_THRESHOLD
        and pct_change_vstoxx > CORRELATION_ALERT_THRESHOLD
    )

    # HAR confirming signal: yellow if EITHER forecast is accelerating
    har_green = not (
        (har_vix    and har_vix["direction"]    == "accelerating") or
        (har_vstoxx and har_vstoxx["direction"] == "accelerating")
    )

    # MOVE/VIX cross-asset stress filter
    move_ratio         = (move / vix) if move is not None else None
    move_filter_active = (move_ratio is not None) and (move_ratio > MOVE_VIX_THRESHOLD)
    effective_enter    = enter and not move_filter_active

    return {
        "spread":             spread,
        "eu_crisis":          eu_crisis_active,
        "week_one":           week_one,
        "enter":              enter,             # week_one & not eu_crisis (ignores MOVE)
        "effective_enter":    effective_enter,   # also accounts for MOVE filter
        "pct_change_vix":     pct_change_vix,
        "pct_change_vstoxx":  pct_change_vstoxx,
        "correlation_alert":  correlation_alert,
        "har_vix":            har_vix,
        "har_vstoxx":         har_vstoxx,
        "har_green":          har_green,
        "move":               move,
        "move_ratio":         move_ratio,
        "move_filter_active": move_filter_active,
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


def build_message(vix: float, vstoxx: float, result: dict, risk_info: dict) -> str:
    today_str          = date.today().strftime("%A, %d %b %Y")
    spread             = result["spread"]
    pct_change_vix     = result["pct_change_vix"]
    pct_change_vstoxx  = result["pct_change_vstoxx"]
    correlation_alert  = result["correlation_alert"]
    eu_crisis          = result["eu_crisis"]
    har_green          = result["har_green"]
    move               = result["move"]
    move_ratio         = result["move_ratio"]
    move_filter_active = result["move_filter_active"]

    # Risk-info convenience aliases
    time_stop_active  = risk_info.get("time_stop_active",  False)
    time_stop_cleared = risk_info.get("time_stop_cleared", False)
    kelly             = risk_info.get("kelly")
    rec_vix           = risk_info.get("recommended_vix_contracts",    1)
    rec_vstoxx        = risk_info.get("recommended_vstoxx_contracts", VSTOXX_PER_VIX)

    lines = [
        "📊 <b>VIX / vStoxx Monitor</b>",
        f"📅 {today_str}",
        "",
        f"  VIX    (^VIX):  <b>{vix:.2f}</b>",
        f"  vStoxx (^V2TX): <b>{vstoxx:.2f}</b>",
        f"  Spread (vStoxx − VIX): <b>{spread:+.2f}</b>",
        "",
    ]

    # ── Persistent time-stop banner (shown at top if active) ──────────────
    if time_stop_cleared:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "✅ <b>TIME STOP CLEARED — Restore Full Size</b>",
            f"   Spread compressed ≥{TIME_STOP_COMPRESSION:.1f} pts from entry",
            f"   Recommended: Short <b>{rec_vix}×</b> VIX, Long <b>{rec_vstoxx}×</b> vStoxx",
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]
    elif time_stop_active:
        days_in = risk_info.get("days_in_trade", 0)
        ts_comp = risk_info.get("time_stop_compression", 0.0)
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            f"⚠️  <b>TIME STOP ACTIVE — Position at 50%</b>",
            f"   {days_in}d in trade | compression {ts_comp:+.2f} pts "
            f"(need ≥{TIME_STOP_COMPRESSION:.1f} to restore)",
            f"   Holding: Short <b>{rec_vix}×</b> VIX, Long <b>{rec_vstoxx}×</b> vStoxx",
            "━━━━━━━━━━━━━━━━━━━━━━━",
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

    # ── MOVE cross-asset stress filter ────────────────────────────────────
    if move is None:
        lines.append("🎯 MOVE Filter:      ⚪ Data unavailable — filter skipped")
    elif move_filter_active:
        lines.append(
            f"🎯 MOVE Filter:      🔴 ACTIVE  "
            f"(MOVE/VIX = {move_ratio:.2f} &gt; {MOVE_VIX_THRESHOLD:.1f} — entry blocked)"
        )
    else:
        lines.append(
            f"🎯 MOVE Filter:      ✅ CLEAR  "
            f"(MOVE/VIX = {move_ratio:.2f} &lt; {MOVE_VIX_THRESHOLD:.1f})"
        )

    # ── Correlation risk ───────────────────────────────────────────────────
    if correlation_alert:
        lines.append(
            f"⚡ Corr. Risk:       🔴 ALERT — "
            f"VIX +{pct_change_vix:.1f}% &amp; vStoxx +{pct_change_vstoxx:.1f}% in 5 days"
        )
    else:
        lines.append(
            f"⚡ Corr. Risk:       ✅ CLEAR  "
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

    # ── Kelly Position Sizing ──────────────────────────────────────────────
    if kelly:
        lines += ["", "🎯 <b>Kelly Position Sizing (K = μ/σ²):</b>"]
        lines.append(
            f"  📊 Backtested edge:  {kelly['ann_return'] * 100:+.1f}%/yr  "
            f"| vol {kelly['ann_vol'] * 100:.1f}%  | Sharpe {kelly['sharpe']:.2f}  "
            f"| {kelly['sample_days']}d sample"
        )
        raw_str = f"{kelly['raw_kelly']:.2f}×" if kelly["raw_kelly"] <= KELLY_MAX_MULTIPLIER else f"{kelly['raw_kelly']:.2f}× → capped at {KELLY_MAX_MULTIPLIER:.1f}×"
        lines.append(
            f"  📐 Kelly fraction:   <b>{kelly['kelly_fraction']:.2f}×</b>  (raw: {raw_str})"
        )
        lines.append(
            f"  🔢 Recommended now:  Short <b>{kelly['kelly_vix_contracts']}×</b> VIX, "
            f"Long <b>{kelly['kelly_vstoxx_contracts']}×</b> vStoxx  "
            f"[{kelly['regime']} regime, {kelly['regime_multiplier']:.1f}× overlay]"
        )
        if time_stop_active:
            lines.append(
                f"  ⚠️  Time stop active: holding at <b>{rec_vix}×</b> VIX / "
                f"<b>{rec_vstoxx}×</b> vStoxx (50% of Kelly recommendation)"
            )
    else:
        lines += ["", "🎯 <b>Kelly Position Sizing:</b>  ⚪ Insufficient history — using base 1×/8×"]

    # ── Risk Management ────────────────────────────────────────────────────
    lines += ["", "🔧 <b>Risk Management:</b>"]

    # Roll status (always shown)
    roll_days       = risk_info["roll_days_to_expiry"]
    roll_expiry     = risk_info["roll_expiry"]
    roll_expiry_str = roll_expiry.strftime("%-d %b")   # e.g. "19 Mar"
    if risk_info["roll_should_alert"]:
        lines.append(
            f"  📅 VIX Roll: 🔔 <b>ROLL NOW</b> — expiry {roll_expiry_str} in {roll_days} days"
        )
    else:
        lines.append(
            f"  📅 VIX Roll: next expiry {roll_expiry_str} ({roll_days} days away)"
        )

    # Dollar drift + time stop (only meaningful when a trade is active)
    if risk_info["trade_active"]:
        drift_pct     = risk_info["dollar_drift_pct"] * 100
        entry_ratio   = risk_info["entry_ratio"]
        current_ratio = risk_info["current_ratio"]
        if risk_info["dollar_drift_alert"]:
            lines.append(
                f"  ⚖️  Dollar Drift: 🔴 <b>REBALANCE</b> — drift {drift_pct:.1f}%  "
                f"(entry {entry_ratio:.3f} → now {current_ratio:.3f})"
            )
        else:
            lines.append(
                f"  ⚖️  Dollar Drift: ✅ {drift_pct:.1f}%  "
                f"(entry {entry_ratio:.3f} → now {current_ratio:.3f})"
            )

        days_in = risk_info["days_in_trade"]
        ts_comp = risk_info["time_stop_compression"]
        if risk_info["time_stop_alert"]:
            lines.append(
                f"  ⏰ Time Stop: 🔴 <b>REDUCE TO 50% NOW</b> — {days_in}d in trade, "
                f"compression only {ts_comp:+.2f} pts "
                f"(need ≥ {TIME_STOP_COMPRESSION:.1f} in {TIME_STOP_DAYS}d)"
            )
        elif time_stop_active:
            lines.append(
                f"  ⏰ Time Stop: ⚠️  ACTIVE — {days_in}d in trade, "
                f"compression {ts_comp:+.2f} pts  (need ≥{TIME_STOP_COMPRESSION:.1f} to clear)"
            )
        elif days_in >= TIME_STOP_DAYS:
            lines.append(
                f"  ⏰ Time Stop: ✅ {days_in}d — compressed {ts_comp:+.2f} pts  OK"
            )
        else:
            remaining = TIME_STOP_DAYS - days_in
            lines.append(
                f"  ⏰ Time Stop: ✅ {days_in}d in trade  "
                f"(stop activates in {remaining}d w/o {TIME_STOP_COMPRESSION:.1f} pt compression)"
            )
    else:
        lines.append("  ⚖️  Dollar Drift: — (no active trade)")
        lines.append("  ⏰ Time Stop:    — (no active trade)")

    lines.append("")

    # ── Primary verdict ────────────────────────────────────────────────────
    if correlation_alert:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🔴 <b>FLATTEN POSITION</b> — Correlation spike detected",
            "   Both VIX and vStoxx rose &gt;30% in 5 days",
            "   Spread thesis breaks down — exit immediately",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    elif result["effective_enter"] and har_green:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🟢 <b>ENTER TRADE</b>  (all signals clear)",
            f"   • Short  <b>{rec_vix}×</b>  VIX futures   (^VIX)",
            f"   • Long   <b>{rec_vstoxx}×</b>  vStoxx futures (^V2TX)",
            f"   • Kelly sizing ({kelly['kelly_fraction']:.2f}× — {kelly['regime']} regime)" if kelly else "   • Base sizing (Kelly insufficient history)",
            "   • Dollar-neutral position",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    elif result["effective_enter"] and not har_green:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🟡 <b>ENTER — REDUCED SIZE</b>",
            "   Week 1 open &amp; all filters clear,",
            "   but HAR-RV shows accelerating vol.",
            f"   • Short  <b>{rec_vix}×</b>  VIX futures   (^VIX)",
            f"   • Long   <b>{rec_vstoxx}×</b>  vStoxx futures (^V2TX)",
            "   ⚠️  Consider half-size position vs Kelly recommendation",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    elif move_filter_active and result["enter"]:
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "🟡 <b>SKIP ENTRY</b> — MOVE/VIX stress filter",
            f"   MOVE/VIX = {move_ratio:.2f} &gt; {MOVE_VIX_THRESHOLD:.1f}",
            "   Bond-vol stress elevated — wait for ratio to normalise",
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

    # ── Additional action alerts (can fire alongside primary verdict) ──────
    if risk_info["roll_should_alert"]:
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━",
            f"📅 <b>ROLL VIX FUTURES</b> — expiry {roll_expiry_str} in {roll_days} days",
            "   Close current contract, open next month's position",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]

    if risk_info["time_stop_alert"]:
        days_in = risk_info["days_in_trade"]
        ts_comp = risk_info["time_stop_compression"]
        half_vix    = max(1, (rec_vix    + 1) // 2)
        half_vstoxx = max(4, (rec_vstoxx + 1) // 2)
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "⏰ <b>TIME STOP — REDUCE TO 50%</b>",
            f"   {days_in} days in trade, spread compressed only {ts_comp:+.2f} pts",
            f"   (threshold: ≥ {TIME_STOP_COMPRESSION:.1f} pts within {TIME_STOP_DAYS} days)",
            f"   Action: reduce to Short <b>{half_vix}×</b> VIX, Long <b>{half_vstoxx}×</b> vStoxx",
            "   Thesis not confirming — cut size, let spread decide",
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]

    if risk_info["dollar_drift_alert"]:
        drift_pct     = risk_info["dollar_drift_pct"] * 100
        entry_ratio   = risk_info["entry_ratio"]
        current_ratio = risk_info["current_ratio"]
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━",
            f"⚖️  <b>REBALANCE — Dollar Drift {drift_pct:.1f}%</b>",
            f"   Entry ratio: {entry_ratio:.3f}  →  Current: {current_ratio:.3f}",
            f"   Adjust vStoxx leg to restore dollar neutrality",
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

    today = date.today()
    state = load_state()
    logging.info("Loaded state: %s", state)

    try:
        # ── Spot values ───────────────────────────────────────────────────────
        vix    = fetch_vix()
        vstoxx = fetch_vstoxx()
        move   = fetch_move()

        # ── 5-day % changes (correlation risk) ────────────────────────────────
        pct_change_vix,    _ = fetch_vix_5d_pct()
        pct_change_vstoxx, _ = fetch_vstoxx_5d_pct()

        # ── History fetches (shared by HAR-RV and Kelly) ──────────────────────
        logging.info("Fetching history for HAR-RV and Kelly…")
        vix_history    = fetch_vix_history()
        vstoxx_history = fetch_vstoxx_history()

        # ── HAR-RV forecasts ──────────────────────────────────────────────────
        logging.info("Fitting HAR-RV models…")
        har_vix    = compute_har_forecast(vix_history,    label="VIX")
        har_vstoxx = compute_har_forecast(vstoxx_history, label="vStoxx")

        # ── Kelly position sizing ─────────────────────────────────────────────
        logging.info("Computing Kelly sizing…")
        kelly = compute_kelly_sizing(vix_history, vstoxx_history, vix)

        # ── EU crisis with hysteresis ─────────────────────────────────────────
        spread           = vstoxx - vix
        eu_crisis_active = update_eu_crisis(spread, state["eu_crisis_active"])

        if eu_crisis_active != state["eu_crisis_active"]:
            logging.info(
                "EU crisis state changed: %s → %s  (spread %.2f)",
                state["eu_crisis_active"], eu_crisis_active, spread,
            )

        # ── Evaluate entry conditions ─────────────────────────────────────────
        result = evaluate(
            vix, vstoxx, eu_crisis_active,
            pct_change_vix, pct_change_vstoxx,
            har_vix, har_vstoxx,
            move=move,
            today=today,
        )

        # ── Risk management checks ────────────────────────────────────────────
        time_stop_alert, time_stop_active, time_stop_cleared, days_in_trade, ts_compression = (
            check_time_stop(state, spread, today)
        )
        dollar_drift_alert, dollar_drift_pct, entry_ratio, current_ratio = (
            check_dollar_drift(state, vix, vstoxx)
        )
        roll_should_alert, roll_expiry, roll_days = check_roll_alert(state, today)

        # Current recommended contracts (may be at 50% due to time stop)
        rec_vix    = state.get("recommended_vix_contracts")    or (kelly["kelly_vix_contracts"]    if kelly else 1)
        rec_vstoxx = state.get("recommended_vstoxx_contracts") or (kelly["kelly_vstoxx_contracts"] if kelly else VSTOXX_PER_VIX)

        risk_info = {
            "roll_should_alert":            roll_should_alert,
            "roll_expiry":                  roll_expiry,
            "roll_days_to_expiry":          roll_days,
            "time_stop_alert":              time_stop_alert,
            "time_stop_active":             time_stop_active,
            "time_stop_cleared":            time_stop_cleared,
            "time_stop_compression":        ts_compression,
            "dollar_drift_alert":           dollar_drift_alert,
            "dollar_drift_pct":             dollar_drift_pct,
            "entry_ratio":                  entry_ratio,
            "current_ratio":                current_ratio,
            "trade_active":                 state["trade_active"],
            "trade_entry_date":             state.get("trade_entry_date"),
            "trade_entry_spread":           state.get("trade_entry_spread"),
            "days_in_trade":                days_in_trade,
            "kelly":                        kelly,
            "recommended_vix_contracts":    rec_vix,
            "recommended_vstoxx_contracts": rec_vstoxx,
        }

        # ── Build & send message ──────────────────────────────────────────────
        message = build_message(vix, vstoxx, result, risk_info)
        logging.info("\n%s", message)
        send_telegram(bot_token, chat_id, message)

        # ── Update state ──────────────────────────────────────────────────────
        # EU crisis hysteresis
        state["eu_crisis_active"] = eu_crisis_active

        # Correlation exit: flatten → clear all trade state
        if result["correlation_alert"] and state["trade_active"]:
            logging.info("Correlation alert fired — clearing trade state.")
            state["trade_active"]                = False
            state["trade_entry_date"]            = None
            state["trade_entry_spread"]          = None
            state["trade_entry_vix"]             = None
            state["trade_entry_vstoxx"]          = None
            state["time_stop_alerted"]           = False
            state["time_stop_active"]            = False
            state["recommended_vix_contracts"]   = 0
            state["recommended_vstoxx_contracts"]= 0

        # New entry signal: record trade state (only on first entry, not repeat signals)
        elif result["effective_enter"] and not state["trade_active"]:
            logging.info("Entry signal — recording trade entry state.")
            state["trade_active"]       = True
            state["trade_entry_date"]   = str(today)
            state["trade_entry_spread"] = spread
            state["trade_entry_vix"]    = vix
            state["trade_entry_vstoxx"] = vstoxx
            state["time_stop_alerted"]  = False
            state["time_stop_active"]   = False
            # Set Kelly-based contract recommendations at entry
            if kelly:
                state["recommended_vix_contracts"]    = kelly["kelly_vix_contracts"]
                state["recommended_vstoxx_contracts"] = kelly["kelly_vstoxx_contracts"]
            else:
                state["recommended_vix_contracts"]    = 1
                state["recommended_vstoxx_contracts"] = VSTOXX_PER_VIX

        # Time stop state machine (persistent — not one-shot)
        if time_stop_cleared:
            logging.info("Time stop CLEARED — spread compressed %.2f pts. Restoring full size.", ts_compression)
            state["time_stop_active"] = False
            # Restore Kelly recommendation (or base sizing if Kelly unavailable)
            if kelly:
                state["recommended_vix_contracts"]    = kelly["kelly_vix_contracts"]
                state["recommended_vstoxx_contracts"] = kelly["kelly_vstoxx_contracts"]
            else:
                state["recommended_vix_contracts"]    = 1
                state["recommended_vstoxx_contracts"] = VSTOXX_PER_VIX
        elif time_stop_alert:
            logging.info("Time stop alert fired — activating persistent 50%% reduction.")
            state["time_stop_alerted"] = True
            state["time_stop_active"]  = True
            # Halve the current recommendation (round up to avoid zero)
            state["recommended_vix_contracts"]    = max(1, (rec_vix    + 1) // 2)
            state["recommended_vstoxx_contracts"] = max(4, (rec_vstoxx + 1) // 2)

        # Update stored Kelly fraction for reference
        if kelly:
            state["kelly_fraction"] = kelly["kelly_fraction"]

        # Roll alert: record expiry so we don't re-alert on subsequent days
        if roll_should_alert:
            logging.info("Roll alert fired for expiry %s.", roll_expiry)
            state["roll_alerted_expiry"] = str(roll_expiry)

        state["last_run"] = str(today)
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
