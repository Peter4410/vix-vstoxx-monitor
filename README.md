# vix-vstoxx-monitor

GitHub Actions bot that checks VIX/vStoxx spread entry conditions daily and sends a Telegram alert.

## Strategy

| Parameter | Value |
|-----------|-------|
| **Regime filter** | None — always on |
| **Entry window** | Week 1 of each month (days 1–7) |
| **Crisis skip** | vStoxx − VIX > 10 (EU dislocation) |
| **Position** | Short 1× VIX futures, Long 8× vStoxx futures |
| **Rationale** | Dollar-neutral; exploits mean-reversion in the VIX/vStoxx spread |

### Decision tree

```
Every trading day:
  ├── Is today in week 1 (day 1–7)?
  │     NO  → ⏳ WAIT — no signal
  │     YES ↓
  └── Is vStoxx − VIX > 10?
        YES → 🔴 SKIP — EU crisis filter triggered
        NO  → 🟢 ENTER — Short 1× VIX, Long 8× vStoxx
```

## Schedule

The workflow runs at **21:30 UTC** every day:
- Winter (EST, UTC−5): **4:30 PM ET**
- Summer (EDT, UTC−4): **5:30 PM ET**

This ensures VIX and vStoxx closing prices are always available. You can also trigger it manually via **Actions → Run workflow**.

## Setup

### 1. Fork / clone this repo

### 2. Add Telegram secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret name | Value |
|-------------|-------|
| `TELEGRAM_BOT_TOKEN` | Your bot token from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Your chat / channel ID |

### 3. Enable Actions

Actions are enabled by default on new repos. The workflow will fire automatically at the scheduled time.

## Files

```
.
├── monitor.py                        # Core logic
├── requirements.txt                  # Python dependencies
└── .github/
    └── workflows/
        └── monitor.yml               # GitHub Actions schedule
```

## Data sources

| Index | Ticker | Provider |
|-------|--------|----------|
| VIX   | `^VIX`  | Yahoo Finance via yfinance |
| vStoxx | `^V2TX` | Yahoo Finance via yfinance |

## Sample Telegram alerts

**Entry signal (week 1, spread clear):**
```
📊 VIX / vStoxx Monitor
📅 Monday, 03 Feb 2025

  VIX    (^VIX):  17.43
  vStoxx (^V2TX): 18.92
  Spread (vStoxx − VIX): +1.49

📅 Week 1 of month: ✅ YES — entry window open
🛡️  EU Crisis filter: ✅ CLEAR  (spread +1.49 ≤ 10)

━━━━━━━━━━━━━━━━━━━━━━━
🟢 ENTER TRADE
   • Short  1×  VIX futures   (^VIX)
   • Long   8×  vStoxx futures (^V2TX)
   • Dollar-neutral position
━━━━━━━━━━━━━━━━━━━━━━━
```

**Outside entry window:**
```
📊 VIX / vStoxx Monitor
📅 Wednesday, 12 Feb 2025

  VIX    (^VIX):  16.80
  vStoxx (^V2TX): 17.60
  Spread (vStoxx − VIX): +0.80

📅 Week 1 of month: ⏳ NO  — wait for next week 1
🛡️  EU Crisis filter: ✅ CLEAR  (spread +0.80 ≤ 10)

━━━━━━━━━━━━━━━━━━━━━━━
⏳ NO SIGNAL — not in entry week
   Wait for week 1 of next month
━━━━━━━━━━━━━━━━━━━━━━━
```

**EU crisis filter triggered:**
```
📅 Week 1 of month: ✅ YES — entry window open
⚠️  EU Crisis filter: 🔴 TRIGGERED  (spread +12.30 > 10)

━━━━━━━━━━━━━━━━━━━━━━━
🔴 SKIP ENTRY — EU crisis filter active
   Spread (+12.30) exceeds threshold (10)
   Monitor daily; re-assess when spread normalises
━━━━━━━━━━━━━━━━━━━━━━━
```

## Related

- [vix-ewma-alert](https://github.com/Peter4410/vix-ewma-alert) — VIX vs EWMA(λ=0.97) daily monitor
