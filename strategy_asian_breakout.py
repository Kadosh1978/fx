# strategy_asian_breakout.py
# Rule-based: Asian Range → London Open Breakout (XAUUSD)
#
# Логика:
#   1. Считаем high/low азиатской сессии (00:00–07:00 UTC)
#   2. После 07:00 UTC ждём пробой range
#   3. Пробой вверх → BUY, вниз → SELL. Вход по уровню пробоя
#      (или по open бара, если был гэп через уровень)
#   4. SL = противоположная граница range
#   5. TP = 1.5 × размер range
#   6. Одна сделка в день, только если range >= MIN_RANGE
#   7. Если при риске 1% лот получается меньше MIN_LOT — сделку пропускаем

import os
import pandas as pd

from main import (
    get_historical_data,
    calculate_indicators,
    SPREAD_POINTS,
    COMMISSION_PER_LOT,
    CONTRACT_SIZE,
    RISK_PER_TRADE,
    MIN_LOT,
    calculate_lot_size,
    LOG_DIR,
)

# --- Параметры стратегии ---
ASIA_START = 0
ASIA_END = 7
LONDON_START = 7
TP_RANGE_MULT = 1.5
MAX_HOLD_BARS = 48
MIN_RANGE = 3.0  # минимум $3 — узкие дни пропускаем


def run_asian_breakout(df: pd.DataFrame, initial_balance: float = 10000.0):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour

    balance = initial_balance
    trades = []
    equity = []
    pos = None
    day_range = {}
    traded_days = set()
    skipped_small_lot = 0

    for i in range(len(df)):
        row = df.iloc[i]
        t = row["time"]
        d = row["date"]
        h = row["hour"]
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        # Asian range
        if ASIA_START <= h < ASIA_END:
            if d not in day_range:
                day_range[d] = [high, low]
            else:
                day_range[d][0] = max(day_range[d][0], high)
                day_range[d][1] = min(day_range[d][1], low)

        # Manage open position
        if pos is not None:
            pos["bars"] += 1
            hit_sl = (pos["side"] == "BUY" and low <= pos["sl"]) or \
                     (pos["side"] == "SELL" and high >= pos["sl"])
            hit_tp = (pos["side"] == "BUY" and high >= pos["tp"]) or \
                     (pos["side"] == "SELL" and low <= pos["tp"])
            timeout = pos["bars"] >= MAX_HOLD_BARS

            if hit_sl or hit_tp or timeout:
                exit_price = pos["sl"] if hit_sl else (pos["tp"] if hit_tp else price)
                gross = (exit_price - pos["entry"]) if pos["side"] == "BUY" \
                    else (pos["entry"] - exit_price)
                cost = (SPREAD_POINTS + COMMISSION_PER_LOT / CONTRACT_SIZE) \
                       * pos["lots"] * CONTRACT_SIZE
                pnl = gross * pos["lots"] * CONTRACT_SIZE - cost
                balance += pnl
                reason = "SL" if hit_sl else ("TP" if hit_tp else "TIME")
                trades.append({
                    "time": t.isoformat(),
                    "side": pos["side"],
                    "lots": pos["lots"],
                    "entry": pos["entry"],
                    "exit": exit_price,
                    "profit": pnl,
                    "balance": balance,
                    "reason": reason,
                })
                pos = None

        # Entry: вход по уровню пробоя
        if pos is None and d not in traded_days and h >= LONDON_START:
            if d in day_range:
                asia_high, asia_low = day_range[d]
                range_size = asia_high - asia_low
                if range_size >= MIN_RANGE:
                    side = None
                    if high > asia_high:
                        side = "BUY"
                        entry = max(asia_high, float(row["open"]))
                        sl = asia_low
                        tp = entry + range_size * TP_RANGE_MULT
                    elif low < asia_low:
                        side = "SELL"
                        entry = min(asia_low, float(row["open"]))
                        sl = asia_high
                        tp = entry - range_size * TP_RANGE_MULT

                    if side:
                        traded_days.add(d)
                        sl_dist = abs(entry - sl)
                        raw_lots = balance * RISK_PER_TRADE / (sl_dist * CONTRACT_SIZE)
                        if raw_lots >= MIN_LOT:
                            lots = calculate_lot_size(balance, RISK_PER_TRADE, sl_dist)
                            pos = {
                                "side": side, "entry": entry, "lots": lots,
                                "sl": sl, "tp": tp, "bars": 0,
                            }
                        else:
                            skipped_small_lot += 1

        equity.append({"time": t, "equity": balance})

    os.makedirs(LOG_DIR, exist_ok=True)
    pd.DataFrame(trades).to_csv(os.path.join(LOG_DIR, "asian_breakout_trades.csv"), index=False)
    pd.DataFrame(equity).to_csv(os.path.join(LOG_DIR, "asian_breakout_equity.csv"), index=False)

    n = len(trades)
    if n == 0:
        print("Сделок не было.")
        return

    wins = sum(1 for t in trades if t["profit"] > 0)
    gross_win = sum(t["profit"] for t in trades if t["profit"] > 0)
    gross_loss = -sum(t["profit"] for t in trades if t["profit"] <= 0)
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print(f"Сделок:     {n}")
    print(f"Пропущено (лот < {MIN_LOT}): {skipped_small_lot}")
    print(f"Winrate:    {wins/n:.1%}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Финальный баланс: ${balance:,.2f}")
    print(f"Net P/L:    ${balance - initial_balance:,.2f}")
    print("Файлы: logs/asian_breakout_trades.csv, logs/asian_breakout_equity.csv")


if __name__ == "__main__":
    print("=" * 60)
    print("Asian Range → London Breakout (XAUUSD) | MIN_RANGE filter")
    print("=" * 60)

    df = get_historical_data(bars=9999999)
    df = calculate_indicators(df)

    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2026-07-01", tz="UTC")
    df = df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)

    print(f"Баров: {len(df)}")
    print(f"{df['time'].iloc[0]} .. {df['time'].iloc[-1]}")
    print()

    run_asian_breakout(df, initial_balance=10000.0)