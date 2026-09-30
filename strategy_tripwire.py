# strategy_tripwire.py
# Шаг 2: pending переставляется на каждый бар по актуальным ZigZag-экстремумам

import os
import numpy as np
import pandas as pd

from main import (
    get_historical_data,
    calculate_indicators,
    SPREAD_POINTS,
    COMMISSION_PER_LOT,
    CONTRACT_SIZE,
    RISK_PER_TRADE,
    calculate_lot_size,
    LOG_DIR,
)

ZZ_DEPTH = 12
ZZ_BACKSTEP = 3
SL_POINTS = 300
TP_POINTS = 1500
POINT = 0.01
INITIAL_BALANCE = 10000.0


def zigzag(df: pd.DataFrame, depth: int = 12, backstep: int = 3) -> pd.DataFrame:
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    zz_high = np.full(n, np.nan)
    zz_low = np.full(n, np.nan)

    last_pivot = None
    candidate_high_i, candidate_high = None, -np.inf
    candidate_low_i, candidate_low = None, np.inf

    for i in range(depth, n - depth):
        w_high = high[i - depth:i + depth + 1]
        w_low = low[i - depth:i + depth + 1]
        is_peak = high[i] >= w_high.max()
        is_trough = low[i] <= w_low.min()

        if is_peak:
            if last_pivot is None or last_pivot[0] == "low":
                if candidate_high_i is not None and i - candidate_high_i < backstep:
                    if high[i] > candidate_high:
                        candidate_high_i, candidate_high = i, high[i]
                else:
                    if candidate_high_i is not None:
                        zz_high[candidate_high_i] = candidate_high
                        last_pivot = ("high", candidate_high_i, candidate_high)
                    candidate_high_i, candidate_high = i, high[i]
            elif last_pivot[0] == "high" and high[i] > last_pivot[2]:
                zz_high[last_pivot[1]] = np.nan
                zz_high[i] = high[i]
                last_pivot = ("high", i, high[i])
                candidate_high_i, candidate_high = None, -np.inf

        if is_trough:
            if last_pivot is None or last_pivot[0] == "high":
                if candidate_low_i is not None and i - candidate_low_i < backstep:
                    if low[i] < candidate_low:
                        candidate_low_i, candidate_low = i, low[i]
                else:
                    if candidate_low_i is not None:
                        zz_low[candidate_low_i] = candidate_low
                        last_pivot = ("low", candidate_low_i, candidate_low)
                    candidate_low_i, candidate_low = i, low[i]
            elif last_pivot[0] == "low" and low[i] < last_pivot[2]:
                zz_low[last_pivot[1]] = np.nan
                zz_low[i] = low[i]
                last_pivot = ("low", i, low[i])
                candidate_low_i, candidate_low = None, np.inf

    if candidate_high_i is not None and (last_pivot is None or last_pivot[0] == "low"):
        zz_high[candidate_high_i] = candidate_high
    if candidate_low_i is not None and (last_pivot is None or last_pivot[0] == "high"):
        zz_low[candidate_low_i] = candidate_low

    out = df.copy()
    out["zz_high"] = zz_high
    out["zz_low"] = zz_low
    return out


def last_confirmed_extrema(df: pd.DataFrame, i: int):
    conf_high = conf_low = None
    for j in range(i - 1, -1, -1):
        if conf_high is None and not np.isnan(df["zz_high"].iloc[j]):
            conf_high = float(df["zz_high"].iloc[j])
        if conf_low is None and not np.isnan(df["zz_low"].iloc[j]):
            conf_low = float(df["zz_low"].iloc[j])
        if conf_high is not None and conf_low is not None:
            break
    return conf_high, conf_low


def make_pending(side: str, entry: float, price: float):
    """Создать pending только если цена по правильную сторону от entry."""
    if side == "BUY" and entry <= price:
        return None
    if side == "SELL" and entry >= price:
        return None
    sl = entry - SL_POINTS * POINT if side == "BUY" else entry + SL_POINTS * POINT
    tp = entry + TP_POINTS * POINT if side == "BUY" else entry - TP_POINTS * POINT
    return {"side": side, "entry": entry, "sl": sl, "tp": tp}


def run_tripwire(df: pd.DataFrame, initial_balance: float = INITIAL_BALANCE):
    df = zigzag(df, depth=ZZ_DEPTH, backstep=ZZ_BACKSTEP)

    balance = initial_balance
    trades = []
    pos = None
    pending = None
    last_side = 0

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        t = row["time"]

        conf_high, conf_low = last_confirmed_extrema(df, i)

        # --- 1. Открытая позиция: SL / TP ---
        if pos is not None:
            hit_sl = (pos["side"] == "BUY" and low <= pos["sl"]) or \
                     (pos["side"] == "SELL" and high >= pos["sl"])
            hit_tp = (pos["side"] == "BUY" and high >= pos["tp"]) or \
                     (pos["side"] == "SELL" and low <= pos["tp"])
            if hit_sl or hit_tp:
                exit_price = pos["sl"] if hit_sl else pos["tp"]
                gross = (exit_price - pos["entry"]) if pos["side"] == "BUY" \
                    else (pos["entry"] - exit_price)
                cost = (SPREAD_POINTS + COMMISSION_PER_LOT / CONTRACT_SIZE) \
                       * pos["lots"] * CONTRACT_SIZE
                pnl = gross * pos["lots"] * CONTRACT_SIZE - cost
                balance += pnl
                trades.append({
                    "time": str(t),
                    "side": pos["side"],
                    "entry": pos["entry"],
                    "exit": exit_price,
                    "lots": pos["lots"],
                    "profit": pnl,
                    "balance": balance,
                    "reason": "SL" if hit_sl else "TP",
                })
                last_side = 1 if pos["side"] == "BUY" else -1
                pos = None

        # --- 2. Срабатывание pending ---
        if pos is None and pending is not None:
            filled = (pending["side"] == "BUY" and high >= pending["entry"]) or \
                     (pending["side"] == "SELL" and low <= pending["entry"])
            if filled:
                lots = calculate_lot_size(
                    balance, RISK_PER_TRADE,
                    abs(pending["entry"] - pending["sl"])
                )
                pos = {
                    "side": pending["side"],
                    "entry": pending["entry"],
                    "sl": pending["sl"],
                    "tp": pending["tp"],
                    "lots": lots,
                }
                pending = None

        # --- 3. Перестановка pending на актуальные экстремумы ---
        if pos is None and conf_high is not None and conf_low is not None:
            if last_side == 1:
                side, entry = "SELL", conf_low
            elif last_side == -1:
                side, entry = "BUY", conf_high
            else:
                if abs(conf_high - price) <= abs(conf_low - price):
                    side, entry = "BUY", conf_high
                else:
                    side, entry = "SELL", conf_low

            new_pending = make_pending(side, entry, price)
            # всегда обновляем (как reprice в EA)
            pending = new_pending

    os.makedirs(LOG_DIR, exist_ok=True)
    pd.DataFrame(trades).to_csv(os.path.join(LOG_DIR, "tripwire_trades.csv"), index=False)

    n = len(trades)
    if n == 0:
        print("Сделок не было.")
        return

    wins = sum(1 for t in trades if t["profit"] > 0)
    gw = sum(t["profit"] for t in trades if t["profit"] > 0)
    gl = -sum(t["profit"] for t in trades if t["profit"] <= 0)
    pf = gw / gl if gl > 0 else float("inf")

    print(f"Сделок:        {n}")
    print(f"Winrate:       {wins/n:.1%}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Финальный баланс: ${balance:,.2f}")
    print(f"Net P/L:       ${balance - initial_balance:,.2f}")
    print(f"Файл: logs/tripwire_trades.csv")


if __name__ == "__main__":
    print("=" * 60)
    print("Tripwire rule-based (шаг 2) | XAUUSD | pending refresh")
    print("=" * 60)

    os.environ["RESAMPLE_TF"] = "15min"
    df = get_historical_data(bars=9999999)
    df = calculate_indicators(df)

    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2026-07-01", tz="UTC")
    df = df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)

    print(f"Баров: {len(df)} | {df['time'].iloc[0]} .. {df['time'].iloc[-1]}")
    print()
    run_tripwire(df)