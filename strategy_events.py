# strategy_events.py
# Event breakout v5: отложенный пробой #2–#3, ФИКСИРОВАННЫЙ ЛОТ, XAUUSD M15
# Фильтр событий через USE_EVENTS (None = все события календаря)
# R = прибыль / риск сделки (риск = размер диапазона * лот * контракт)

import os
import pandas as pd

from main import (
    get_historical_data,
    calculate_indicators,
    SPREAD_POINTS,
    COMMISSION_PER_LOT,
    CONTRACT_SIZE,
    LOG_DIR,
)

EVENTS_PATH = os.path.join("data", "macro_event_calendar_long_all.csv")
INITIAL_BALANCE = 10000.0
MIN_RANGE = 1.5
TP_MULT = 2.0
MAX_HOLD_BARS = 32

FIXED_LOT = 0.10                  # фиксированный лот, без сложного процента
USE_EVENTS = {"FOMC", "RETAIL"}   # None = все события
PRINT_EACH = True                 # False = не печатать каждую сделку

# Время выхода (ET). Порядок = приоритет, если несколько событий
# выходят в одно и то же время (остаётся первое).
# unemployment не включён: выходит вместе с NFP, дал бы дубль сделки.
EVENT_TIMES_ET = {
    "fomc": "14:00",
    "nfp": "08:30",
    "cpi": "08:30",
    "ppi": "08:30",
    "retail": "08:30",
    "gdp": "08:30",
    "pce": "08:30",
    "claims": "08:30",
    "indpro": "09:15",
    "ism_mfg": "10:00",
    "ism_services": "10:00",
    "umich": "10:00",
    "cb_confidence": "10:00",
}


def load_events(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["event_code"] = raw["event_code"].str.lower().str.strip()
    raw = raw[raw["event_code"].isin(EVENT_TIMES_ET)].copy()
    if USE_EVENTS:
        raw = raw[raw["event_code"].str.upper().isin(USE_EVENTS)].copy()

    def to_utc(row):
        d = str(row["event_date"])[:10]
        t = EVENT_TIMES_ET[row["event_code"]]
        local = pd.Timestamp(f"{d} {t}:00", tz="America/New_York")
        return local.tz_convert("UTC")

    raw["datetime_utc"] = raw.apply(to_utc, axis=1)
    raw["event"] = raw["event_code"].str.upper()
    prio = list(EVENT_TIMES_ET)
    raw["prio"] = raw["event_code"].map(prio.index)
    out = (
        raw.sort_values(["datetime_utc", "prio"])
        .drop_duplicates("datetime_utc", keep="first")
        [["datetime_utc", "event"]]
    )
    out = out[
        (out["datetime_utc"] >= "2022-01-01")
        & (out["datetime_utc"] < "2026-07-01")
    ]
    return out.reset_index(drop=True)


def run_events(df: pd.DataFrame, events: pd.DataFrame,
               initial_balance: float = INITIAL_BALANCE):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    balance = initial_balance
    trades = []

    for _, ev in events.iterrows():
        t0 = ev["datetime_utc"]
        idx = df["time"].searchsorted(t0)
        if idx >= len(df) - 6:
            continue
        if df["time"].iloc[idx] < t0:
            idx += 1
        if idx + 4 >= len(df):
            continue

        i2, i3 = idx + 1, idx + 2
        rh = max(float(df.iloc[i2]["high"]), float(df.iloc[i3]["high"]))
        rl = min(float(df.iloc[i2]["low"]), float(df.iloc[i3]["low"]))
        range_size = rh - rl
        if range_size < MIN_RANGE:
            continue

        side = None
        entry = sl = tp = None
        lots = None
        entry_i = None
        search_from = i3 + 1

        # окно: до 32 баров на вход + до 32 баров на выход
        for j in range(search_from, min(search_from + 2 * MAX_HOLD_BARS, len(df))):
            row = df.iloc[j]
            high = float(row["high"])
            low = float(row["low"])
            price = float(row["close"])

            if side is None:
                if j >= search_from + MAX_HOLD_BARS:
                    break  # вход не случился за 32 бара
                if high > rh:
                    side = "BUY"
                    entry = rh
                    sl = rl
                    tp = entry + range_size * TP_MULT
                    lots = FIXED_LOT
                    entry_i = j
                elif low < rl:
                    side = "SELL"
                    entry = rl
                    sl = rh
                    tp = entry - range_size * TP_MULT
                    lots = FIXED_LOT
                    entry_i = j
                continue

            hit_sl = (side == "BUY" and low <= sl) or (side == "SELL" and high >= sl)
            hit_tp = (side == "BUY" and high >= tp) or (side == "SELL" and low <= tp)
            timeout = (j - entry_i) >= MAX_HOLD_BARS

            if hit_sl or hit_tp or timeout:
                exit_price = sl if hit_sl else (tp if hit_tp else price)
                gross = (exit_price - entry) if side == "BUY" else (entry - exit_price)
                cost = (SPREAD_POINTS + COMMISSION_PER_LOT / CONTRACT_SIZE) * lots * CONTRACT_SIZE
                pnl = gross * lots * CONTRACT_SIZE - cost
                balance += pnl
                trades.append({
                    "event_time": str(t0),
                    "event": ev["event"],
                    "side": side,
                    "entry": entry,
                    "exit": exit_price,
                    "lots": lots,
                    "profit": pnl,
                    "R": pnl / (range_size * lots * CONTRACT_SIZE),
                    "balance": balance,
                    "reason": "SL" if hit_sl else ("TP" if hit_tp else "TIME"),
                })
                break

    os.makedirs(LOG_DIR, exist_ok=True)
    tdf = pd.DataFrame(trades)
    tdf.to_csv(os.path.join(LOG_DIR, "events_trades.csv"), index=False)

    n = len(trades)
    if n == 0:
        print("Сделок не было.")
        return

    def pf_of(g):
        gw = g.loc[g["profit"] > 0, "profit"].sum()
        gl = -g.loc[g["profit"] <= 0, "profit"].sum()
        return gw / gl if gl > 0 else float("inf")

    # --- Каждая новость отдельно ---
    if PRINT_EACH:
        print("\nКаждая новость:")
        print(f"  {'дата UTC':<22} {'тип':<13} {'side':<5} {'reason':<5} {'profit':>10} {'R':>7}")
        print("  " + "-" * 68)
        for _, r in tdf.iterrows():
            dt = str(r["event_time"])[:19]
            print(f"  {dt:<22} {r['event']:<13} {r['side']:<5} {r['reason']:<5} "
                  f"{r['profit']:>+10.2f} {r['R']:>+7.2f}")

    # --- По типам ---
    print("\nПо типам событий (R = прибыль / риск сделки):")
    for ev_name, g in tdf.groupby("event"):
        print(f"  {ev_name:13s} n={len(g):3d}  WR={(g['profit'] > 0).mean():.1%}  "
              f"PF={pf_of(g):.2f}  PnL=${g['profit'].sum():+.0f}  sumR={g['R'].sum():+.1f}")

    # --- По направлению ---
    print("\nПо направлению:")
    for (ev_name, sd), g in tdf.groupby(["event", "side"]):
        print(f"  {ev_name:13s} {sd:<5} n={len(g):3d}  WR={(g['profit'] > 0).mean():.0%}  "
              f"PnL=${g['profit'].sum():+.0f}  sumR={g['R'].sum():+.1f}")

    # --- По событиям и годам ---
    tdf["year"] = pd.to_datetime(tdf["event_time"], utc=True).dt.year
    years = sorted(tdf["year"].unique())
    W = 17

    def cell(g):
        if len(g) == 0:
            return "—"
        return f"{g['profit'].sum():+.0f} ({int((g['profit'] > 0).sum())}/{len(g)})"

    def cell_r(g):
        if len(g) == 0:
            return "—"
        return f"{g['R'].sum():+.1f}R ({len(g)})"

    order = tdf.groupby("event")["profit"].sum().sort_values(ascending=False).index

    print("\nПрибыль по событиям и годам: прибыль (побед/сделок)")
    head = f"  {'Событие':<14}" + "".join(f"|{y:^{W}}" for y in years) + f"|{'Сумма':>{W}}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for e in order:
        ge = tdf[tdf["event"] == e]
        row = f"  {e:<14}"
        for y in years:
            row += f"|{cell(ge[ge['year'] == y]):>{W}}"
        row += f"|{cell(ge):>{W}}"
        print(row)
    print("  " + "-" * (len(head) - 2))
    row = f"  {'Итого':<14}"
    for y in years:
        row += f"|{cell(tdf[tdf['year'] == y]):>{W}}"
    row += f"|{cell(tdf):>{W}}"
    print(row)

    print("\nR по событиям и годам: сумма R (сделок)")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for e in order:
        ge = tdf[tdf["event"] == e]
        row = f"  {e:<14}"
        for y in years:
            row += f"|{cell_r(ge[ge['year'] == y]):>{W}}"
        row += f"|{cell_r(ge):>{W}}"
        print(row)
    print("  " + "-" * (len(head) - 2))
    row = f"  {'Итого':<14}"
    for y in years:
        row += f"|{cell_r(tdf[tdf['year'] == y]):>{W}}"
    row += f"|{cell_r(tdf):>{W}}"
    print(row)

    # --- По месяцам ---
    tdf["month"] = (
        pd.to_datetime(tdf["event_time"], utc=True).dt.tz_localize(None).dt.to_period("M")
    )
    print("\nПо месяцам:")
    by_month = tdf.groupby("month").agg(
        n=("profit", "count"),
        pnl=("profit", "sum"),
        wr=("profit", lambda x: (x > 0).mean()),
    )
    for period, row in by_month.iterrows():
        print(f"  {period}  n={int(row['n']):2d}  WR={row['wr']:.0%}  PnL=${row['pnl']:+.0f}")

    # --- Итого ---
    print(f"\nСобытий в календаре: {len(events)}")
    print(f"Сделок:        {n}")
    print(f"Winrate:       {(tdf['profit'] > 0).mean():.1%}")
    print(f"Profit Factor: {pf_of(tdf):.2f}")
    print(f"Сумма R:       {tdf['R'].sum():+.1f}  (средняя {tdf['R'].mean():+.3f} на сделку)")
    print(f"Лот:           {FIXED_LOT} (фиксированный)")
    print(f"Финальный баланс: ${balance:,.2f}")
    print(f"Net P/L:       ${balance - initial_balance:,.2f}")
    print(f"Файл: logs/events_trades.csv")


if __name__ == "__main__":
    print("=" * 60)
    names = ",".join(sorted(USE_EVENTS)) if USE_EVENTS else "ALL"
    print(f"Event breakout v5 | {names} | лот {FIXED_LOT} | XAUUSD M15")
    print("=" * 60)

    events = load_events(EVENTS_PATH)
    print(f"Загружено событий: {len(events)}")

    os.environ["RESAMPLE_TF"] = "15min"
    df = get_historical_data(bars=9999999)
    df = calculate_indicators(df)

    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2026-07-01", tz="UTC")
    df = df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)

    print(f"Баров: {len(df)}")
    print()
    run_events(df, events)