# strategy_fomc_multi.py
# FOMC-пробой по нескольким инструментам, результат по месяцам + сумма по всем парам.
# Логика входа/выхода та же, что в strategy_events*.py (диапазон баров #2–#3, TP=2R,
# вход до 32 баров, потом до 32 баров на выход).
# Кладётся рядом со strategy_events_eurusd.py (берёт оттуда загрузку данных и календаря).
# Запуск:  python strategy_fomc_multi.py [папка_с_csv]

import os
import io
import sys
import contextlib
import pandas as pd

import strategy_events_eurusd as se

# ---------------- НАСТРОЙКИ ----------------
SIZING = "lot"        # "lot"  = фиксированный лот из INSTRUMENTS[...]["lot"]
                      # "risk" = лот подбирается так, чтобы риск сделки = RISK_USD
RISK_USD = 10.0       # используется только при SIZING = "risk"
PRICE_DIRS = ["data", "."]
OUT_DIR = "logs_fomc_multi"

# spread — в ценовых единицах инструмента; commission — $ за круг на 1 лот;
# min_range — минимальный диапазон баров #2–#3 (в цене); jpy=True: прибыль считается в иенах
# и пересчитывается в доллары. Спреды/комиссии ниже приблизительные, поправь под брокера.
INSTRUMENTS = {
    "USDJPY": {"file": "USDJPY_M1.csv", "contract": 100000, "spread": 0.010,   "commission": 7.0,
               "min_range": 0.05,   "lot": 0.01, "jpy": True,  "tz_shift": 0},
    "GBPUSD": {"file": "GBPUSD_M1.csv", "contract": 100000, "spread": 0.00012, "commission": 7.0,
               "min_range": 0.0005, "lot": 0.01, "jpy": False, "tz_shift": 0},
    "EURUSD": {"file": "EURUSD_M1.csv", "contract": 100000, "spread": 0.00010, "commission": 7.0,
               "min_range": 0.0005, "lot": 0.01, "jpy": False, "tz_shift": 0},
    "XAUUSD": {"file": "XAUUSD_M1.csv", "contract": 100,    "spread": 0.45,    "commission": 7.0,
               "min_range": 1.5,    "lot": 0.01, "jpy": False, "tz_shift": 0},
}


# ---------------- БЭКТЕСТ ОДНОГО ИНСТРУМЕНТА ----------------
def backtest(pair, df, events, cfg):
    tv = pd.DatetimeIndex(df["time"])
    H = df["high"].to_numpy(float)
    L = df["low"].to_numpy(float)
    C = df["close"].to_numpy(float)
    n = len(df)
    contract = cfg["contract"]
    hold = se.MAX_HOLD_BARS
    trades = []

    for _, ev in events.iterrows():
        t0 = ev["datetime_utc"]
        if t0 < tv[0] or t0 > tv[-1]:
            continue
        idx = tv.searchsorted(t0)
        if idx >= n - 6:
            continue
        if tv[idx] - t0 > pd.Timedelta(hours=1):   # дыра в данных
            continue

        i2, i3 = idx + 1, idx + 2
        rh = max(H[i2], H[i3])
        rl = min(L[i2], L[i3])
        rng = rh - rl
        if rng < cfg["min_range"]:
            continue

        side = None
        entry = sl = tp = None
        entry_i = None
        start = i3 + 1

        for j in range(start, min(start + 2 * hold, n)):
            if side is None:
                if j >= start + hold:
                    break                       # вход не случился за 32 бара
                if H[j] > rh:
                    side, entry, sl = "BUY", rh, rl
                    tp = entry + rng * se.TP_MULT
                    entry_i = j
                elif L[j] < rl:
                    side, entry, sl = "SELL", rl, rh
                    tp = entry - rng * se.TP_MULT
                    entry_i = j
                continue

            hit_sl = (side == "BUY" and L[j] <= sl) or (side == "SELL" and H[j] >= sl)
            hit_tp = (side == "BUY" and H[j] >= tp) or (side == "SELL" and L[j] <= tp)
            timeout = (j - entry_i) >= hold

            if hit_sl or hit_tp or timeout:
                exit_price = sl if hit_sl else (tp if hit_tp else C[j])
                gross = (exit_price - entry) if side == "BUY" else (entry - exit_price)
                conv = (1.0 / entry) if cfg["jpy"] else 1.0       # котируемая валюта -> USD
                lots = cfg["lot"] if SIZING == "lot" else RISK_USD / (rng * contract * conv)
                units = lots * contract
                pnl = (gross * units - cfg["spread"] * units) * conv - cfg["commission"] * lots
                risk = rng * units * conv
                trades.append({
                    "pair": pair,
                    "event_time": str(t0),
                    "month": t0.strftime("%Y-%m"),
                    "side": side,
                    "entry": entry,
                    "exit": exit_price,
                    "lots": lots,
                    "profit": pnl,
                    "R": pnl / risk,
                    "reason": "SL" if hit_sl else ("TP" if hit_tp else "TIME"),
                })
                break
    return trades


# ---------------- ВЫВОД ----------------
def pf_of(g):
    gw = g.loc[g["profit"] > 0, "profit"].sum()
    gl = -g.loc[g["profit"] <= 0, "profit"].sum()
    return gw / gl if gl > 0 else float("inf")


def print_pivot(title, pv, fmt):
    W = 10
    cols = list(pv.columns)
    row_sum = pv.sum(axis=1, skipna=True)
    head = f"  {'Месяц':<9}" + "".join(f"|{c:>{W}}" for c in cols) + f"|{'Сумма':>{W + 1}}"
    print(f"\n{title}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for m, r in pv.iterrows():
        cells = "".join(f"|{('—' if pd.isna(r[c]) else format(r[c], fmt)):>{W}}" for c in cols)
        print(f"  {m:<9}{cells}|{format(row_sum[m], fmt):>{W + 1}}")
    print("  " + "-" * (len(head) - 2))
    col_sum = pv.sum(axis=0, skipna=True)
    cells = "".join(f"|{format(col_sum[c], fmt):>{W}}" for c in cols)
    print(f"  {'Итого':<9}{cells}|{format(col_sum.sum(), fmt):>{W + 1}}")
    pos = int((row_sum > 0).sum())
    print(f"  Месяцев в плюсе по сумме всех пар: {pos} из {len(row_sum)}")


def find_file(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


if __name__ == "__main__":
    dirs = [sys.argv[1]] + PRICE_DIRS if len(sys.argv) > 1 else PRICE_DIRS

    se.USE_EVENTS = {"FOMC"}
    events_all = None

    print("=" * 72)
    print(f"FOMC-пробой | {', '.join(INSTRUMENTS)} | sizing={SIZING}"
          + (f" (риск ${RISK_USD}/сделку)" if SIZING == "risk" else ""))
    print("=" * 72)

    all_trades = []
    used = []
    for pair, cfg in INSTRUMENTS.items():
        path = find_file(cfg["file"], dirs)
        if path is None:
            print(f"\n[{pair}] файл {cfg['file']} не найден в {dirs} — пропускаю")
            continue

        se.TZ_SHIFT_HOURS = cfg["tz_shift"]
        df = se.load_prices(path)
        events = se.load_events(se.EVENTS_PATH)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            se.tz_check(df, events)
        verdict = [l.strip() for l in buf.getvalue().splitlines() if "->" in l]
        print(f"\n[{pair}] {path}")
        print(f"  баров {se.TF}: {len(df)}, {df['time'].iloc[0]:%Y-%m-%d} .. {df['time'].iloc[-1]:%Y-%m-%d}")
        print(f"  часовой пояс: {verdict[0] if verdict else 'проверка не удалась'}")
        lot_txt = f"лот {cfg['lot']}" if SIZING == "lot" else f"риск ${RISK_USD}"
        print(f"  {lot_txt}, спред {cfg['spread']}, комиссия ${cfg['commission']}/лот")

        start = pd.Timestamp(se.DATE_FROM, tz="UTC")
        end = pd.Timestamp(se.DATE_TO, tz="UTC")
        df = df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)

        tr = backtest(pair, df, events, cfg)
        print(f"  событий FOMC: {len(events)}, сделок: {len(tr)}")
        all_trades += tr
        used.append(pair)

    if not all_trades:
        sys.exit("\nСделок нет — проверь файлы и пути.")

    tdf = pd.DataFrame(all_trades)
    os.makedirs(OUT_DIR, exist_ok=True)
    tdf.to_csv(os.path.join(OUT_DIR, "fomc_trades_all.csv"), index=False)

    # --- Сводка по парам ---
    print("\nСводка по парам:")
    print(f"  {'пара':<8}{'n':>4} {'WR':>5} {'PF':>5} {'E $/сд':>8} {'PnL $':>9} {'sumR':>7} {'avgR':>7}")
    print("  " + "-" * 58)
    for p in used:
        g = tdf[tdf["pair"] == p]
        if len(g) == 0:
            continue
        print(f"  {p:<8}{len(g):>4} {(g['profit'] > 0).mean():>5.0%} {pf_of(g):>5.2f} "
              f"{g['profit'].mean():>+8.3f} {g['profit'].sum():>+9.2f} {g['R'].sum():>+7.1f} {g['R'].mean():>+7.3f}")
    print("  " + "-" * 58)
    print(f"  {'ВСЕ':<8}{len(tdf):>4} {(tdf['profit'] > 0).mean():>5.0%} {pf_of(tdf):>5.2f} "
          f"{tdf['profit'].mean():>+8.3f} {tdf['profit'].sum():>+9.2f} {tdf['R'].sum():>+7.1f} {tdf['R'].mean():>+7.3f}")

    # --- Таблицы по месяцам ---
    order = [p for p in used if p in set(tdf["pair"])]
    pv_usd = tdf.pivot_table(index="month", columns="pair", values="profit", aggfunc="sum").reindex(columns=order)
    pv_r = tdf.pivot_table(index="month", columns="pair", values="R", aggfunc="sum").reindex(columns=order)
    print_pivot("Прибыль по месяцам, $ (строка = месяц FOMC, справа сумма по всем парам):", pv_usd, "+.2f")
    print_pivot("То же в R (выравнивает риск между инструментами):", pv_r, "+.2f")

    # --- По годам ---
    tdf["year"] = tdf["month"].str[:4]
    pv_y = tdf.pivot_table(index="year", columns="pair", values="profit", aggfunc="sum").reindex(columns=order)
    print_pivot("По годам, $:", pv_y, "+.2f")
    pv_yr = tdf.pivot_table(index="year", columns="pair", values="R", aggfunc="sum").reindex(columns=order)
    print_pivot("По годам, R:", pv_yr, "+.2f")

    print(f"\nЛог сделок: {OUT_DIR}/fomc_trades_all.csv")