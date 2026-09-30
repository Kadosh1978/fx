# strategy_events_eurusd.py
# Event breakout для EURUSD M1 -> M15: отложенный пробой диапазона баров #2–#3
# Автономный скрипт (main.py не нужен). Фиксированный лот.
# Все события календаря + рейтинг: 2 лучших и 2 худших.
# Запуск:  python strategy_events_eurusd.py [путь_к_EURUSD_M1.csv]

import os
import sys
import pandas as pd

# ---------------- НАСТРОЙКИ ----------------
EVENTS_PATH = os.path.join("data", "macro_event_calendar_long_all.csv")
PRICE_PATHS = [os.path.join("data", "USDJPY_M1.csv"), "USDJPY_M1.csv"]
LOG_DIR = "logs"

TF = "15min"
DATE_FROM = "2022-01-01"
DATE_TO = "2026-07-01"

TZ_SHIFT_HOURS = 0       # время файла = UTC + TZ_SHIFT_HOURS (см. проверку пояса в выводе)
USE_EVENTS = None        # None = все события; или {"FOMC", "NFP"}
PRINT_EACH = False       # True = печатать каждую сделку

MIN_N_RANK = 20          # события с меньшим числом сделок в рейтинг не берём
RANK_BY = "avgR"         # по чему ранжировать: "avgR" (средняя R на сделку) или "sumR"

MIN_RANGE = 0.0005       # 5 пипсов: минимальный диапазон баров #2–#3
TP_MULT = 2.0
MAX_HOLD_BARS = 32
FIXED_LOT = 1.0          # 1 лот: 1 пипс = $10
CONTRACT_SIZE = 100000
SPREAD_PRICE = 0.00010   # 1 пипс, в ценовых единицах
COMMISSION_PER_LOT = 7.0 # $ за круг на 1 лот
INITIAL_BALANCE = 10000.0

# Время выхода (ET). Порядок = приоритет при совпадении времени.
# unemployment не включён: выходит вместе с NFP (дубль сделки).
EVENT_TIMES_ET = {
    "fomc": "14:00", "nfp": "08:30", "cpi": "08:30", "ppi": "08:30",
    "retail": "08:30", "gdp": "08:30", "pce": "08:30", "claims": "08:30",
    "indpro": "09:15", "ism_mfg": "10:00", "ism_services": "10:00",
    "umich": "10:00", "cb_confidence": "10:00",
}


# ---------------- ЗАГРУЗКА КОТИРОВОК ----------------
def _first_line(path):
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        return f.readline().strip()


def load_prices(path: str) -> pd.DataFrame:
    line = _first_line(path)
    sep = next((s for s in ("\t", ";", ",") if s in line), ",")
    has_header = not line.split(sep)[0].strip()[:1].isdigit()

    if has_header:
        raw = pd.read_csv(path, sep=sep)
        raw.columns = [str(c).strip().lower().strip("<>") for c in raw.columns]
    else:
        raw = pd.read_csv(path, sep=sep, header=None)
        n = raw.shape[1]
        if n >= 7:
            names = ["date", "time", "open", "high", "low", "close", "vol"] + [f"x{i}" for i in range(n - 7)]
        elif n == 6:
            names = ["time", "open", "high", "low", "close", "vol"]
        else:
            names = ["time", "open", "high", "low", "close"]
        raw.columns = names

    if "date" in raw.columns and "time" in raw.columns:
        ts = raw["date"].astype(str) + " " + raw["time"].astype(str)
    else:
        tcol = next((c for c in ("datetime", "timestamp", "time", "date", "gmt time", "local time")
                     if c in raw.columns), None)
        if tcol is None:
            raise ValueError(f"Не нашёл колонку времени. Колонки: {list(raw.columns)}")
        ts = raw[tcol]

    out = pd.DataFrame({"time": pd.to_datetime(ts, utc=True, errors="coerce")})
    for c in ("open", "high", "low", "close"):
        if c not in raw.columns:
            raise ValueError(f"Не нашёл колонку {c}. Колонки: {list(raw.columns)}")
        out[c] = pd.to_numeric(raw[c], errors="coerce")
    out = out.dropna().drop_duplicates("time").sort_values("time")
    out["time"] = out["time"] - pd.Timedelta(hours=TZ_SHIFT_HOURS)

    m15 = (
        out.set_index("time")
        .resample(TF)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    return m15


# ---------------- КАЛЕНДАРЬ ----------------
def load_events(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["event_code"] = raw["event_code"].str.lower().str.strip()
    raw = raw[raw["event_code"].isin(EVENT_TIMES_ET)].copy()
    if USE_EVENTS:
        raw = raw[raw["event_code"].str.upper().isin(USE_EVENTS)].copy()

    def to_utc(row):
        d = str(row["event_date"])[:10]
        t = EVENT_TIMES_ET[row["event_code"]]
        return pd.Timestamp(f"{d} {t}:00", tz="America/New_York").tz_convert("UTC")

    raw["datetime_utc"] = raw.apply(to_utc, axis=1)
    raw["event"] = raw["event_code"].str.upper()
    prio = list(EVENT_TIMES_ET)
    raw["prio"] = raw["event_code"].map(prio.index)
    out = (
        raw.sort_values(["datetime_utc", "prio"])
        .drop_duplicates("datetime_utc", keep="first")[["datetime_utc", "event"]]
    )
    out = out[(out["datetime_utc"] >= DATE_FROM) & (out["datetime_utc"] < DATE_TO)]
    return out.reset_index(drop=True)


# ---------------- ПРОВЕРКА ЧАСОВОГО ПОЯСА ----------------
def tz_check(df: pd.DataFrame, events: pd.DataFrame):
    """Средний размах свечи в момент новости / средний размах свечи вообще.
    На верном часовом поясе пик приходится на сдвиг 0."""
    idx_time = pd.DatetimeIndex(df["time"])
    rng = (df["high"] - df["low"]).values
    base = rng.mean()
    big = events[events["event"].isin(["NFP", "FOMC", "CPI"])]
    if len(big) == 0:
        big = events
    print("\nПроверка часового пояса (размах свечи в момент новости / средний):")
    best = None
    for h in range(-3, 4):
        target = pd.DatetimeIndex(big["datetime_utc"] + pd.Timedelta(hours=h))
        ix = idx_time.get_indexer(target, method="bfill")
        ix = ix[ix >= 0]
        if len(ix) == 0:
            continue
        ratio = rng[ix].mean() / base
        print(f"  сдвиг {h:+d}ч: x{ratio:.2f}")
        if best is None or ratio > best[1]:
            best = (h, ratio)
    if best:
        if best[0] == 0:
            print("  -> пик на 0: часовой пояс файла = UTC, всё верно")
        else:
            print(f"  -> ВНИМАНИЕ: пик на {best[0]:+d}ч. Поставь TZ_SHIFT_HOURS = {-best[0]} и перезапусти")


# ---------------- БЭКТЕСТ ----------------
def run_events(df: pd.DataFrame, events: pd.DataFrame):
    df = df.reset_index(drop=True)
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
        entry_i = None
        search_from = i3 + 1

        # окно: до 32 баров на вход + до 32 баров на выход
        for j in range(search_from, min(search_from + 2 * MAX_HOLD_BARS, len(df))):
            row = df.iloc[j]
            high, low, price = float(row["high"]), float(row["low"]), float(row["close"])

            if side is None:
                if j >= search_from + MAX_HOLD_BARS:
                    break  # вход не случился за 32 бара
                if high > rh:
                    side, entry, sl = "BUY", rh, rl
                    tp = entry + range_size * TP_MULT
                    entry_i = j
                elif low < rl:
                    side, entry, sl = "SELL", rl, rh
                    tp = entry - range_size * TP_MULT
                    entry_i = j
                continue

            hit_sl = (side == "BUY" and low <= sl) or (side == "SELL" and high >= sl)
            hit_tp = (side == "BUY" and high >= tp) or (side == "SELL" and low <= tp)
            timeout = (j - entry_i) >= MAX_HOLD_BARS

            if hit_sl or hit_tp or timeout:
                exit_price = sl if hit_sl else (tp if hit_tp else price)
                gross = (exit_price - entry) if side == "BUY" else (entry - exit_price)
                units = FIXED_LOT * CONTRACT_SIZE
                cost = SPREAD_PRICE * units + COMMISSION_PER_LOT * FIXED_LOT
                pnl = gross * units - cost
                risk = range_size * units
                trades.append({
                    "event_time": str(t0),
                    "event": ev["event"],
                    "side": side,
                    "range_pips": round(range_size / 0.0001, 1),
                    "entry": entry,
                    "exit": exit_price,
                    "profit": pnl,
                    "R": pnl / risk,
                    "reason": "SL" if hit_sl else ("TP" if hit_tp else "TIME"),
                })
                break

    os.makedirs(LOG_DIR, exist_ok=True)
    tdf = pd.DataFrame(trades)
    tdf.to_csv(os.path.join(LOG_DIR, "events_trades_eurusd.csv"), index=False)

    n = len(tdf)
    if n == 0:
        print("Сделок не было.")
        return

    def pf_of(g):
        gw = g.loc[g["profit"] > 0, "profit"].sum()
        gl = -g.loc[g["profit"] <= 0, "profit"].sum()
        return gw / gl if gl > 0 else float("inf")

    tdf["year"] = pd.to_datetime(tdf["event_time"], utc=True).dt.year
    years = sorted(tdf["year"].unique())

    if PRINT_EACH:
        print("\nКаждая новость:")
        print(f"  {'дата UTC':<22} {'тип':<13} {'side':<5} {'pips':>6} {'reason':<5} {'profit':>10} {'R':>7}")
        print("  " + "-" * 74)
        for _, r in tdf.iterrows():
            print(f"  {str(r['event_time'])[:19]:<22} {r['event']:<13} {r['side']:<5} "
                  f"{r['range_pips']:>6.1f} {r['reason']:<5} {r['profit']:>+10.2f} {r['R']:>+7.2f}")

    # --- Рейтинг событий ---
    stats = []
    for e, g in tdf.groupby("event"):
        yr = g.groupby("year")["R"].sum()
        stats.append({
            "event": e, "n": len(g),
            "wr": (g["profit"] > 0).mean(),
            "pf": pf_of(g),
            "avgR": g["R"].mean(),
            "sumR": g["R"].sum(),
            "pnl": g["profit"].sum(),
            "yrs_plus": int((yr > 0).sum()),
            "yrs": len(yr),
        })
    st = pd.DataFrame(stats)
    ranked = st[st["n"] >= MIN_N_RANK].sort_values(RANK_BY, ascending=False).reset_index(drop=True)
    small = st[st["n"] < MIN_N_RANK]

    print(f"\nРейтинг событий по {RANK_BY} (только n >= {MIN_N_RANK}):")
    print(f"  {'#':>2} {'событие':<14}{'n':>4} {'WR':>5} {'PF':>5} {'avgR':>7} {'sumR':>7} {'PnL $':>8}  лет в плюсе")
    print("  " + "-" * 72)
    for i, r in ranked.iterrows():
        print(f"  {i+1:>2} {r['event']:<14}{int(r['n']):>4} {r['wr']:>4.0%} {r['pf']:>5.2f} "
              f"{r['avgR']:>+7.3f} {r['sumR']:>+7.1f} {r['pnl']:>+8.0f}  {r['yrs_plus']}/{r['yrs']}")
    if len(small):
        names = ", ".join(f"{r['event']}(n={int(r['n'])})" for _, r in small.iterrows())
        print(f"  Не вошли в рейтинг (мало сделок): {names}")

    if len(ranked) >= 4:
        print("\nЛУЧШИЕ 2:")
        for _, r in ranked.head(2).iterrows():
            print(f"  {r['event']:<14} avgR={r['avgR']:+.3f}  sumR={r['sumR']:+.1f}  "
                  f"PF={r['pf']:.2f}  лет в плюсе {r['yrs_plus']}/{r['yrs']}")
        print("ХУДШИЕ 2:")
        for _, r in ranked.tail(2).iloc[::-1].iterrows():
            print(f"  {r['event']:<14} avgR={r['avgR']:+.3f}  sumR={r['sumR']:+.1f}  "
                  f"PF={r['pf']:.2f}  лет в плюсе {r['yrs_plus']}/{r['yrs']}")

    # --- По направлению ---
    print("\nПо направлению:")
    for (e, sd), g in tdf.groupby(["event", "side"]):
        print(f"  {e:13s} {sd:<5} n={len(g):3d}  WR={(g['profit'] > 0).mean():.0%}  "
              f"PnL=${g['profit'].sum():+.0f}  sumR={g['R'].sum():+.1f}")

    # --- Таблицы по событиям и годам ---
    W = 17
    order = list(ranked["event"]) + list(small["event"])

    def cell(g):
        if len(g) == 0:
            return "—"
        return f"{g['profit'].sum():+.0f} ({int((g['profit'] > 0).sum())}/{len(g)})"

    def cell_r(g):
        if len(g) == 0:
            return "—"
        return f"{g['R'].sum():+.1f}R ({len(g)})"

    head = f"  {'Событие':<14}" + "".join(f"|{y:^{W}}" for y in years) + f"|{'Сумма':>{W}}"

    def year_table(title, cf):
        print(f"\n{title}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for e in order:
            ge = tdf[tdf["event"] == e]
            row = f"  {e:<14}"
            for y in years:
                row += f"|{cf(ge[ge['year'] == y]):>{W}}"
            row += f"|{cf(ge):>{W}}"
            print(row)
        print("  " + "-" * (len(head) - 2))
        row = f"  {'Итого':<14}"
        for y in years:
            row += f"|{cf(tdf[tdf['year'] == y]):>{W}}"
        row += f"|{cf(tdf):>{W}}"
        print(row)

    year_table("Прибыль по событиям и годам: прибыль (побед/сделок)", cell)
    year_table("R по событиям и годам: сумма R (сделок)", cell_r)

    # --- По месяцам ---
    tdf["month"] = pd.to_datetime(tdf["event_time"], utc=True).dt.tz_localize(None).dt.to_period("M")
    print("\nПо месяцам:")
    by_month = tdf.groupby("month").agg(
        n=("profit", "count"), pnl=("profit", "sum"),
        wr=("profit", lambda x: (x > 0).mean()),
    )
    for period, r in by_month.iterrows():
        print(f"  {period}  n={int(r['n']):2d}  WR={r['wr']:.0%}  PnL=${r['pnl']:+.0f}")

    # --- Итого ---
    net = tdf["profit"].sum()
    print(f"\nСобытий в календаре: {len(events)}")
    print(f"Сделок:        {n}")
    print(f"Winrate:       {(tdf['profit'] > 0).mean():.1%}")
    print(f"Profit Factor: {pf_of(tdf):.2f}")
    print(f"Сумма R:       {tdf['R'].sum():+.1f}  (средняя {tdf['R'].mean():+.3f} на сделку)")
    print(f"Лот:           {FIXED_LOT} (фиксированный)")
    print(f"Финальный баланс: ${INITIAL_BALANCE + net:,.2f}")
    print(f"Net P/L:       ${net:,.2f}")
    print("Файл: logs/events_trades_eurusd.csv")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else next((p for p in PRICE_PATHS if os.path.exists(p)), None)
    if path is None:
        sys.exit(f"Не нашёл USDJPY_M1.csv. Искал: {PRICE_PATHS}. Передай путь аргументом.")

    names = ",".join(sorted(USE_EVENTS)) if USE_EVENTS else "ALL"
    print("=" * 60)
    print(f"Event breakout | {names} | EURUSD {TF} | лот {FIXED_LOT}")
    print("=" * 60)
    print(f"Файл котировок: {path}")

    df = load_prices(path)
    print(f"Баров {TF}: {len(df)}, {df['time'].iloc[0]} .. {df['time'].iloc[-1]}")

    events = load_events(EVENTS_PATH)
    print(f"Загружено событий: {len(events)}")

    tz_check(df, events)

    start = pd.Timestamp(DATE_FROM, tz="UTC")
    end = pd.Timestamp(DATE_TO, tz="UTC")
    df = df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)
    print(f"\nБаров в периоде: {len(df)}")

    run_events(df, events)