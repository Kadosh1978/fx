# strategy_events_stress.py
# Стресс-тест издержек: прогоняет strategy_events.py с добавочными издержками
# (проскальзывание/спред) на каждую сделку. Кладётся рядом со strategy_events.py.
# Запуск:  python strategy_events_stress.py
# Логи основного скрипта (logs/events_trades.csv) НЕ перезаписываются.

import os
os.environ["RESAMPLE_TF"] = "15min"

import io
import contextlib
import pandas as pd

import strategy_events as se
from main import (
    SPREAD_POINTS,
    COMMISSION_PER_LOT,
    CONTRACT_SIZE,
    get_historical_data,
    calculate_indicators,
)

# доп. издержки на сделку, в долларах на унцию (сверх текущих спреда и комиссии)
EXTRA_SLIP = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
SETS = {
    "FOMC": {"FOMC"},
    "RETAIL": {"RETAIL"},
    "FOMC+RETAIL": {"FOMC", "RETAIL"},
}
STRESS_DIR = "logs_stress"


def pf_of(g):
    gw = g.loc[g["profit"] > 0, "profit"].sum()
    gl = -g.loc[g["profit"] <= 0, "profit"].sum()
    return gw / gl if gl > 0 else float("inf")


def run_once(df, events, extra):
    se.SPREAD_POINTS = SPREAD_POINTS + extra
    se.LOG_DIR = STRESS_DIR
    with contextlib.redirect_stdout(io.StringIO()):
        se.run_events(df, events)
    return pd.read_csv(os.path.join(STRESS_DIR, "events_trades.csv"))


if __name__ == "__main__":
    print("=" * 66)
    print("Стресс-тест издержек | XAUUSD M15 | лот", se.FIXED_LOT)
    print("=" * 66)
    print(f"Текущие издержки: спред {SPREAD_POINTS} (в цене), комиссия {COMMISSION_PER_LOT}/лот, "
          f"контракт {CONTRACT_SIZE}")
    print(f"Стоимость 1 $/oz на сделку при лоте {se.FIXED_LOT}: "
          f"${1.0 * se.FIXED_LOT * CONTRACT_SIZE:.0f}")

    df = get_historical_data(bars=9999999)
    df = calculate_indicators(df)
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2026-07-01", tz="UTC")
    df = df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)
    print(f"Баров: {len(df)}")

    for name, evset in SETS.items():
        se.USE_EVENTS = evset
        events = se.load_events(se.EVENTS_PATH)

        rows = []
        for extra in EXTRA_SLIP:
            t = run_once(df, events, extra)
            rows.append((extra, len(t), (t["profit"] > 0).mean(), pf_of(t),
                         t["profit"].sum(), t["R"].sum(), t["R"].mean()))

        print(f"\n{name}  (событий: {len(events)})")
        print(f"  {'доп.$/oz':>9} | {'n':>3} | {'WR':>5} | {'PF':>5} | {'PnL $':>8} | {'sumR':>7} | {'avgR':>6}")
        print("  " + "-" * 60)
        for extra, n, wr, pf, pnl, sr, ar in rows:
            print(f"  {extra:>9.1f} | {n:>3d} | {wr:>4.0%} | {pf:>5.2f} | {pnl:>+8.0f} | {sr:>+7.1f} | {ar:>+6.3f}")

        # sumR линейна по добавочным издержкам -> безубыточный уровень
        e1, sr0, sr1 = rows[1][0], rows[0][5], rows[1][5]
        k = (sr0 - sr1) / e1 if e1 else 0
        if k > 0 and sr0 > 0:
            print(f"  Безубыточные доп. издержки по sumR: ~{sr0 / k:.1f} $/oz на сделку")
        else:
            print("  Безубыточный уровень: не определён (нет прибыли в базовом прогоне)")

    print("\nЛоги стресс-прогона: logs_stress/ (основной logs/ не тронут)")
    