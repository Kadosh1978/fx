# strategy_events_eurusd_monthly.py
# EURUSD, ВСЕ события календаря, лот 0.01.
# Матожидание (средний результат на сделку, $) по месяцам.
# Кладётся рядом со strategy_events_eurusd.py (использует его логику).
# Запуск:  python strategy_events_eurusd_monthly.py [путь_к_EURUSD_M1.csv]

import os
import io
import sys
import math
import contextlib
import pandas as pd

import strategy_events_eurusd as se

LOT = 0.01
OUT_DIR = "logs_monthly"


def stats(g):
    n = len(g)
    p = g["profit"]
    wins = p[p > 0]
    losses = p[p <= 0]
    e = p.mean()
    se_ = p.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "wr": (p > 0).mean(),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "e": e,
        "se": se_,
        "pnl": p.sum(),
        "avgR": g["R"].mean(),
    }


def line(label, s, w=8):
    return (f"  {label:<{w}} {s['n']:>4d} {s['wr']:>5.0%} {s['avg_win']:>+8.2f} {s['avg_loss']:>+9.2f} "
            f"{s['e']:>+8.3f} {s['se']:>6.3f} {s['pnl']:>+8.2f} {s['avgR']:>+7.3f}")


HEADER = (f"  {'':<8} {'n':>4} {'WR':>5} {'ср.вин':>8} {'ср.проигр':>9} "
          f"{'E $':>8} {'±SE':>6} {'PnL $':>8} {'avgR':>7}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else next((p for p in se.PRICE_PATHS if os.path.exists(p)), None)
    if path is None:
        sys.exit(f"Не нашёл EURUSD_M1.csv. Искал: {se.PRICE_PATHS}. Передай путь аргументом.")

    # настройки: все события, лот 0.01
    se.USE_EVENTS = None
    se.FIXED_LOT = LOT
    se.LOG_DIR = OUT_DIR
    se.PRINT_EACH = False

    print("=" * 72)
    print(f"EURUSD {se.TF} | ВСЕ события | лот {LOT} | матожидание по месяцам")
    print("=" * 72)
    print(f"Файл котировок: {path}")

    df = se.load_prices(path)
    events = se.load_events(se.EVENTS_PATH)
    start = pd.Timestamp(se.DATE_FROM, tz="UTC")
    end = pd.Timestamp(se.DATE_TO, tz="UTC")
    df = df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)
    print(f"Баров в периоде: {len(df)}, событий: {len(events)}")

    with contextlib.redirect_stdout(io.StringIO()):
        se.run_events(df, events)
    tdf = pd.read_csv(os.path.join(OUT_DIR, "events_trades_eurusd.csv"))

    t = pd.to_datetime(tdf["event_time"], utc=True).dt.tz_localize(None)
    tdf["month"] = t.dt.to_period("M")
    tdf["cal_month"] = t.dt.month

    # --- По месяцам (хронологически) ---
    print("\nМатожидание по месяцам (E $ = средний результат на сделку при лоте 0.01):")
    print(HEADER)
    print("  " + "-" * 74)
    for m, g in tdf.groupby("month"):
        print(line(str(m), stats(g)))
    print("  " + "-" * 74)
    total = stats(tdf)
    print(line("ИТОГО", total))

    # --- По календарным месяцам (все годы вместе) ---
    names = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
    print("\nПо календарным месяцам (все годы вместе):")
    print(HEADER)
    print("  " + "-" * 74)
    for m, g in tdf.groupby("cal_month"):
        print(line(names[m - 1], stats(g)))

    # --- Итог ---
    lo, hi = total["e"] - 2 * total["se"], total["e"] + 2 * total["se"]
    pos_months = int((tdf.groupby("month")["profit"].mean() > 0).sum())
    all_months = tdf["month"].nunique()
    print(f"\nМатожидание на сделку: {total['e']:+.3f} $ (лот {LOT}), ±2SE: [{lo:+.3f} .. {hi:+.3f}]")
    print(f"Средняя R на сделку:   {total['avgR']:+.3f}")
    print(f"Месяцев с положительным матожиданием: {pos_months} из {all_months}")
    print(f"Сделок: {total['n']}, суммарно ${total['pnl']:+.2f}")
    print(f"Логи: {OUT_DIR}/events_trades_eurusd.csv (основной logs/ не тронут)")