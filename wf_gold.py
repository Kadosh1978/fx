# wf_gold.py
# Walk-forward (скользящее окно) обучение модели на XAUUSD.
#
# Как работает:
#   1. Индикаторы (8 штук) считаются на барах выбранного таймфрейма.
#   2. Метка (triple barrier): что сработает первым — тейк (2 ATR) или стоп (1.5 ATR)
#      в течение HORIZON баров. Вход по открытию следующего бара.
#      Две модели: одна для BUY, одна для SELL.
#   3. Окно: обучение на TRAIN_MONTHS, тест на следующих TEST_MONTHS, сдвиг на TEST_MONTHS.
#      Между обучением и тестом зазор HORIZON баров (purge), чтобы не было утечки.
#   4. Все внешние (out-of-sample) прогнозы склеиваются и прогоняются
#      одним непрерывным бэктестом с реальными издержками.
#
# Запуск (из папки проекта):
#   python wf_gold.py
#
# Настройки через переменные окружения (необязательно):
#   CSV_PATH      путь к CSV (по умолчанию data\XAUUSD_M1.csv)
#   TFS           таймфреймы через запятую, например "1h" или "1h,4h"  (по умолчанию 1h)
#   TRAIN_MONTHS  окно обучения, месяцев (по умолчанию 24)
#   TEST_MONTHS   окно теста и шаг сдвига, месяцев (по умолчанию 3)
#   HORIZON       сколько баров держим сделку максимум (по умолчанию 24)
#   EDGE_MARGIN   запас над безубыточной вероятностью (по умолчанию 0.03)
#   EDGE_OVER_BASE  вход только если вероятность выше базовой (средней по обучающему окну)
#                 хотя бы на эту величину (по умолчанию 0.03). Нужен, чтобы модель
#                 не зарабатывала на одном лишь росте золота, а показывала вклад индикаторов.
#   SEED          зерно перемешивания для SHUFFLE (1, 2, 3 ...) — чтобы прогнать серию
#   SHUFFLE=1     ПРОВЕРКА НА ЧЕСТНОСТЬ: перемешать метки в обучении.
#                 Результат обязан стать плохим (PF около 1 или ниже, AUC около 0.5).

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
except ImportError:
    sys.exit("Нужен xgboost: pip install xgboost")

from main import (
    _load_csv,
    SPREAD_POINTS,
    COMMISSION_PER_LOT,
    CONTRACT_SIZE,
    RISK_PER_TRADE,
    MIN_LOT,
    MAX_LOT,
    LOG_DIR,
)

# ---------------------------------------------------------------- настройки
CSV = os.getenv("CSV_PATH", os.path.join("data", "XAUUSD_M1.csv"))
TFS = [x.strip() for x in os.getenv("TFS", "1h").split(",") if x.strip()]
TRAIN_MONTHS = int(os.getenv("TRAIN_MONTHS", "24"))
TEST_MONTHS = int(os.getenv("TEST_MONTHS", "3"))
HORIZON = int(os.getenv("HORIZON", "24"))
EDGE_MARGIN = float(os.getenv("EDGE_MARGIN", "0.03"))
EDGE_OVER_BASE = float(os.getenv("EDGE_OVER_BASE", "0.03"))
SHUFFLE = os.getenv("SHUFFLE", "0") == "1"
SEED = int(os.getenv("SEED", "0"))  # зерно перемешивания меток (для серии проверок)

SL_ATR = 1.5
TP_ATR = 2.0
INITIAL_BALANCE = 10000.0
WARMUP = 120
MIN_TRAIN_SAMPLES = 1500

START = pd.Timestamp("2022-01-01", tz="UTC")
END = pd.Timestamp("2026-07-01", tz="UTC")

FEATURES = [
    "rsi", "ema_diff_atr", "macd_hist_atr", "atr_regime",
    "bb_pctb", "adx", "stoch_k", "hour_sin", "hour_cos",
]


# ---------------------------------------------------------------- индикаторы
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    c, h, l = df["close"], df["high"], df["low"]
    pc = c.shift()

    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["atr"] = atr

    # RSI(14), Wilder
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = up / dn.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # EMA20 - EMA50 в единицах ATR
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    df["ema_diff_atr"] = (ema20 - ema50) / atr

    # MACD гистограмма в единицах ATR
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist_atr"] = (macd - macd_sig) / atr

    # Режим волатильности
    df["atr_regime"] = atr / atr.rolling(100).mean()

    # Bollinger %B (20, 2)
    ma = c.rolling(20).mean()
    sd = c.rolling(20).std().replace(0, np.nan)
    df["bb_pctb"] = (c - (ma - 2 * sd)) / (4 * sd)

    # ADX(14)
    up_move = h.diff()
    dn_move = -l.diff()
    plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # Stochastic %K(14)
    ll = l.rolling(14).min()
    hh = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)

    # Час суток (UTC), циклически
    hour = pd.to_datetime(df["time"], utc=True).dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Отбрасываем прогрев, редкие дыры (деление на ноль) заполняем без заглядывания вперёд
    df = df.iloc[WARMUP:].reset_index(drop=True)
    df[FEATURES] = (
        df[FEATURES].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    )
    return df


# ---------------------------------------------------------------- метки
def make_labels(o, h, l, atr, horizon):
    """
    Для бара i: вход по open[i+1], SL=SL_ATR*atr[i], TP=TP_ATR*atr[i].
    1 = тейк сработал раньше стопа в течение horizon баров, иначе 0.
    Если стоп и тейк в одном баре — считаем, что стоп был первым (консервативно).
    """
    n = len(o)
    y_long = np.full(n, np.nan)
    y_short = np.full(n, np.nan)
    for i in range(0, n - horizon):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        e = o[i + 1]
        sl_l, tp_l = e - SL_ATR * a, e + TP_ATR * a
        sl_s, tp_s = e + SL_ATR * a, e - TP_ATR * a

        res = 0
        for k in range(i + 1, i + horizon + 1):
            if l[k] <= sl_l:
                break
            if h[k] >= tp_l:
                res = 1
                break
        y_long[i] = res

        res = 0
        for k in range(i + 1, i + horizon + 1):
            if h[k] >= sl_s:
                break
            if l[k] <= tp_s:
                res = 1
                break
        y_short[i] = res
    return y_long, y_short


def new_model():
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_lambda=5.0,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )


# ---------------------------------------------------------------- walk-forward
def walk_forward(df: pd.DataFrame):
    n = len(df)
    t = df["time"]
    o, h, l = df["open"].values, df["high"].values, df["low"].values
    atr = df["atr"].values
    X = df[FEATURES].values

    y_long, y_short = make_labels(o, h, l, atr, HORIZON)

    p_long = np.full(n, np.nan)
    p_short = np.full(n, np.nan)
    base_long = np.full(n, np.nan)
    base_short = np.full(n, np.nan)
    importance = np.zeros(len(FEATURES))
    n_models = 0
    rng = np.random.RandomState(SEED)

    test_start = t.iloc[0].normalize() + pd.DateOffset(months=TRAIN_MONTHS)
    last = t.iloc[-1]

    print(f"{'окно теста':<24}{'train':>7}{'база L':>8}{'база S':>8}{'AUC L':>8}{'AUC S':>8}")

    while test_start <= last:
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
        train_start = test_start - pd.DateOffset(months=TRAIN_MONTHS)

        i_tr0 = int(t.searchsorted(train_start))
        i_te0 = int(t.searchsorted(test_start))
        i_te1 = int(t.searchsorted(test_end))
        i_tr1 = i_te0 - HORIZON  # purge: метки не должны заходить в тестовый период

        label = f"{test_start:%Y-%m-%d}..{min(test_end, last):%Y-%m-%d}"
        if i_te1 <= i_te0:
            break

        tr = np.arange(i_tr0, max(i_tr0, i_tr1))
        tr = tr[np.isfinite(y_long[tr])]
        if len(tr) < MIN_TRAIN_SAMPLES:
            print(f"{label:<24} пропуск: мало данных для обучения ({len(tr)})")
            test_start = test_end
            continue

        yl_tr = y_long[tr].astype(int)
        ys_tr = y_short[tr].astype(int)
        if SHUFFLE:
            yl_tr = rng.permutation(yl_tr)
            ys_tr = rng.permutation(ys_tr)
        if len(np.unique(yl_tr)) < 2 or len(np.unique(ys_tr)) < 2:
            test_start = test_end
            continue

        m_long = new_model().fit(X[tr], yl_tr)
        m_short = new_model().fit(X[tr], ys_tr)

        te = np.arange(i_te0, i_te1)
        p_long[te] = m_long.predict_proba(X[te])[:, 1]
        p_short[te] = m_short.predict_proba(X[te])[:, 1]
        base_long[te] = np.mean(y_long[tr])    # базовая вероятность по обучающему окну
        base_short[te] = np.mean(y_short[tr])

        importance += m_long.feature_importances_ + m_short.feature_importances_
        n_models += 2

        def auc(y, p):
            ok = np.isfinite(y)
            if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
                return float("nan")
            return roc_auc_score(y[ok], p[ok])

        print(
            f"{label:<24}{len(tr):>7}"
            f"{np.nanmean(y_long[te]):>8.3f}{np.nanmean(y_short[te]):>8.3f}"
            f"{auc(y_long[te], p_long[te]):>8.3f}{auc(y_short[te], p_short[te]):>8.3f}"
        )
        test_start = test_end

    if n_models:
        importance /= n_models
    return p_long, p_short, base_long, base_short, importance


# ---------------------------------------------------------------- бэктест
def backtest(df: pd.DataFrame, p_long, p_short, base_long, base_short):
    n = len(df)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = df["atr"].values
    times = df["time"].values

    valid = np.where(np.isfinite(p_long))[0]
    if len(valid) == 0:
        return pd.DataFrame(), pd.DataFrame()
    i0 = int(valid[0])

    cost_price = SPREAD_POINTS + COMMISSION_PER_LOT / CONTRACT_SIZE
    balance = INITIAL_BALANCE
    pos = None
    pending = None
    trades, equity = [], []
    skipped_lot = 0

    for i in range(i0, n):
        # 1) сигнал прошлого бара исполняется по открытию этого бара
        if pending is not None and pos is None:
            side, a = pending
            pending = None
            e = o[i]
            sl_d, tp_d = SL_ATR * a, TP_ATR * a
            raw = balance * RISK_PER_TRADE / (sl_d * CONTRACT_SIZE)
            if raw >= MIN_LOT:
                lots = float(np.clip(round(raw, 2), MIN_LOT, MAX_LOT))
                if side == "BUY":
                    sl, tp = e - sl_d, e + tp_d
                else:
                    sl, tp = e + sl_d, e - tp_d
                pos = {"side": side, "entry": e, "lots": lots, "sl": sl, "tp": tp,
                       "bars": 0, "t_open": times[i]}
            else:
                skipped_lot += 1

        # 2) сопровождение позиции (включая бар входа)
        if pos is not None:
            pos["bars"] += 1
            if pos["side"] == "BUY":
                hit_sl = l[i] <= pos["sl"]
                hit_tp = h[i] >= pos["tp"]
            else:
                hit_sl = h[i] >= pos["sl"]
                hit_tp = l[i] <= pos["tp"]
            timeout = pos["bars"] >= HORIZON

            if hit_sl or hit_tp or timeout:
                if hit_sl:
                    # гэп через стоп — исполнение по open, а не по уровню
                    exit_price = min(pos["sl"], o[i]) if pos["side"] == "BUY" else max(pos["sl"], o[i])
                    reason = "SL"
                elif hit_tp:
                    exit_price, reason = pos["tp"], "TP"
                else:
                    exit_price, reason = c[i], "TIME"

                gross = (exit_price - pos["entry"]) if pos["side"] == "BUY" else (pos["entry"] - exit_price)
                pnl = (gross - cost_price) * pos["lots"] * CONTRACT_SIZE
                balance += pnl
                trades.append({
                    "time_open": pd.Timestamp(pos["t_open"]).isoformat(),
                    "time": pd.Timestamp(times[i]).isoformat(),
                    "side": pos["side"], "lots": pos["lots"],
                    "entry": pos["entry"], "exit": exit_price,
                    "profit": pnl, "balance": balance, "reason": reason,
                })
                pos = None

        # 3) новый сигнал на закрытии бара (вход на следующем open)
        if pos is None and i < n - 1 and np.isfinite(p_long[i]) and atr[i] > 0:
            sl_d = SL_ATR * atr[i]
            cost_r = cost_price / sl_d
            thr = (1 + cost_r) / (1 + TP_ATR / SL_ATR) + EDGE_MARGIN  # безубыточная p + запас
            thr_l = max(thr, base_long[i] + EDGE_OVER_BASE)
            thr_s = max(thr, base_short[i] + EDGE_OVER_BASE)
            ex_l = p_long[i] - thr_l
            ex_s = p_short[i] - thr_s
            if max(ex_l, ex_s) > 0:
                pending = ("BUY" if ex_l >= ex_s else "SELL", atr[i])

        equity.append({"time": pd.Timestamp(times[i]).isoformat(), "equity": balance})

    if skipped_lot:
        print(f"Пропущено сигналов из-за лота < {MIN_LOT}: {skipped_lot}")
    return pd.DataFrame(trades), pd.DataFrame(equity)


def summarize(name, trades: pd.DataFrame, equity: pd.DataFrame):
    print(f"\n=== {name}: итоги (только out-of-sample) ===")
    if trades.empty:
        print("Сделок не было. Модель нигде не дала вероятность выше порога.")
        return
    n = len(trades)
    wins = (trades["profit"] > 0).sum()
    gw = trades.loc[trades["profit"] > 0, "profit"].sum()
    gl = -trades.loc[trades["profit"] <= 0, "profit"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    eq = equity["equity"].values
    dd = (np.maximum.accumulate(eq) - eq).max()
    print(f"Сделок:          {n}")
    print(f"Winrate:         {wins / n:.1%}")
    print(f"Profit Factor:   {pf:.2f}")
    print(f"Net P/L:         ${trades['profit'].sum():,.2f}")
    print(f"Финал. баланс:   ${eq[-1]:,.2f}")
    print(f"Макс. просадка:  ${dd:,.2f}")
    print("Причины выхода:  " + ", ".join(
        f"{k}={v}" for k, v in trades["reason"].value_counts().items()))
    print("BUY/SELL:        " + ", ".join(
        f"{k}={v}" for k, v in trades["side"].value_counts().items()))


def monthly_table(results: dict) -> pd.DataFrame:
    cols = {}
    for tf, trades in results.items():
        if trades.empty:
            cols[tf.upper()] = pd.Series(dtype=float)
            continue
        m = pd.to_datetime(trades["time"], utc=True).dt.strftime("%Y-%m")
        cols[tf.upper()] = trades.groupby(m)["profit"].sum()
    table = pd.DataFrame(cols).fillna(0.0).sort_index()
    table["Сумма"] = table.sum(axis=1)
    table.loc["Итого"] = table.sum(axis=0)
    table.index.name = None
    return table


# ---------------------------------------------------------------- main
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print("=" * 70)
    print("Walk-forward XAUUSD | train=%dм test=%dм horizon=%d | margin=%.2f | над базой=%.2f%s"
          % (TRAIN_MONTHS, TEST_MONTHS, HORIZON, EDGE_MARGIN, EDGE_OVER_BASE,
             " | SHUFFLE (проверка)" if SHUFFLE else ""))
    print("Индикаторы:", ", ".join(FEATURES))
    print("=" * 70)

    results = {}
    for tf in TFS:
        print(f"\n########## Таймфрейм {tf} ##########")
        df = _load_csv(CSV, resample=tf)
        df = df[(df["time"] >= START) & (df["time"] < END)].reset_index(drop=True)
        df = add_features(df)
        print(f"Баров: {len(df)} | {df['time'].iloc[0]} .. {df['time'].iloc[-1]}\n")

        p_long, p_short, b_long, b_short, imp = walk_forward(df)
        trades, equity = backtest(df, p_long, p_short, b_long, b_short)
        results[tf] = trades

        summarize(tf, trades, equity)

        print("\nВажность индикаторов (среднее по окнам):")
        s = pd.Series(imp, index=FEATURES).sort_values(ascending=False)
        print(s.round(3).to_string())

        suffix = "_shuffle" if SHUFFLE else ""
        suffix += "_v2"
        trades.to_csv(os.path.join(LOG_DIR, f"wf_trades_{tf}{suffix}.csv"), index=False)
        equity.to_csv(os.path.join(LOG_DIR, f"wf_equity_{tf}{suffix}.csv"), index=False)

    print("\n=== Прибыль по месяцам, $ ===")
    print(monthly_table(results).to_string(float_format=lambda x: f"{x:,.0f}"))

    print("\nКАК ЧИТАТЬ:")
    print("  AUC около 0.50 во всех окнах  -> у индикаторов нет предсказательной силы")
    print("  AUC стабильно 0.52+           -> есть слабый сигнал, смотрим PF и просадку")
    print("  PF выше 1.2 и плюс в большинстве месяцев на out-of-sample -> уже интересно")
    print("  Обязательно прогнать SHUFFLE=1: там результат должен быть плохим")


if __name__ == "__main__":
    main()