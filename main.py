# main.py
# Ядро ИИ-скальпинг-бота (XAUUSD).
# CSV-котировки, mock-режим, опционально MetaTrader5.
#
# Принципы:
#   - никакого look-ahead: train/val/test по времени
#   - бэктест только на out-of-sample (test)
#   - учёт спреда и комиссии
#   - размер позиции от риска
#   - разметка: TP раньше SL (ближе к реальной сделке)

import os
import time
import random
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# --- Опциональный MT5 ---
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# =========================================================
# КОНФИГ
# =========================================================

SYMBOL = "XAUUSD"
CSV_PATH = os.getenv("CSV_PATH", os.path.join("data", "XAUUSD_M1.csv"))
RESAMPLE_TF = os.getenv("RESAMPLE_TF", "5min")
DATA_SOURCE = os.getenv("DATA_SOURCE", "auto")
MAX_BARS = int(os.getenv("MAX_BARS", "100000"))

MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Издержки (XAUUSD)
SPREAD_POINTS = 0.45          # ~$0.30
COMMISSION_PER_LOT = 7.0      # round-trip USD за 1.0 лот
CONTRACT_SIZE = 100           # 1 лот = 100 унций
RISK_PER_TRADE = 0.01         # 1% баланса
MIN_LOT = 0.01
MAX_LOT = 5.0

# Разметка TP/SL
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0
MAX_BARS_AHEAD = 24

CONFIDENCE_THRESHOLD = 0.55

# --- Группы признаков ---
OLD_FEATURES = [
    "rsi", "ema_diff", "macd", "macd_signal",
    "bb_width", "atr", "returns", "htf_trend",
]

NEW_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "session_asia", "session_london", "session_ny", "session_overlap",
    "atr_regime",
    "dist_day_open_atr", "range_position",
    "body_ratio",
]

FEATURES = OLD_FEATURES + NEW_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("algrythm")


# =========================================================
# CSV + MOCK
# =========================================================

def _load_csv(path: str, resample: str = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV не найден: {path}")

    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"timestamp": "time"})

    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV нет колонок: {missing}")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).copy()

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    if resample:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        df = df.set_index("time").resample(resample).agg(agg).dropna().reset_index()

    log.info(f"CSV загружен: {len(df)} баров, {df['time'].iloc[0]} .. {df['time'].iloc[-1]}")
    return df


def _mock_data(bars: int = 3000) -> pd.DataFrame:
    log.info(f"[MOCK] генерирую {bars} баров")
    np.random.seed(42)
    price, rows = 2650.0, []
    now = int(time.time())
    for i in range(bars):
        price = max(100.0, price + np.random.normal(0, 0.8))
        rows.append({
            "time": now - (bars - i) * 300,
            "open": price + np.random.normal(0, 0.3),
            "high": price + abs(np.random.normal(0, 0.5)),
            "low": price - abs(np.random.normal(0, 0.5)),
            "close": price,
            "volume": int(abs(np.random.normal(1000, 300))) + 1,
        })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def get_historical_data(bars: int = None) -> pd.DataFrame:
    if bars is None:
        bars = MAX_BARS
    if DATA_SOURCE in ("auto", "csv"):
        try:
            df = _load_csv(CSV_PATH, resample=RESAMPLE_TF)
            return df.tail(bars).reset_index(drop=True)
        except Exception as e:
            if DATA_SOURCE == "csv":
                raise
            log.warning(f"CSV не загрузился ({e}), падаю в mock.")
    return _mock_data(bars)


# =========================================================
# ИНДИКАТОРЫ
# =========================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    df["ema_fast"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    ma20 = df["close"].rolling(20).mean()
    sd20 = df["close"].rolling(20).std()
    df["bb_width"] = (4 * sd20) / ma20.replace(0, np.nan)

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df["returns"] = df["close"].pct_change()
    df["htf_trend"] = np.sign(df["ema_slow"] - df["ema_slow"].shift(20)).fillna(0)

    t = pd.to_datetime(df["time"], utc=True)
    hour = t.dt.hour
    dow = t.dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["session_asia"] = ((hour >= 0) & (hour < 9)).astype(int)
    df["session_london"] = ((hour >= 7) & (hour < 16)).astype(int)
    df["session_ny"] = ((hour >= 12) & (hour < 21)).astype(int)
    df["session_overlap"] = ((hour >= 12) & (hour < 16)).astype(int)

    atr_baseline = df["atr"].rolling(100).mean()
    df["atr_regime"] = df["atr"] / atr_baseline.replace(0, np.nan)

    df["_date"] = t.dt.date
    day_open = df.groupby("_date")["open"].transform("first")
    day_high_so_far = df.groupby("_date")["high"].cummax()
    day_low_so_far = df.groupby("_date")["low"].cummin()
    day_range = (day_high_so_far - day_low_so_far).replace(0, np.nan)

    df["dist_day_open_atr"] = (df["close"] - day_open) / df["atr"].replace(0, np.nan)
    df["range_position"] = (df["close"] - day_low_so_far) / day_range

    body = (df["close"] - df["open"]).abs()
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = body / bar_range

    df = df.drop(columns=["_date"])
    return df.dropna().reset_index(drop=True)


# =========================================================
# РАЗМЕТКА: TP раньше SL
# =========================================================

def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    0=HOLD, 1=BUY, 2=SELL.

    BUY  — лонг: TP достигнут раньше SL.
    SELL — шорт: TP достигнут раньше SL.
    HOLD — иначе (оба, ни один, таймаут).
    """
    df = df.copy()
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values

    labels = np.full(n, np.nan)

    for i in range(n - 1):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        sl_dist = a * SL_ATR_MULT
        tp_dist = a * TP_ATR_MULT

        long_sl = close[i] - sl_dist
        long_tp = close[i] + tp_dist
        short_sl = close[i] + sl_dist
        short_tp = close[i] - tp_dist

        long_hit = None
        short_hit = None

        end = min(n, i + 1 + MAX_BARS_AHEAD)
        for j in range(i + 1, end):
            if long_hit is None:
                if low[j] <= long_sl:
                    long_hit = "SL"
                elif high[j] >= long_tp:
                    long_hit = "TP"
            if short_hit is None:
                if high[j] >= short_sl:
                    short_hit = "SL"
                elif low[j] <= short_tp:
                    short_hit = "TP"
            if long_hit and short_hit:
                break

        if long_hit == "TP" and short_hit != "TP":
            labels[i] = 1
        elif short_hit == "TP" and long_hit != "TP":
            labels[i] = 2
        else:
            labels[i] = 0

    df["label"] = labels
    return df.dropna(subset=["label"]).reset_index(drop=True).assign(
        label=lambda x: x.label.astype(int)
    )


# =========================================================
# SPLIT + МОДЕЛИ
# =========================================================

def split_data(df: pd.DataFrame, train_frac=0.6, val_frac=0.2):
    n = len(df)
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def _build_model(model_type: str, params: dict):
    if model_type == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 8),
        n_jobs=-1,
        random_state=42,
    )


def _quick_pf(df_val, model, features, label_mapping, initial_balance=10000.0):
    rev = {v: k for k, v in label_mapping.items()}
    wins = losses = 0.0
    pos = None
    for i in range(len(df_val)):
        row = df_val.iloc[i]
        price, atr = float(row["close"]), float(row["atr"])
        if pos is not None:
            hit_sl = (pos["side"] == "BUY" and price <= pos["sl"]) or \
                     (pos["side"] == "SELL" and price >= pos["sl"])
            hit_tp = (pos["side"] == "BUY" and price >= pos["tp"]) or \
                     (pos["side"] == "SELL" and price <= pos["tp"])
            if hit_sl or hit_tp:
                gross = (price - pos["entry"]) if pos["side"] == "BUY" else (pos["entry"] - price)
                pnl = gross * pos["lots"] * CONTRACT_SIZE \
                      - SPREAD_POINTS * pos["lots"] * CONTRACT_SIZE
                if pnl > 0:
                    wins += pnl
                else:
                    losses += -pnl
                pos = None
        if pos is None and atr > 0:
            X = row[features].values.reshape(1, -1).astype(float)
            sig = rev.get(model.predict(X)[0], "HOLD")
            if sig in ("BUY", "SELL"):
                sl_d = atr * SL_ATR_MULT
                tp_d = atr * TP_ATR_MULT
                lots = calculate_lot_size(initial_balance, RISK_PER_TRADE, sl_d)
                if sig == "BUY":
                    pos = {"side": "BUY", "entry": price, "lots": lots,
                           "sl": price - sl_d, "tp": price + tp_d}
                else:
                    pos = {"side": "SELL", "entry": price, "lots": lots,
                           "sl": price + sl_d, "tp": price - tp_d}
    return wins / losses if losses > 0 else (wins if wins > 0 else 0.0)


def train_model(df: pd.DataFrame, model_type: str = "xgboost",
                genetic_optimization: bool = False):
    train, val, test = split_data(df)
    log.info(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    X_tr, y_tr = train[FEATURES].values, train["label"].values
    X_val, y_val = val[FEATURES].values, val["label"].values
    label_mapping = {"HOLD": 0, "BUY": 1, "SELL": 2}

    if genetic_optimization:
        log.info("Random search по гиперпараметрам...")
        space = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        }
        best_params, best_score = None, -np.inf
        random.seed(42)
        n_iter = 8
        for i in range(n_iter):
            p = {k: random.choice(v) for k, v in space.items()}
            m = _build_model(model_type, p)
            m.fit(X_tr, y_tr)
            pf = _quick_pf(val, m, FEATURES, label_mapping)
            acc = balanced_accuracy_score(y_val, m.predict(X_val))
            score = pf + acc
            log.info(f"  [{i+1}/{n_iter}] {p} PF={pf:.2f} acc={acc:.3f}")
            if score > best_score:
                best_score, best_params = score, p
        log.info(f"Лучшие: {best_params}")
        final = best_params
    else:
        final = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05}

    X_full = np.vstack([X_tr, X_val])
    y_full = np.concatenate([y_tr, y_val])
    model = _build_model(model_type, final)
    model.fit(X_full, y_full)

    y_test = test["label"].values
    pred_train = model.predict(X_full)
    pred_val = model.predict(X_val)
    pred_test = model.predict(test[FEATURES].values)

    acc_train = balanced_accuracy_score(y_full, pred_train)
    acc_val = balanced_accuracy_score(y_val, pred_val)
    acc_test = balanced_accuracy_score(y_test, pred_test)

    log.info(f"Accuracy (balanced): train={acc_train:.3f} val={acc_val:.3f} test={acc_test:.3f}")

    cm = confusion_matrix(y_test, pred_test, labels=[0, 1, 2])
    log.info("Confusion matrix (test):")
    log.info("  истина\\пред  HOLD  BUY  SELL")
    for i, row in enumerate(cm):
        cls = ["HOLD", "BUY ", "SELL"][i]
        log.info(f"  {cls}      {row[0]:5d} {row[1]:4d} {row[2]:4d}")

    if acc_train - acc_test > 0.15:
        log.warning(">>> ПЕРЕОБУЧЕНИЕ: train намного выше test.")
    elif acc_test < 0.36:
        log.warning(">>> СЛАБЫЙ СИГНАЛ: test ≈ случайность.")
    elif acc_test > 0.40:
        log.info(">>> Есть сигнал, можно копать дальше.")

    return model, FEATURES, label_mapping


def save_model(model, features, label_mapping, model_type: str = "xgboost"):
    path = os.path.join(MODEL_DIR, f"model_{model_type}.joblib")
    joblib.dump({"model": model, "features": features,
                 "label_mapping": label_mapping}, path)
    log.info(f"Модель сохранена: {path}")


def load_model(model_type: str = "xgboost"):
    path = os.path.join(MODEL_DIR, f"model_{model_type}.joblib")
    if os.path.exists(path):
        d = joblib.load(path)
        log.info(f"Модель загружена: {path}")
        return d["model"], d["features"], d["label_mapping"]
    log.warning(f"Файл модели не найден: {path}")
    return None, None, None


# =========================================================
# POSITION SIZING
# =========================================================

def calculate_lot_size(balance: float, risk_pct: float, sl_distance: float,
                       contract_size: int = CONTRACT_SIZE) -> float:
    if sl_distance <= 0 or balance <= 0:
        return MIN_LOT
    risk_money = balance * risk_pct
    lots = risk_money / (sl_distance * contract_size)
    return float(np.clip(round(lots, 2), MIN_LOT, MAX_LOT))


# =========================================================
# БЭКТЕСТ (только test)
# =========================================================

def run_backtest(df, model, features, initial_balance=10000.0,
                 label_mapping=None, use_test_split_only=True):
    if model is None or features is None:
        raise ValueError("Модель не загружена.")

    if use_test_split_only:
        _, _, df = split_data(df)
        log.info(f"Бэктест на out-of-sample test: {len(df)} баров")

    rev = {v: k for k, v in (label_mapping or {"HOLD": 0, "BUY": 1, "SELL": 2}).items()}
    balance = initial_balance
    pos, trades, equity = None, [], []

    for i in range(len(df)):
        row = df.iloc[i]
        price, atr = float(row["close"]), float(row["atr"])

        if pos is not None:
            hit_sl = (pos["side"] == "BUY" and price <= pos["sl"]) or \
                     (pos["side"] == "SELL" and price >= pos["sl"])
            hit_tp = (pos["side"] == "BUY" and price >= pos["tp"]) or \
                     (pos["side"] == "SELL" and price <= pos["tp"])
            if hit_sl or hit_tp:
                gross = (price - pos["entry"]) if pos["side"] == "BUY" \
                    else (pos["entry"] - price)
                cost = (SPREAD_POINTS + COMMISSION_PER_LOT / CONTRACT_SIZE) \
                       * pos["lots"] * CONTRACT_SIZE
                pnl = gross * pos["lots"] * CONTRACT_SIZE - cost
                balance += pnl
                trades.append({
                    "time": row["time"].isoformat(),
                    "side": pos["side"], "lots": pos["lots"],
                    "entry": pos["entry"], "exit": price,
                    "profit": pnl, "balance": balance,
                    "reason": "TP" if hit_tp else "SL",
                })
                pos = None

        if pos is None and atr > 0:
            X = row[features].values.reshape(1, -1).astype(float)
            proba = model.predict_proba(X)[0]
            pred = model.predict(X)[0]
            sig = rev.get(pred, "HOLD")
            conf = float(proba.max())
            if sig in ("BUY", "SELL") and conf > CONFIDENCE_THRESHOLD:
                sl_d = atr * SL_ATR_MULT
                tp_d = atr * TP_ATR_MULT
                lots = calculate_lot_size(balance, RISK_PER_TRADE, sl_d)
                if sig == "BUY":
                    pos = {"side": "BUY", "entry": price, "lots": lots,
                           "sl": price - sl_d, "tp": price + tp_d}
                else:
                    pos = {"side": "SELL", "entry": price, "lots": lots,
                           "sl": price + sl_d, "tp": price - tp_d}

        equity.append({"time": row["time"], "equity": balance})

    pd.DataFrame(trades).to_csv(os.path.join(LOG_DIR, "backtest_trades.csv"), index=False)
    pd.DataFrame(equity).to_csv(os.path.join(LOG_DIR, "backtest_equity.csv"), index=False)

    n = len(trades)
    if n > 0:
        wins = sum(1 for t in trades if t["profit"] > 0)
        gross_win = sum(t["profit"] for t in trades if t["profit"] > 0)
        gross_loss = -sum(t["profit"] for t in trades if t["profit"] <= 0)
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        log.info(f"Бэктест: сделок={n}, winrate={wins/n:.1%}, "
                 f"PF={pf:.2f}, финал={balance:.2f}")
    else:
        log.info("Бэктест: сделок не было.")

    return {"trades": trades, "equity": equity, "final_balance": balance}


# =========================================================
# SMOKE-TEST
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Algrythm — smoke-test (XAUUSD, TP-before-SL labels)")
    print("=" * 60)

    df = get_historical_data()
    print(f"Загружено баров: {len(df)}")
    print(f"Диапазон: {df['time'].iloc[0]} .. {df['time'].iloc[-1]}")

    df = calculate_indicators(df)
    df = label_data(df)
    print(f"После индикаторов и разметки: {len(df)}")
    print(f"Классы: {df['label'].value_counts().to_dict()}")

    model, feats, mp = train_model(df, model_type="xgboost",
                                   genetic_optimization=False)
    save_model(model, feats, mp, model_type="xgboost")

    run_backtest(df, model, feats, initial_balance=10000.0,
                 label_mapping=mp, use_test_split_only=True)

    print("\nOK. Файлы: logs/backtest_trades.csv, logs/backtest_equity.csv")