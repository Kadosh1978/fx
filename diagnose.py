# diagnose.py
# Сравнивает старый и новый наборы признаков на одних данных.
#
# Запуск:
#   python diagnose.py                    # M5 по умолчанию
#   $env:RESAMPLE_TF="1h"; python diagnose.py
#   $env:RESAMPLE_TF="15min"; python diagnose.py

import os
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from main import (
    _load_csv, calculate_indicators, split_data,
    CSV_PATH, RESAMPLE_TF,
    OLD_FEATURES, NEW_FEATURES, FEATURES,
)

logging.basicConfig(level=logging.WARNING)

print("=" * 78)
print(f"ДИАГНОСТИКА фич | TF={RESAMPLE_TF} | CSV={CSV_PATH}")
print("=" * 78)

df = _load_csv(CSV_PATH, resample=RESAMPLE_TF)
df = calculate_indicators(df)
print(f"Баров после индикаторов: {len(df)}")
print(f"Диапазон: {df['time'].iloc[0]} .. {df['time'].iloc[-1]}")
print()


def make_binary_labels(df, horizon):
    fut = df["close"].shift(-horizon) / df["close"] - 1
    labels = (fut > 0).astype(float)
    labels[fut.isna().values] = np.nan
    df = df.copy()
    df["label"] = labels
    return df.dropna(subset=["label"]).reset_index(drop=True)


def evaluate(name, df, features):
    train, val, test = split_data(df, 0.6, 0.2)
    X_tr, y_tr = train[features].values, train["label"].values
    X_te, y_te = test[features].values, test["label"].values

    baseline_maj = (y_te == pd.Series(y_tr).mode()[0]).mean()

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, n_jobs=-1, random_state=42,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    bal = balanced_accuracy_score(y_te, pred)
    raw = (pred == y_te).mean()

    print(f"{name:<42} raw={raw:.3f} bal={bal:.3f} "
          f"| base_maj={baseline_maj:.3f} random=0.500")
    return bal, model


print("--- СРАВНЕНИЕ НАБОРОВ ПРИЗНАКОВ (binary UP/DOWN, h=3) ---")
d = make_binary_labels(df, horizon=3)
_ = evaluate("OLD (rsi/ema/macd/bb/atr)", d, OLD_FEATURES)
_ = evaluate("NEW (time/session/vol/context)", d, NEW_FEATURES)
bal_all, model_all = evaluate("ALL (old + new)", d, FEATURES)
print()


print("--- ALL на разных горизонтах ---")
for h in [1, 3, 5, 10, 20]:
    dh = make_binary_labels(df, horizon=h)
    evaluate(f"ALL, h={h}", dh, FEATURES)
print()


print("--- FEATURE IMPORTANCE (модель на ALL, h=3) ---")
importances = pd.Series(
    model_all.feature_importances_, index=FEATURES,
).sort_values(ascending=False)
print(importances.to_string())
print()


print("--- SANITY: перемешанные метки на ALL ---")
d_shuf = d.copy()
d_shuf["label"] = np.random.RandomState(0).permutation(d_shuf["label"].values)
evaluate("ALL, shuffled labels", d_shuf, FEATURES)
print()


print("=" * 78)
print("КАК ЧИТАТЬ:")
print("  bal(ALL) - bal(OLD) > 0.02  → новые фичи реально помогают")
print("  bal(ALL) ~ 0.50             → сигнала по-прежнему нет")
print("  feature_importance          → смотри, что реально несёт информацию")
print("  shuffled ~ 0.50             → утечки нет, pipeline честный")
print("=" * 78)