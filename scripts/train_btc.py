"""Baixa BTC-USD do Yahoo Finance, treina regressão linear e exporta o artefato.

Equivalente a sklearn.linear_model.LinearRegression (minimos quadrados via NumPy).
O artefato guarda coeficientes para o backend carregar sem sklearn.

Uso (na raiz do projeto):
    python scripts/train_btc.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Retorna (coef_, intercept_) como no LinearRegression do sklearn."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ones = np.ones((X.shape[0], 1))
    beta, *_ = np.linalg.lstsq(np.hstack([ones, X]), y, rcond=None)
    return beta[1:], float(beta[0])


def predict_linear(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(X, dtype=float) @ coef + intercept


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def yf_to_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Normaliza o DataFrame do yfinance para Date + Close."""
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [c[0] for c in frame.columns]
    frame = frame.reset_index()
    # coluna de data pode vir como Date ou Datetime
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    close_col = "Close"
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame[date_col], errors="coerce"),
            "Close": pd.to_numeric(frame[close_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)


def main() -> None:
    # yfinance: end e exclusivo -> 2026-10-06 inclui 05/10/2026
    print("Baixando BTC-USD do Yahoo Finance (2021-10-05 a 2026-10-05)...")
    data = yf.download("BTC-USD", start="2021-10-05", end="2026-10-06", auto_adjust=True)
    if data.empty:
        raise RuntimeError("Download do Yahoo Finance retornou vazio. Verifique a conexao.")

    csv_path = DATA_DIR / "btc_usd_dados.csv"
    data.to_csv(csv_path)
    print(f"CSV salvo: {csv_path} ({len(data)} linhas)")

    df = yf_to_frame(data)
    df["close"] = df["Close"]
    df["return_1d"] = df["close"].pct_change()
    df["ma_7"] = df["close"].rolling(7).mean()
    df["target_next_close"] = df["close"].shift(-1)

    feature_cols = ["close", "return_1d", "ma_7"]
    model_df = df.dropna(subset=feature_cols + ["target_next_close"]).copy()
    print(f"Amostras apos features: {len(model_df)}")
    print(
        f"Periodo: {model_df['Date'].iloc[0].date()} -> {model_df['Date'].iloc[-1].date()}"
    )

    split_idx = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_idx]
    test = model_df.iloc[split_idx:]

    X_train = train[feature_cols].to_numpy()
    y_train = train["target_next_close"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["target_next_close"].to_numpy()

    coef, intercept = fit_linear_regression(X_train, y_train)
    preds = predict_linear(X_test, coef, intercept)

    print(f"Treino: {len(train)} | Teste: {len(test)}")
    print(f"MAE (teste): {mae(y_test, preds):.2f}")
    print(f"R2 (teste): {r2_score(y_test, preds):.4f}")
    print("Coeficientes:", dict(zip(feature_cols, coef)))
    print("Intercepto:", intercept)

    artifact = {
        "coef": coef.tolist(),
        "intercept": intercept,
        "feature_cols": feature_cols,
        "ticker": "BTC-USD",
        "horizon": "next_day_close",
        "model_type": "LinearRegression",
    }
    model_path = MODELS_DIR / "btc_model.joblib"
    joblib.dump(artifact, model_path)
    print(f"Modelo salvo: {model_path}")


if __name__ == "__main__":
    main()
