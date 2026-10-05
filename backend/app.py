"""Backend de inferência: carrega o modelo BTC e expõe /health e /predict."""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "btc_model.joblib"

app = FastAPI(
    title="BTC Next-Day Close Predictor",
    description="Predições experimentais. Não é recomendação de investimento.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    close: float = Field(..., description="Preço de fechamento de hoje (USD)")
    return_1d: float = Field(..., description="Retorno diário (ex.: 0.01 = +1%)")
    ma_7: float = Field(..., description="Média móvel de 7 dias do close")


class PredictResponse(BaseModel):
    prediction: float
    horizon: str = "next_day_close"
    ticker: str = "BTC-USD"


artifact = None


@app.on_event("startup")
def load_model() -> None:
    global artifact
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Artefato não encontrado em {MODEL_PATH}. "
            "Execute: python scripts/train_btc.py "
            "ou o notebook notebooks/train_btc.ipynb antes do build/run."
        )
    artifact = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": artifact is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    if artifact is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    feature_cols = artifact["feature_cols"]
    row = {
        "close": body.close,
        "return_1d": body.return_1d,
        "ma_7": body.ma_7,
    }
    X = np.array([[row[col] for col in feature_cols]], dtype=float)
    coef = np.array(artifact["coef"], dtype=float)
    intercept = float(artifact["intercept"])
    value = float((X @ coef + intercept).ravel()[0])

    return PredictResponse(
        prediction=value,
        horizon=artifact.get("horizon", "next_day_close"),
        ticker=artifact.get("ticker", "BTC-USD"),
    )
