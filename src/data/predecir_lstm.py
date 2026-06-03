"""
Predicción de la LSTM para el sistema RAG.

Carga el modelo entrenado (lstm_final.pt) + scalers, coge los últimos SEQ_LEN días
del df_model, y devuelve la señal de los próximos HORIZON días.

USO COMO FUNCIÓN (desde la app del RAG):
    from predecir_lstm import predecir
    senal = predecir()
    # senal es un dict con retornos, precios estimados, régimen y nota de fiabilidad

USO COMO SCRIPT (para probar):
    python predecir_lstm.py
    # imprime la señal por pantalla

NO reentrena: solo carga y aplica el modelo ya congelado.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn

# ─── RUTAS ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent   # src/senal/ -> raíz
RUTA_MODEL  = BASE_DIR / "data" / "csv" / "processed" / "df_model.csv"
RUTA_PT     = BASE_DIR / "models" / "lstm_final.pt"
RUTA_SCALER = BASE_DIR / "models" / "scaler_lstm.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nombres legibles del régimen (por si el df_model trae regime_0/1/2)
NOMBRES_REGIMEN = {0: "Acumulacion", 1: "Bajista", 2: "Alcista"}


# ─── ARQUITECTURA (debe coincidir con la del entrenamiento) ───────────────────
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_sizes=(32,), horizon=3, dropout=0.35):
        super().__init__()
        capas, in_size = [], n_features
        for h in hidden_sizes:
            capas.append(nn.LSTM(in_size, h, batch_first=True)); in_size = h
        self.lstms = nn.ModuleList(capas)
        self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in hidden_sizes])
        self.head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], max(hidden_sizes[-1]//2,1)), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(max(hidden_sizes[-1]//2,1), horizon))
    def forward(self, x):
        out = x
        for lstm, drop in zip(self.lstms, self.drops):
            out, _ = lstm(out); out = drop(out)
        return self.head(out[:, -1, :])


def predecir():
    """
    Devuelve un dict con la señal de la LSTM para los próximos días:
      - fecha_base, precio_hoy
      - retornos_diarios: [r1, r2, r3]  (% diario, encadenados día a día)
      - precios_estimados: [p1, p2, p3] (precio encadenado: cada día sobre el anterior)
      - retorno_acumulado: % total a HORIZON días (encadenado, no suma)
      - regimen_actual: nombre del régimen del último día (contexto a contrastar)
      - fiabilidad: nota honesta sobre magnitud vs dirección
    """
    # --- cargar modelo y scalers ---
    if not RUTA_PT.exists() or not RUTA_SCALER.exists():
        raise FileNotFoundError("Faltan models/lstm_final.pt o models/scaler_lstm.pkl")
    ckpt = torch.load(RUTA_PT, map_location=DEVICE, weights_only=False)
    scalers = joblib.load(RUTA_SCALER)
    sx, sy = scalers["sx"], scalers["sy"]
    feature_cols = ckpt["feature_cols"]
    regime_cols  = ckpt["regime_cols"]
    seq_len, horizon = ckpt["seq_len"], ckpt["horizon"]
    arq = ckpt["arquitectura"]

    modelo = LSTMRegressor(arq["n_features"], tuple(arq["hidden_sizes"]),
                           arq["horizon"], arq["dropout"]).to(DEVICE)
    modelo.load_state_dict(ckpt["state_dict"]); modelo.eval()

    # --- cargar df_model y coger los últimos seq_len días ---
    df = pd.read_csv(RUTA_MODEL, parse_dates=["date"], index_col="date").sort_index()
    if len(df) < seq_len:
        raise ValueError(f"df_model tiene {len(df)} filas, se necesitan {seq_len}")
    ventana = df.iloc[-seq_len:]
    fecha_base = ventana.index[-1]

    # --- construir X igual que en entrenamiento (escalar continuas + régimen sin escalar) ---
    Xc = sx.transform(ventana[feature_cols].values)
    Xr = ventana[regime_cols].values.astype(np.float32)
    X = np.hstack([Xc, Xr]).astype(np.float32)
    X_t = torch.from_numpy(X).unsqueeze(0).to(DEVICE)   # (1, seq_len, n_features)

    # --- predecir y devolver a escala real (%) ---
    with torch.no_grad():
        pred_esc = modelo(X_t).cpu().numpy().ravel()      # horizon valores escalados
    retornos = sy.inverse_transform(pred_esc.reshape(-1,1)).ravel()   # % reales

    # --- precio de hoy: reconstruir desde el último retorno conocido si hace falta ---
    # df_model no tiene eth_close (se excluyó). Usamos el precio si está, o None.
    precio_hoy = float(ventana["eth_close"].iloc[-1]) if "eth_close" in ventana.columns else None

    # --- encadenar precios (cada día sobre el anterior) ---
    precios = None
    if precio_hoy is not None:
        precios, p = [], precio_hoy
        for r in retornos:
            p = p * (1 + r/100.0); precios.append(round(float(p), 2))

    # --- retorno acumulado encadenado (no suma) ---
    acum = (np.prod([1 + r/100.0 for r in retornos]) - 1) * 100

    # --- régimen actual (contexto a contrastar, no dogma) ---
    regimen = None
    if "regimen" in ventana.columns:
        regimen = str(ventana["regimen"].iloc[-1])
    elif regime_cols:
        activo = ventana[regime_cols].iloc[-1]
        if activo.sum() > 0:
            idx = int(activo.idxmax().split("_")[-1])
            regimen = NOMBRES_REGIMEN.get(idx, f"Estado {idx}")

    return {
        "fecha_base": str(fecha_base.date()),
        "precio_hoy": precio_hoy,
        "horizonte_dias": int(horizon),
        "retornos_diarios": [round(float(r), 3) for r in retornos],
        "precios_estimados": precios,
        "retorno_acumulado": round(float(acum), 3),
        "regimen_actual": regimen,
        "fiabilidad": ("La MAGNITUD del movimiento es orientativa; la DIRECCIÓN (signo) "
                       "es poco fiable (~52%, cercano al azar). El régimen es contexto "
                       "del HMM y puede ir con retraso: contrástalo, no lo tomes como certeza."),
    }


def main():
    print("="*60); print("Predicción LSTM (señal para el RAG)"); print("="*60)
    try:
        s = predecir()
    except Exception as e:
        print(f"❌ {e}"); sys.exit(1)
    print(f"Fecha base       : {s['fecha_base']}")
    print(f"Precio hoy       : {s['precio_hoy']}")
    print(f"Régimen actual   : {s['regimen_actual']}")
    print(f"Retornos diarios : {s['retornos_diarios']} %")
    print(f"Precios estimados: {s['precios_estimados']}")
    print(f"Retorno acum. {s['horizonte_dias']}d: {s['retorno_acumulado']} %")
    print(f"\nNota: {s['fiabilidad']}")
    print("="*60)


if __name__ == "__main__":
    main()
