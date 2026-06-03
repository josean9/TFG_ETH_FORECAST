"""
Cálculo de features (df_model) para el pipeline diario.

Junta df_merged.csv (mercado) + regimenes.csv (HMM) y calcula TODAS las features
derivadas (indicadores técnicos + sentimiento + régimen one-hot), dejando el
resultado en data/csv/processed/df_model.csv — el CSV que consume la LSTM.

La ingeniería de features es IDÉNTICA a la del notebook de entrenamiento de la LSTM.
Si esto cambiara, las predicciones en producción no coincidirían con el modelo entrenado.

Orden en el pipeline: actualizar_diario.py -> calcular_regimenes.py -> [ESTE]
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ─── RUTAS ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent   # src/data/ -> raíz
RUTA_MERGED  = BASE_DIR / "data" / "csv" / "raw" / "df_merged.csv"
RUTA_REGIM   = BASE_DIR / "data" / "csv" / "raw" / "regimenes.csv"
RUTA_SALIDA  = BASE_DIR / "data" / "csv" / "processed" / "df_model.csv"


# ─── INDICADORES (idénticos al notebook de la LSTM) ───────────────────────────
def rsi(close, n=14):
    delta = close.diff()
    g = delta.where(delta > 0, 0).rolling(n).mean()
    p = (-delta.where(delta < 0, 0)).rolling(n).mean()
    return 100 - (100 / (1 + g / (p + 1e-8)))

def estocastico(df, pfx, n=14, d=3):
    ll = df[f"{pfx}_low"].rolling(n).min()
    hh = df[f"{pfx}_high"].rolling(n).max()
    k = 100 * (df[f"{pfx}_close"] - ll) / (hh - ll + 1e-8)
    return k, k.rolling(d).mean()

def mfi(df, pfx, n=14):
    tp = (df[f"{pfx}_high"] + df[f"{pfx}_low"] + df[f"{pfx}_close"]) / 3
    flujo = tp * df[f"{pfx}_volume"]
    delta = tp.diff()
    fp = flujo.where(delta > 0, 0).rolling(n).sum()
    fn = flujo.where(delta < 0, 0).rolling(n).sum()
    return 100 - (100 / (1 + fp / (fn + 1e-8)))

def bollinger(close, n=20, k=2):
    ma = close.rolling(n).mean(); sd = close.rolling(n).std()
    sup, inf = ma + k*sd, ma - k*sd
    return (close - inf) / (sup - inf + 1e-8), (sup - inf) / (ma + 1e-8)


def construir_features(df_raw):
    """Replica EXACTAMENTE la ingeniería de features del notebook (bloques A-F, E2)."""
    # OHE del sentimiento
    ohe = pd.get_dummies(df_raw["FearGreed_Label"], prefix="fng", dtype=int)
    ohe.columns = [c.replace(" ", "_") for c in ohe.columns]
    df_raw = df_raw.drop(columns=["FearGreed_Label"]).join(ohe)
    OHE_COLS = [c for c in df_raw.columns if c.startswith("fng_")]

    df = df_raw.copy()
    # Bloque A: retornos
    for col in ["btc_close","btc_volume","eth_close","eth_volume","btc_mcap","eth_mcap"]:
        df[f"{col}_ret"] = df[col].pct_change() * 100
    # Bloque B: dominancias
    for col in ["btc_dominance","eth_dominance","alt_dominance"]:
        df[f"{col}_diff"] = df[col].diff()
    for col in ["btc_dominance","eth_dominance"]:
        for w in [14,30,60]:
            df[f"{col}_chg{w}d"] = df[col] - df[col].shift(w)
    # Bloque C: macro + volatilidad
    for col in ["inflation","fed_rate"]:
        df[f"{col}_chg30"] = df[col].diff(30)
    for w in [7,14,30]:
        df[f"eth_vol_{w}d"] = df["eth_close_ret"].rolling(w).std()
    # Bloque D: osciladores
    df["eth_rsi14"] = rsi(df["eth_close"]); df["btc_rsi14"] = rsi(df["btc_close"])
    df["eth_stoch_k"], df["eth_stoch_d"] = estocastico(df, "eth")
    df["btc_stoch_k"], df["btc_stoch_d"] = estocastico(df, "btc")
    df["eth_mfi14"] = mfi(df, "eth"); df["btc_mfi14"] = mfi(df, "btc")
    df["eth_bb_pctb"], df["eth_bb_width"] = bollinger(df["eth_close"])
    for w in [10,15]:
        df[f"eth_rsi_sobrecompra_{w}d"] = (df["eth_rsi14"] > 70).rolling(w).sum()
        df[f"eth_rsi_sobreventa_{w}d"]  = (df["eth_rsi14"] < 30).rolling(w).sum()
    # Bloque E: sentimiento
    df["fear_greed_scaled"] = df["fear_greed"] / 100.0
    for w in [15,30]:
        fc = [c for c in OHE_COLS if "Fear" in c]; gc = [c for c in OHE_COLS if "Greed" in c]
        fs = sum(df[c].rolling(w).sum() for c in fc); gs = sum(df[c].rolling(w).sum() for c in gc)
        df[f"fear_greed_ratio_{w}d"] = fs / (gs + 1e-8)
    for w in [7,15,30]:
        df[f"eth_cum_ret_{w}d"] = df["eth_close_ret"].rolling(w).sum()
    # Bloque F: nivel
    df["eth_btc_ratio"] = df["eth_close"] / df["btc_close"]
    df["eth_drawdown"] = (df["eth_close"] / df["eth_close"].cummax() - 1) * 100
    df["btc_drawdown"] = (df["btc_close"] / df["btc_close"].cummax() - 1) * 100
    for w in [50,200]:
        df[f"eth_dist_sma{w}"] = (df["eth_close"] / df["eth_close"].rolling(w).mean() - 1) * 100
    # Bloque E2: sentimiento acumulado
    estados = {"miedo_ext":"fng_Extreme_Fear","miedo":"fng_Fear","neutral":"fng_Neutral",
               "codicia":"fng_Greed","codicia_ext":"fng_Extreme_Greed"}
    presentes = {k:v for k,v in estados.items() if v in df.columns}
    for w in [15,30,60,90]:
        for nombre,col in presentes.items():
            df[f"n_{nombre}_{w}d"] = df[col].rolling(w).sum()
        if "fng_Extreme_Greed" in df.columns and "fng_Extreme_Fear" in df.columns:
            df[f"presion_ext_neta_{w}d"] = (df["fng_Extreme_Greed"].rolling(w).sum()
                                            - df["fng_Extreme_Fear"].rolling(w).sum())
        df[f"fg_presion_{w}d"] = (df["fear_greed"] - 50).rolling(w).sum()

    return df


def main():
    print("=" * 60)
    print("Cálculo de features (df_model)")
    print("=" * 60)

    # ─── 1. Cargar df_merged ──────────────────────────────────────────────
    if not RUTA_MERGED.exists():
        print(f"❌ ERROR: no existe {RUTA_MERGED}"); sys.exit(1)
    df_raw = pd.read_csv(RUTA_MERGED, parse_dates=["date"], index_col="date").sort_index()
    print(f"✓ df_merged: {df_raw.shape}  ({df_raw.index.min().date()} -> {df_raw.index.max().date()})")

    # ─── 2. Construir features ────────────────────────────────────────────
    df = construir_features(df_raw)
    print(f"✓ Features construidas: {df.shape[1]} columnas")

    # ─── 3. Cruzar el régimen (one-hot de estado_hmm_causal) ──────────────
    if not RUTA_REGIM.exists():
        print(f"❌ ERROR: no existe {RUTA_REGIM} (ejecuta antes calcular_regimenes.py)")
        sys.exit(1)
    reg = pd.read_csv(RUTA_REGIM, parse_dates=["date"], index_col="date")
    col_reg = "estado_hmm_causal" if "estado_hmm_causal" in reg.columns else "estado_hmm"
    ohe_reg = pd.get_dummies(reg[col_reg], prefix="regime", dtype=int)
    df = df.join(ohe_reg, how="left")
    print(f"✓ Régimen cruzado ({col_reg}): {list(ohe_reg.columns)}")

    # ─── 4. Limpiar NaN (las ventanas largas dejan NaN al principio) ──────
    df_model = df.dropna().copy()
    print(f"✓ df_model final: {df_model.shape}  ({df_model.index.min().date()} -> {df_model.index.max().date()})")

    # ─── 5. Guardar ───────────────────────────────────────────────────────
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(RUTA_SALIDA)
    print(f"\n✓ Guardado en {RUTA_SALIDA}")
    print(f"  Filas: {len(df_model)}  |  Columnas: {df_model.shape[1]}")
    print(f"  Último día: {df_model.index[-1].date()}")

    print("\n" + "=" * 60)
    print("Cálculo de features completado ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
