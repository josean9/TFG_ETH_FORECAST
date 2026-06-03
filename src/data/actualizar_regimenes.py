"""
Cálculo del régimen de mercado (HMM) para el pipeline diario.

Carga el modelo HMM ya entrenado (models/hmm_final.pkl) y clasifica TODA la serie
de df_merged en sus tres regímenes (Acumulacion / Bajista / Alcista), guardando el
resultado en data/csv/raw/regimenes.csv.

NO reentrena el HMM: solo lo carga y aplica. El modelo se entrena una vez (en el
notebook regimenes_mercado.ipynb) y se reutiliza aquí en producción.

Las 5 variables de entrada se calculan EXACTAMENTE igual que en el notebook de
entrenamiento (celda 4), porque el HMM debe ver los datos tal y como los vio al
entrenarse. El escalado, el PCA y el HMM vienen guardados dentro del .pkl.

Orden en el pipeline: actualizar_diario.py -> [ESTE] -> calcular_features.py
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ─── RUTAS (estructura: data/csv/raw, data/csv/processed, models) ──────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent   # raíz del repo (src/data/ -> 3 niveles)
RUTA_MERGED = BASE_DIR / "data" / "csv" / "raw" / "df_merged.csv"
RUTA_SALIDA = BASE_DIR / "data" / "csv" / "raw" / "regimenes.csv"
RUTA_MODELO = BASE_DIR / "models" / "hmm_final.pkl"


def calcular_variables_regimen(df):
    """
    Calcula las 5 variables crudas del régimen, IDÉNTICAS a la celda 4 del
    notebook de entrenamiento. Si esto cambia, el régimen en producción no
    coincidiría con el del entrenamiento.
    """
    eth = df["eth_close"]
    ret = eth.pct_change() * 100                       # retorno % diario

    reg = pd.DataFrame(index=df.index)
    reg["ret"]          = ret
    reg["vol_30d"]      = ret.rolling(30).std()                          # volatilidad 30d
    reg["cum_ret_60d"]  = ret.rolling(60).sum()                         # tendencia 60d
    reg["dist_sma200"]  = (eth / eth.rolling(200).mean() - 1) * 100     # distancia a SMA200
    reg["drawdown"]     = (eth / eth.cummax() - 1) * 100                # caída desde máximos
    reg["fg"]           = df["fear_greed"]                              # Fear & Greed crudo
    reg["precio"]       = eth

    return reg.dropna().copy()


def main():
    print("=" * 60)
    print("Cálculo de regímenes de mercado (HMM)")
    print("=" * 60)

    # ─── 1. Cargar el modelo entrenado ────────────────────────────────────
    if not RUTA_MODELO.exists():
        print(f"❌ ERROR: no existe el modelo en {RUTA_MODELO}")
        print("   Genera primero hmm_final.pkl ejecutando regimenes_mercado.ipynb")
        sys.exit(1)

    modelo = joblib.load(RUTA_MODELO)
    hmm          = modelo["hmm"]
    scaler       = modelo["scaler"]
    pca          = modelo["pca"]
    vars_regimen = modelo["vars_regimen"]
    nombres      = modelo["nombres"]
    print(f"✓ Modelo cargado: {RUTA_MODELO.name}")
    print(f"  Variables: {vars_regimen}")
    print(f"  Mapeo estados: {nombres}")

    # ─── 2. Cargar df_merged ──────────────────────────────────────────────
    if not RUTA_MERGED.exists():
        print(f"❌ ERROR: no existe {RUTA_MERGED}")
        sys.exit(1)

    df = pd.read_csv(RUTA_MERGED, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    print(f"✓ df_merged cargado: {len(df)} filas "
          f"({df.index.min().date()} -> {df.index.max().date()})")

    # ─── 3. Calcular las 5 variables (idénticas al entrenamiento) ─────────
    reg = calcular_variables_regimen(df)
    print(f"✓ Variables calculadas: {len(reg)} filas con datos completos")

    # ─── 4. Escalar -> PCA -> HMM (con los objetos guardados) ─────────────
    X  = scaler.transform(reg[vars_regimen].values)   # MISMO scaler del entrenamiento
    Xp = pca.transform(X)                             # MISMO PCA
    estados = hmm.predict(Xp)                         # clasificar

    reg["estado_hmm"] = estados
    reg["regimen"]    = [nombres.get(int(e), f"Estado {e}") for e in estados]

    # ─── 5. Guardar ───────────────────────────────────────────────────────
    salida = reg[["precio", "vol_30d", "cum_ret_60d", "dist_sma200", "drawdown",
                  "fg", "estado_hmm", "regimen"]].copy()
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(RUTA_SALIDA)

    print(f"\n✓ Regímenes guardados en {RUTA_SALIDA}")
    print(f"  Filas: {len(salida)}")
    print(f"  Régimen del último día ({salida.index[-1].date()}): "
          f"{salida['regimen'].iloc[-1]}")
    print("\nReparto de regímenes:")
    print(salida["regimen"].value_counts().to_string())

    print("\n" + "=" * 60)
    print("Cálculo de regímenes completado ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()