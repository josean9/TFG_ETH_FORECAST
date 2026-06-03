"""
Actualización diaria del CSV de criptomonedas.

Versión .py del notebook actualizar_diario.ipynb, pensada para ejecutarse
mediante GitHub Actions en cron diario.

Lo que hace:
  1. Carga data/csv/df_merged.csv y detecta cuántos días faltan.
  2. Si está al día, sale con código 0 (no es error).
  3. Si faltan 1-7 días, descarga de CoinGecko (BTC, ETH, total mcap) y
     Alternative.me (Fear & Greed).
  4. Calcula dominancias reales con el total mcap del momento (CoinGecko /global).
  5. Hace backup del CSV anterior en data/csv/backups/.
  6. Mergea, deduplica y guarda el CSV actualizado.

Si faltan más de 7 días, sale con código 1 (error) y avisa para usar el
script manual de CoinGecko Global Charts.

Diseñado para ejecutarse desde la raíz del repositorio:
  python scripts/actualizar_diario.py
"""

import os
import sys
import time
import shutil
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path


# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────

# Rutas relativas al raíz del repositorio para que funcionen tanto en local
# como en GitHub Actions.
BASE_DIR    = Path(__file__).resolve().parent.parent.parent  # raíz del repo

RUTA_CSV    = BASE_DIR / "data" / "csv" / "raw" / "df_merged.csv"
DIR_BACKUPS = BASE_DIR / "data" / "csv" / "backups"

# APIs
COINGECKO_BASE   = "https://api.coingecko.com/api/v3"
ALTERNATIVE_BASE = "https://api.alternative.me/fng/"

# Rate limit (free tier ~10-30 calls/min, vamos conservadores)
PAUSA_API = 2.5

# Si faltan más días que esto, el script avisa para usar el script manual
MAX_DIAS_AUTO = 7


# ─── FUNCIONES DE DESCARGA (con reintento) ────────────────────────────────

def get_con_reintento(url, params=None, intentos=3, espera=5):
    """GET con reintentos si la API tiene rate limit u otro 5xx."""
    for n in range(1, intentos + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503, 504):
                print(f"  ⚠️  {r.status_code} - reintento {n}/{intentos} en {espera}s")
                time.sleep(espera)
                continue
            raise RuntimeError(f"Error {r.status_code}: {r.text[:200]}")
        except requests.exceptions.RequestException as e:
            if n == intentos:
                raise
            print(f"  ⚠️  excepción - reintento {n}/{intentos}: {e}")
            time.sleep(espera)
    raise RuntimeError("Falló tras reintentos")


def ajustar_dias_coingecko(dias):
    """CoinGecko free solo acepta valores discretos."""
    for v in [1, 7, 14, 30, 90, 180, 365]:
        if dias <= v:
            return v
    return 365


def descargar_market_chart(coin_id, dias):
    """Descarga close, volume, market_cap diarios. Devuelve DataFrame indexado."""
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    data = get_con_reintento(
        url, {"vs_currency": "usd", "days": dias, "interval": "daily"}
    )

    df_p = pd.DataFrame(data["prices"],        columns=["ts", "close"])
    df_v = pd.DataFrame(data["total_volumes"], columns=["ts", "volume"])
    df_m = pd.DataFrame(data["market_caps"],   columns=["ts", "market_cap"])

    for d in (df_p, df_v, df_m):
        d["date"] = pd.to_datetime(d["ts"], unit="ms").dt.normalize()
        d.drop(columns="ts", inplace=True)
        d.set_index("date", inplace=True)

    df = df_p.join(df_v).join(df_m)
    return df.groupby(df.index).last()


def descargar_global():
    """Devuelve el market cap total del mercado cripto en este momento (USD)."""
    data = get_con_reintento(f"{COINGECKO_BASE}/global")
    return float(data["data"]["total_market_cap"]["usd"])


def descargar_fear_greed(limit_dias):
    """Descarga Fear & Greed reciente."""
    data = get_con_reintento(
        ALTERNATIVE_BASE, {"limit": limit_dias, "format": "json"}
    )
    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df["fear_greed"] = df["value"].astype(int)
    df["FearGreed_Label"] = df["value_classification"]
    return (
        df[["date", "fear_greed", "FearGreed_Label"]]
        .set_index("date")
        .sort_index()
    )


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────

def main():
    DIR_BACKUPS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Actualización diaria del CSV de criptomonedas")
    print("=" * 60)
    print(f"CSV principal: {RUTA_CSV}")
    print(f"Backups en   : {DIR_BACKUPS}")
    print()

    # ─── 1. Diagnóstico ──────────────────────────────────────────────────
    if not RUTA_CSV.exists():
        print(f"❌ ERROR: no existe el CSV principal en {RUTA_CSV}")
        sys.exit(1)

    df_merged = pd.read_csv(RUTA_CSV, parse_dates=["date"], index_col="date")
    df_merged.sort_index(inplace=True)

    ultima_fecha = df_merged.index.max().normalize()
    hoy          = pd.Timestamp.now().normalize()
    dias_faltan  = (hoy - ultima_fecha).days

    print(f"Última fecha en CSV : {ultima_fecha.date()}")
    print(f"Hoy                 : {hoy.date()}")
    print(f"Días a actualizar   : {dias_faltan}")
    print(f"Total filas actuales: {len(df_merged)}")
    print()

    # Salida temprana: ya al día
    if dias_faltan <= 0:
        print("✓ El CSV ya está al día. No hay nada que actualizar.")
        sys.exit(0)  # éxito limpio para GitHub Actions

    # Salida temprana: hueco demasiado grande
    if dias_faltan > MAX_DIAS_AUTO:
        print(f"⚠️  Faltan {dias_faltan} días, más del umbral seguro de {MAX_DIAS_AUTO}.")
        print("   Para huecos grandes, usa el script manual con el CSV de")
        print("   CoinGecko Global Charts para evitar extrapolar dominancias.")
        sys.exit(1)  # fallo: requiere intervención manual

    print(f"→ Procediendo a descargar {dias_faltan} día(s) nuevo(s).\n")

    # ─── 2. Descarga ─────────────────────────────────────────────────────
    dias_validos = ajustar_dias_coingecko(dias_faltan + 2)
    print(f"Solicitando {dias_validos} días a CoinGecko (cubre {dias_faltan} + margen)\n")

    print("→ Bitcoin...")
    df_btc = descargar_market_chart("bitcoin", dias_validos)
    df_btc.columns = ["btc_close", "btc_volume", "btc_mcap"]
    print(f"  {len(df_btc)} filas")
    time.sleep(PAUSA_API)

    print("\n→ Ethereum...")
    df_eth = descargar_market_chart("ethereum", dias_validos)
    df_eth.columns = ["eth_close", "eth_volume", "eth_mcap"]
    print(f"  {len(df_eth)} filas")
    time.sleep(PAUSA_API)

    print("\n→ Market cap total del mercado (CoinGecko /global)...")
    total_mcap_actual = descargar_global()
    print(f"  Total mcap actual: ${total_mcap_actual:,.0f}")
    time.sleep(PAUSA_API)

    print("\n→ Fear & Greed...")
    df_fg = descargar_fear_greed(limit_dias=dias_validos + 5)
    print(f"  {len(df_fg)} filas")

    # ─── 3. Construcción del DataFrame nuevo ─────────────────────────────
    print("\n" + "─" * 60)
    print("Construyendo DataFrame nuevo...")

    # Combinar BTC + ETH
    df_nuevo = df_btc.join(df_eth, how="outer")

    # Filtrar solo días estrictamente nuevos
    df_nuevo = df_nuevo[df_nuevo.index > ultima_fecha].copy()
    print(f"Días nuevos a añadir: {len(df_nuevo)}")

    if df_nuevo.empty:
        print("✓ No hay días nuevos tras filtrar. Salida limpia.")
        sys.exit(0)

    # Dominancias usando el total mcap actual (error <0.5% para 1-2 días)
    df_nuevo["btc_dominance"] = df_nuevo["btc_mcap"] / total_mcap_actual
    df_nuevo["eth_dominance"] = df_nuevo["eth_mcap"] / total_mcap_actual
    df_nuevo["alt_dominance"] = 1 - df_nuevo["btc_dominance"] - df_nuevo["eth_dominance"]

    # OHLC desde close (limitación API gratuita)
    for activo in ("btc", "eth"):
        df_nuevo[f"{activo}_open"] = df_nuevo[f"{activo}_close"]
        df_nuevo[f"{activo}_high"] = df_nuevo[f"{activo}_close"]
        df_nuevo[f"{activo}_low"]  = df_nuevo[f"{activo}_close"]

    # Fear & Greed
    df_nuevo = df_nuevo.join(df_fg, how="left")
    df_nuevo["fear_greed"]      = df_nuevo["fear_greed"].ffill()
    df_nuevo["FearGreed_Label"] = df_nuevo["FearGreed_Label"].ffill()

    # Inflation y fed_rate: forward-fill desde el último valor
    # (se actualizan manualmente cuando sale el dato mensual)
    df_nuevo["inflation"] = df_merged["inflation"].iloc[-1]
    df_nuevo["fed_rate"]  = df_merged["fed_rate"].iloc[-1]

    # Reordenar columnas igual que el CSV viejo
    df_nuevo = df_nuevo[df_merged.columns.tolist()]
    df_nuevo.index.name = "date"

    print(f"DataFrame nuevo listo: {df_nuevo.shape}")
    nulos = df_nuevo.isna().sum().sum()
    print(f"Nulos: {nulos}")
    if nulos > 0:
        print("⚠️  Hay nulos en el DataFrame nuevo. Revisar antes de seguir.")
        # No abortamos; los ffill posteriores deberían cubrirlo, pero avisamos.

    # ─── 4. Backup y merge ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    fecha_backup = hoy.strftime("%Y%m%d")
    ruta_backup  = DIR_BACKUPS / f"df_merged_backup_{fecha_backup}.csv"
    shutil.copy(RUTA_CSV, ruta_backup)
    print(f"✓ Backup guardado: {ruta_backup}")

    df_actualizado = pd.concat([df_merged, df_nuevo])
    df_actualizado = df_actualizado[~df_actualizado.index.duplicated(keep="last")]
    df_actualizado.sort_index(inplace=True)

    df_actualizado.to_csv(RUTA_CSV)

    print(f"\n✓ CSV actualizado guardado: {RUTA_CSV}")
    print(f"  Filas totales: {len(df_actualizado)} "
          f"({len(df_merged)} previas + {len(df_nuevo)} nuevas)")
    print(f"  Rango: {df_actualizado.index.min().date()} → "
          f"{df_actualizado.index.max().date()}")
    print("\nÚltimas 3 filas:")
    print(df_actualizado.tail(3).to_string())

    print("\n" + "=" * 60)
    print("Actualización completada con éxito ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
