"""
Módulo RAG para el sistema de apoyo a la decisión sobre Ethereum.

Orquesta:
  - Snapshot de mercado AMPLIADO (df_model + comparativas temporales + macro)
  - Señal cuantitativa de la LSTM (predicción a 3 días)
  - Régimen del HMM (viene en df_model)
  - Noticias frescas (noticias.csv, con muestreo ponderado 8+8+8)
  - Conocimiento experto (TXT fijos) + documentos recuperados (TXT embebidos, retriever)
  - Generación con LLM (Gemini por defecto; Claude enchufable)

Pensado para ser importado por app.py (Gradio). Toda la lógica vive aquí;
la app solo gestiona la interfaz y la memoria conversacional.

NOTA SOBRE EL SNAPSHOT (lo importante de esta versión):
  El snapshot es una FOTO COMPLETA del mercado en el día del análisis. No se queda
  en el valor de hoy: para las variables clave (precio, dominancia, sentimiento...)
  muestra también el valor de hace 2 semanas, 1 mes, 2 meses y 3 meses, para que el
  modelo pueda razonar sobre la TRAYECTORIA, no solo sobre la foto fija. Además añade
  un bloque MACRO (dólar, oro, Nasdaq, bono 10 años, yen y sus correlaciones con BTC)
  que se descarga de yfinance al arrancar. Si esa descarga falla, el RAG sigue
  funcionando igual, solo que sin el bloque macro.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import torch
import torch.nn as nn
import joblib

from google import genai
from google.genai import types

# ─── RUTAS ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

RUTA_DFMODEL   = BASE_DIR / "data" / "csv" / "processed" / "df_model.csv"
RUTA_NOTICIAS  = BASE_DIR / "data" / "news" / "noticias.csv"
DIR_TXT_FIJOS  = BASE_DIR / "data" / "txt_rag" / "fijos"
DIR_TXT_EMBEB  = BASE_DIR / "data" / "txt_rag" / "embebidos"
DIR_MODELOS    = BASE_DIR / "models"

# ─── CONFIG ───────────────────────────────────────────────────────────────
MODELO_LLM   = "gemini-3.5-flash"
N_ETH, N_CRIPTO, N_MACRO = 8, 8, 8     # noticias por categoría
TOP_K_EMBEB  = 4                        # fragmentos de docs embebidos
CHUNK_SIZE, CHUNK_OVERLAP = 800, 150
NOMBRES_REGIMEN = {0: "Acumulacion", 1: "Bajista", 2: "Alcista"}

# Ventanas temporales (en días) para la comparativa "hoy vs antes".
# Son las que pediste: 2 semanas, 1 mes, 2 meses y 3 meses.
OFFSETS_COMPARATIVA = [14, 30, 60, 90]

# Tickers de yfinance para el bloque macro. Si algún día quieres añadir o quitar
# activos, se hace aquí (clave = nombre interno, valor = símbolo de yfinance).
TICKERS_MACRO = {
    "dxy":    "DX-Y.NYB",   # Índice del dólar (fortaleza del dólar frente a otras divisas)
    "oro":    "GC=F",       # Oro (futuros)
    "nasdaq": "^NDX",       # Nasdaq 100 (bolsa tecnológica USA)
    "us10y":  "^TNX",       # Rendimiento del bono del Tesoro USA a 10 años
    "usdjpy": "JPY=X",      # Tipo de cambio dólar / yen japonés
}

# ─── CLIENTE GEMINI ──────────────────────────────────────────────────────
_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Convierte una lista de textos en vectores numéricos (embeddings).

    En cristiano: un embedding es una forma de representar un texto como una
    lista de números, de modo que textos parecidos tengan números parecidos.
    Eso es lo que luego permite "buscar por significado". Google solo deja
    mandar 100 textos por llamada, así que si hay más, los troceamos en
    grupos de 100 y los vamos acumulando.
    """
    if isinstance(texts, str):
        texts = [texts]
    vectores = []
    for i in range(0, len(texts), 100):
        lote = texts[i:i+100]
        r = _client.models.embed_content(
            model="gemini-embedding-001", contents=lote,
            config=types.EmbedContentConfig(task_type=task_type))
        vectores.extend([e.values for e in r.embeddings])
    return vectores


# ─── LSTM ────────────────────────────────────────────────────────────────
class LSTMRegressor(nn.Module):
    """La red neuronal LSTM que estima el retorno del precio a 3 días.

    En cristiano: es el "cerebro numérico" que mira los últimos 30 días de
    datos e intenta estimar cómo se moverá el precio en los 3 días siguientes.
    Su DIRECCIÓN acierta poco (casi como tirar una moneda), así que se usa
    como una pista más, no como una verdad.
    """
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


# ─── ESTADO GLOBAL (se carga una vez al importar) ────────────────────────
_df = None                 # el df_model entero, con todo el histórico
_noticias_df = None        # las noticias del día (noticias.csv)
_emb_chunks, _emb_matrix = [], None   # trozos de los TXT embebidos + sus vectores
_contexto_fijo = ""        # los TXT fijos concatenados (conocimiento experto)
_macro_diario = None       # datos macro de yfinance, alineados a las fechas del df_model
_macro_fecha = None        # fecha real del último dato macro (puede ser "ayer")
_macro_correls = {}        # correlaciones BTC vs nasdaq/dxy/oro (ventana 30 días)


def _cargar_df():
    """Carga el df_model.csv (la tabla con todo el histórico de mercado y features)."""
    global _df
    _df = pd.read_csv(RUTA_DFMODEL, parse_dates=["date"], index_col="date").sort_index()


def _cargar_txt_fijos():
    """Lee los TXT 'fijos' (conocimiento experto que SIEMPRE va al prompt).

    En cristiano: son los documentos de contexto y de razonamiento que quieres
    que el modelo tenga siempre delante (cómo interpretar el oro frente a BTC,
    mentalidad de inversión, etc.). Se concatenan todos en un solo bloque.
    """
    global _contexto_fijo
    partes = []
    if DIR_TXT_FIJOS.is_dir():
        for fn in sorted(os.listdir(DIR_TXT_FIJOS)):
            if fn.endswith(".txt"):
                partes.append(f"### {fn} ###\n" + (DIR_TXT_FIJOS/fn).read_text(encoding="utf-8"))
    _contexto_fijo = "\n\n".join(partes)


def _chunkear(texto, size, overlap):
    """Parte un texto largo en trozos solapados.

    En cristiano: corta un documento en pedazos de 'size' caracteres, dejando
    que cada pedazo solape un poco con el anterior ('overlap') para no cortar
    ideas por la mitad.
    """
    chunks, ini = [], 0
    while ini < len(texto):
        chunks.append(texto[ini:ini+size]); ini += max(size - overlap, 1)
    return [c.strip() for c in chunks if c.strip()]


# Tope de trozos para no pasar del límite de embeddings/minuto de Google (free = 100).
MAX_CHUNKS_TOTAL = 90
# Máximo de caracteres por trozo (el modelo de embeddings acepta ~2048 tokens ≈ 8000 chars).
CHUNK_SIZE_MAX = 6000


def _indexar_embebidos():
    """Trocea + embebe los TXT 'embebidos' en memoria (el buscador por significado).

    En cristiano: coge los documentos largos (historia de Ethereum, conceptos
    cripto...), los parte en trozos, y convierte cada trozo en números
    (embeddings). Así, cuando llegue una pregunta, podemos recuperar SOLO los
    trozos que hablan de lo que se pregunta. El tamaño del trozo se calcula
    solo para que el total no pase de ~90 trozos y no choquemos con el límite
    gratuito de Google (100 embeddings por minuto).
    """
    global _emb_chunks, _emb_matrix

    docs = []
    if DIR_TXT_EMBEB.is_dir():
        for fn in sorted(os.listdir(DIR_TXT_EMBEB)):
            if fn.endswith(".txt"):
                docs.append((fn, (DIR_TXT_EMBEB/fn).read_text(encoding="utf-8")))
    if not docs:
        _emb_chunks, _emb_matrix = [], None
        print("  (sin documentos embebidos)")
        return

    total_chars = sum(len(t) for _, t in docs)
    overlap = 150
    size = int(total_chars / MAX_CHUNKS_TOTAL) + overlap + 1
    size = min(max(size, 800), CHUNK_SIZE_MAX)

    chunks = []
    for fn, texto in docs:
        for ch in _chunkear(texto, size, overlap):
            chunks.append({"fuente": fn, "text": ch})

    if len(chunks) > 100:
        chunks = chunks[:100]

    print(f"  Embebidos: {len(docs)} docs, {total_chars:,} chars → "
          f"{len(chunks)} chunks de ~{size} chars")

    # Si los embeddings fallan (p.ej. 429 RESOURCE_EXHAUSTED por agotar la cuota
    # gratuita de Google), NO tumbamos la app: arranca igual, solo que sin el
    # buscador de documentos embebidos. Los TXT fijos y las noticias no dependen
    # de esto, así que el RAG sigue respondiendo casi completo.
    try:
        M = np.array(embed_texts([c["text"] for c in chunks]), dtype=np.float32)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        _emb_chunks, _emb_matrix = chunks, M
    except Exception as e:
        _emb_chunks, _emb_matrix = [], None
        print(f"  ⚠️ No se pudieron calcular los embeddings ({e}).")
        print("     La app arrancará SIN buscador de documentos embebidos "
              "(probable cuota de Google agotada; reponer y reiniciar).")


def _cargar_noticias():
    """Carga las noticias del día desde noticias.csv (con su nota 0-10)."""
    global _noticias_df
    if RUTA_NOTICIAS.exists():
        _noticias_df = pd.read_csv(RUTA_NOTICIAS)
    else:
        _noticias_df = pd.DataFrame(columns=["date","category","nota","source","title","text"])


# ─── BLOQUE MACRO (yfinance, opcional y a prueba de fallos) ───────────────
def _descargar_macro():
    """Descarga datos macro (dólar, oro, Nasdaq, bono 10a, yen) y los prepara.

    En cristiano: se baja de internet (yfinance) el valor diario del dólar, el
    oro, el Nasdaq, el bono americano a 10 años y el yen, durante los últimos
    meses. Como esos mercados cierran findes y festivos (y el cierre de HOY no
    está disponible hasta que cierra la bolsa), se "rellena hacia delante": cada
    día sin dato hereda el último valor conocido. Guardamos también la FECHA del
    último dato real, porque puede ser de ayer o del viernes pasado, y queremos
    decírselo al modelo con honestidad.

    Además calcula las correlaciones de Bitcoin con el Nasdaq, el dólar y el oro
    en los últimos 30 días (si suben juntos, si se mueven al revés, etc.).

    IMPORTANTE: si algo falla (sin internet, yfinance caído...), esta función no
    rompe nada: devuelve (None, None, {}) y el RAG sigue funcionando sin macro.
    """
    global _macro_diario, _macro_fecha, _macro_correls
    try:
        import yfinance as yf
    except Exception as e:
        print(f"  ⚠️ yfinance no disponible ({e}); el snapshot irá sin bloque macro.")
        _macro_diario, _macro_fecha, _macro_correls = None, None, {}
        return _macro_diario, _macro_fecha, _macro_correls

    try:
        hoy = pd.Timestamp.today().normalize()
        inicio = (hoy - pd.Timedelta(days=220)).strftime("%Y-%m-%d")
        fin    = (hoy + pd.Timedelta(days=2)).strftime("%Y-%m-%d")  # +2 por husos horarios

        series = {}
        for nombre, ticker in TICKERS_MACRO.items():
            try:
                data = yf.download(ticker, start=inicio, end=fin,
                                   progress=False, auto_adjust=True)
                if data is None or data.empty:
                    print(f"    ⚠️ sin datos para {nombre} ({ticker})")
                    continue
                cierre = data["Close"]
                # yfinance moderno a veces devuelve columnas multinivel: nos quedamos con la serie
                if isinstance(cierre, pd.DataFrame):
                    cierre = cierre.iloc[:, 0]
                series[nombre] = cierre
            except Exception as e:
                print(f"    ⚠️ error bajando {nombre} ({ticker}): {e}")

        if not series:
            print("  ⚠️ no se pudo bajar ningún activo macro; snapshot sin macro.")
            _macro_diario, _macro_fecha, _macro_correls = None, None, {}
            return _macro_diario, _macro_fecha, _macro_correls

        macro = pd.concat(series, axis=1)
        macro.columns = list(series.keys())
        macro.index = pd.to_datetime(macro.index).tz_localize(None)

        # Fecha del último dato macro REAL (antes de rellenar). Puede ser "ayer".
        _macro_fecha = macro.dropna(how="all").index.max()

        # Alinear a las fechas del df_model (diario 24/7) rellenando hacia delante.
        idx = _df.index
        macro_alineado = macro.reindex(idx.union(macro.index)).ffill().reindex(idx)
        _macro_diario = macro_alineado

        # Correlaciones de BTC con los activos macro (ventana 30 días).
        if _df is not None and "btc_close" in _df.columns:
            ret_btc = _df["btc_close"].pct_change()
            for activo in ["nasdaq", "dxy", "oro"]:
                if activo in macro_alineado.columns:
                    ret_a = macro_alineado[activo].pct_change()
                    corr = ret_btc.rolling(30).corr(ret_a).iloc[-1]
                    _macro_correls[f"corr_btc_{activo}_30d"] = (
                        round(float(corr), 2) if pd.notna(corr) else None)

        print(f"  Macro OK: {list(series.keys())} | último dato real: "
              f"{_macro_fecha.date() if _macro_fecha is not None else '?'}")
        return _macro_diario, _macro_fecha, _macro_correls

    except Exception as e:
        print(f"  ⚠️ fallo general bajando macro ({e}); snapshot sin macro.")
        _macro_diario, _macro_fecha, _macro_correls = None, None, {}
        return _macro_diario, _macro_fecha, _macro_correls


def inicializar():
    """Carga todo el estado una vez (df, txt, embebidos, noticias, macro).

    En cristiano: prepara todo lo que el RAG necesita para responder, y lo deja
    en memoria para no tener que cargarlo en cada pregunta. Se llama una sola vez
    al arrancar la app.
    """
    _cargar_df(); _cargar_txt_fijos(); _indexar_embebidos(); _cargar_noticias()
    _descargar_macro()   # opcional: si falla, seguimos sin macro
    print(f"RAG inicializado: df={_df.shape if _df is not None else None}, "
          f"noticias={len(_noticias_df)}, embebidos={len(_emb_chunks)} chunks, "
          f"macro={'sí' if _macro_diario is not None else 'no'}")


# ─── HELPERS DEL SNAPSHOT ────────────────────────────────────────────────
def _regimen_actual(fecha=None):
    """Devuelve el régimen del mercado según el HMM (Acumulación / Bajista / Alcista)."""
    fila = (_df.loc[:fecha] if fecha else _df).iloc[-1]
    for i, nombre in NOMBRES_REGIMEN.items():
        if f"regime_{i}" in fila and fila[f"regime_{i}"] == 1:
            return nombre
    return "Desconocido"


def _etiqueta_fg(valor):
    """Traduce el número de miedo/codicia (0-100) a su etiqueta en español."""
    if valor is None:
        return "desconocido"
    if valor < 25:  return "Miedo extremo"
    if valor < 45:  return "Miedo"
    if valor < 55:  return "Neutral"
    if valor < 75:  return "Codicia"
    return "Codicia extrema"


def _hace(sub, col, n):
    """Valor de una columna hace 'n' días (n filas atrás, porque el df es diario continuo)."""
    if col not in sub.columns or len(sub) < n + 1:
        return None
    v = sub[col].iloc[-(n+1)]
    return float(v) if pd.notna(v) else None


def _delta_pct(actual, pasado):
    """Variación porcentual entre el valor actual y uno pasado (en %)."""
    if actual is None or pasado is None or pasado == 0:
        return None
    return round(((actual / pasado) - 1) * 100, 2)


def _delta_pp(actual, pasado):
    """Diferencia en puntos porcentuales (para cosas que ya son %, como dominancias)."""
    if actual is None or pasado is None:
        return None
    return round(actual - pasado, 2)


def _comparativa(sub, col, escala=1.0):
    """Foto de una variable hoy y hace 2 semanas / 1 mes / 2 meses / 3 meses.

    En cristiano: para una variable (ej. el precio de ETH), devuelve su valor
    actual y los valores de hace 14, 30, 60 y 90 días, junto con cuánto ha
    cambiado en % respecto a cada uno. Así el modelo ve la TRAYECTORIA, no solo
    la foto de hoy. 'escala' sirve para pasar fracciones a % (las dominancias
    vienen como 0.51 y queremos 51).
    """
    actual = _hace(sub, col, 0)
    actual = actual * escala if actual is not None else None
    out = {"actual": round(actual, 4) if actual is not None else None, "historico": {}}
    for n in OFFSETS_COMPARATIVA:
        v = _hace(sub, col, n)
        v = v * escala if v is not None else None
        out["historico"][n] = {
            "valor": round(v, 4) if v is not None else None,
            "cambio_pct": _delta_pct(actual, v),
        }
    return out


def calcular_snapshot(fecha=None):
    """Construye la FOTO COMPLETA del mercado para el día del análisis.

    En cristiano: reúne en un solo diccionario TODO lo que el modelo debería
    saber del mercado hoy: precios y su evolución, posición en el ciclo (máximos
    históricos, halving), dominancias, sentimiento (miedo/codicia y su historia),
    indicadores técnicos, retornos, contexto macro y el régimen del HMM. Para las
    variables clave incluye además la comparativa temporal (hoy vs hace 2 sem / 1
    mes / 2 meses / 3 meses). Casi todo se LEE del df_model (ya calculado por tu
    pipeline); solo se calcula a mano lo que el CSV no trae (máximos históricos,
    halving, dominancias pasadas, distribución del sentimiento y el bloque macro).
    """
    sub = _df.loc[:fecha] if fecha else _df
    fila = sub.iloc[-1]
    fecha_ts = sub.index[-1]

    def g(c, d=2):
        """Lee una columna del último día (con redondeo y a prueba de huecos)."""
        return round(float(fila[c]), d) if c in fila and pd.notna(fila[c]) else None

    # ── Precios y ratio ───────────────────────────────────────────────
    precio_eth = g("eth_close"); precio_btc = g("btc_close")
    ratio = g("eth_btc_ratio", 5)
    if ratio is None and precio_eth and precio_btc:
        ratio = round(precio_eth / precio_btc, 5)

    # ── Máximos históricos (ATH) y posición en el ciclo ───────────────
    eth_ath = float(sub["eth_close"].max()); eth_ath_fecha = sub["eth_close"].idxmax()
    btc_ath = float(sub["btc_close"].max()); btc_ath_fecha = sub["btc_close"].idxmax()

    halvings = [pd.Timestamp("2012-11-28"), pd.Timestamp("2016-07-09"),
                pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-19")]
    halvings_pasados = [h for h in halvings if h <= fecha_ts]
    dias_post_halving = (fecha_ts - halvings_pasados[-1]).days if halvings_pasados else None

    ath_eth_ciclo = ath_btc_ciclo = None
    supera_eth = supera_btc = None
    if halvings_pasados:
        df_ciclo = sub.loc[halvings_pasados[-1]:fecha_ts]
        ath_eth_ciclo = round(float(df_ciclo["eth_close"].max()), 2)
        ath_btc_ciclo = round(float(df_ciclo["btc_close"].max()), 2)
        supera_eth = bool(ath_eth_ciclo >= eth_ath)
        supera_btc = bool(ath_btc_ciclo >= btc_ath)

    # Drawdown desde el máximo del último mes (complementa al drawdown desde ATH)
    eth_max_30d = float(sub["eth_close"].iloc[-30:].max()) if len(sub) >= 30 else None
    btc_max_30d = float(sub["btc_close"].iloc[-30:].max()) if len(sub) >= 30 else None
    dd_eth_mes = round((precio_eth / eth_max_30d - 1) * 100, 2) if (precio_eth and eth_max_30d) else None
    dd_btc_mes = round((precio_btc / btc_max_30d - 1) * 100, 2) if (precio_btc and btc_max_30d) else None

    # ── Sentimiento: racha, medias y distribución ─────────────────────
    fg_actual = int(fila["fear_greed"]) if pd.notna(fila["fear_greed"]) else None
    fg_serie = sub["fear_greed"]
    racha_miedo = 0
    for v in fg_serie.iloc[::-1]:
        if pd.notna(v) and v < 25: racha_miedo += 1
        else: break

    media_fg_mes = round(float(fg_serie.iloc[-30:].mean()), 1) if len(fg_serie) >= 30 else None
    media_fg_mes_ant = round(float(fg_serie.iloc[-60:-30].mean()), 1) if len(fg_serie) >= 60 else None

    def distribucion_fg(ventana):
        s = fg_serie.iloc[-ventana:]
        return {
            "miedo_extremo":   int((s < 25).sum()),
            "miedo":           int(((s >= 25) & (s < 45)).sum()),
            "neutral":         int(((s >= 45) & (s < 55)).sum()),
            "codicia":         int(((s >= 55) & (s < 75)).sum()),
            "codicia_extrema": int((s >= 75).sum()),
        }

    # ── Percentil de la volatilidad actual respecto al último año ─────
    vol_percentil = None
    if "eth_vol_30d" in sub.columns:
        serie_vol = sub["eth_vol_30d"].dropna()
        vol_hoy = g("eth_vol_30d")
        ventana_ano = serie_vol.iloc[-365:] if len(serie_vol) >= 365 else serie_vol
        if vol_hoy is not None and len(ventana_ano) > 0:
            vol_percentil = round((ventana_ano < vol_hoy).sum() / len(ventana_ano) * 100, 1)

    # ── Comparativas temporales (hoy vs 2sem/1mes/2meses/3meses) ──────
    comp_precio_eth = _comparativa(sub, "eth_close")
    comp_precio_btc = _comparativa(sub, "btc_close")
    comp_ratio      = _comparativa(sub, "eth_btc_ratio") if "eth_btc_ratio" in sub.columns else None
    comp_dom_btc    = _comparativa(sub, "btc_dominance", escala=100)
    comp_dom_eth    = _comparativa(sub, "eth_dominance", escala=100)
    comp_fg         = _comparativa(sub, "fear_greed")
    comp_inflacion  = _comparativa(sub, "inflation")
    comp_fed_rate   = _comparativa(sub, "fed_rate")

    # ── Bloque macro (si se pudo descargar) ───────────────────────────
    macro = None
    if _macro_diario is not None:
        m = _macro_diario.loc[:fecha_ts] if fecha else _macro_diario
        def gm(col, d=2):
            if col not in m.columns: return None
            v = m[col].iloc[-1]
            return round(float(v), d) if pd.notna(v) else None
        macro = {
            "fecha_datos": str(_macro_fecha.date()) if _macro_fecha is not None else None,
            "dias_retraso": (fecha_ts - _macro_fecha).days if _macro_fecha is not None else None,
            "dxy":    gm("dxy"),    "dxy_chg_30d":    _delta_pct(gm("dxy"),    _hace(m, "dxy", 30)),
            "oro":    gm("oro"),    "oro_chg_30d":    _delta_pct(gm("oro"),    _hace(m, "oro", 30)),
            "nasdaq": gm("nasdaq"), "nasdaq_chg_30d": _delta_pct(gm("nasdaq"), _hace(m, "nasdaq", 30)),
            "us10y":  gm("us10y"),  "us10y_chg_30d_pp": _delta_pp(gm("us10y"), _hace(m, "us10y", 30)),
            "usdjpy": gm("usdjpy"),
            **_macro_correls,
        }

    return {
        "fecha": fecha_ts.strftime("%Y-%m-%d"),
        "regimen": _regimen_actual(fecha),

        # Precios + comparativa temporal
        "precio_eth": precio_eth, "precio_btc": precio_btc, "ratio_eth_btc": ratio,
        "comp_precio_eth": comp_precio_eth, "comp_precio_btc": comp_precio_btc,
        "comp_ratio": comp_ratio,

        # Posición en el ciclo
        "drawdown_eth": g("eth_drawdown"), "drawdown_btc": g("btc_drawdown"),
        "drawdown_eth_mes": dd_eth_mes, "drawdown_btc_mes": dd_btc_mes,
        "ath_eth": round(eth_ath, 2), "ath_eth_fecha": str(eth_ath_fecha.date()),
        "dias_desde_ath_eth": (fecha_ts - eth_ath_fecha).days,
        "ath_btc": round(btc_ath, 2), "ath_btc_fecha": str(btc_ath_fecha.date()),
        "dias_desde_ath_btc": (fecha_ts - btc_ath_fecha).days,
        "ath_eth_ciclo": ath_eth_ciclo, "ath_eth_ciclo_supera": supera_eth,
        "ath_btc_ciclo": ath_btc_ciclo, "ath_btc_ciclo_supera": supera_btc,
        "dias_desde_halving": dias_post_halving,
        "dist_sma50": g("eth_dist_sma50"), "dist_sma200": g("eth_dist_sma200"),

        # Dominancias + comparativa temporal
        "dom_btc": round(g("btc_dominance") * 100, 2) if g("btc_dominance") is not None else None,
        "dom_eth": round(g("eth_dominance") * 100, 2) if g("eth_dominance") is not None else None,
        "dom_alt": round(g("alt_dominance") * 100, 2) if g("alt_dominance") is not None else None,
        "comp_dom_btc": comp_dom_btc, "comp_dom_eth": comp_dom_eth,
        "dom_btc_chg14d": g("btc_dominance_chg14d"), "dom_btc_chg30d": g("btc_dominance_chg30d"),
        "dom_eth_chg14d": g("eth_dominance_chg14d"), "dom_eth_chg30d": g("eth_dominance_chg30d"),

        # Sentimiento
        "fear_greed": fg_actual, "fear_greed_etiqueta": _etiqueta_fg(fg_actual),
        "comp_fear_greed": comp_fg,
        "racha_miedo_extremo": racha_miedo,
        "media_fg_mes": media_fg_mes, "media_fg_mes_anterior": media_fg_mes_ant,
        "fg_dist_14d": distribucion_fg(14), "fg_dist_30d": distribucion_fg(30),
        "n_miedo_ext_30d": g("n_miedo_ext_30d", 0), "n_codicia_30d": g("n_codicia_30d", 0),
        "presion_ext_neta_15d": g("presion_ext_neta_15d", 0),
        "presion_ext_neta_30d": g("presion_ext_neta_30d", 0),
        "fear_greed_scaled": g("fear_greed_scaled", 3),

        # Indicadores técnicos (leídos del df_model)
        "rsi_eth": g("eth_rsi14"), "rsi_btc": g("btc_rsi14"),
        "mfi_eth": g("eth_mfi14"), "mfi_btc": g("btc_mfi14"),
        "vol_eth_7d": g("eth_vol_7d"), "vol_eth_14d": g("eth_vol_14d"), "vol_eth_30d": g("eth_vol_30d"),
        "vol_eth_percentil_1y": vol_percentil,
        "stoch_d_eth": g("eth_stoch_d"), "stoch_d_btc": g("btc_stoch_d"),
        "bb_pctb": g("eth_bb_pctb", 3), "bb_width": g("eth_bb_width", 3),

        # Retornos (leídos del df_model + algún BTC calculado)
        "eth_ret_dia": g("eth_close_ret"),
        "eth_cum_ret_7d": g("eth_cum_ret_7d"), "eth_cum_ret_15d": g("eth_cum_ret_15d"),
        "eth_cum_ret_30d": g("eth_cum_ret_30d"),
        "btc_ret_7d": _delta_pct(precio_btc, _hace(sub, "btc_close", 7)),
        "btc_ret_30d": _delta_pct(precio_btc, _hace(sub, "btc_close", 30)),
        "eth_mcap_ret": g("eth_mcap_ret", 4), "btc_mcap_ret": g("btc_mcap_ret", 4),

        # Macro (inflación/tipos del df_model + bloque yfinance)
        "inflacion": g("inflation"), "inflacion_chg30": g("inflation_chg30"),
        "fed_rate": g("fed_rate"), "fed_rate_chg30": g("fed_rate_chg30"),
        "comp_inflacion": comp_inflacion, "comp_fed_rate": comp_fed_rate,
        "macro": macro,
    }


def _linea_comparativa(titulo, comp, sufijo="", dec=2):
    """Convierte una comparativa temporal en una línea de texto legible."""
    if not comp or comp.get("actual") is None:
        return None
    L = [f"  {titulo}: {round(comp['actual'], dec)}{sufijo} (hoy)"]
    etiquetas = {14: "hace 2 semanas", 30: "hace 1 mes", 60: "hace 2 meses", 90: "hace 3 meses"}
    for n in OFFSETS_COMPARATIVA:
        h = comp["historico"].get(n, {})
        if h.get("valor") is not None:
            chg = h.get("cambio_pct")
            chg_txt = f", {chg:+.2f}% desde entonces" if chg is not None else ""
            L.append(f"      {etiquetas[n]}: {round(h['valor'], dec)}{sufijo}{chg_txt}")
    return "\n".join(L)


def snapshot_a_texto(s):
    """Convierte la foto del mercado (el dict) en texto en castellano natural.

    En cristiano: traduce todos los números del snapshot a frases claras que el
    modelo de lenguaje pueda leer y razonar. Es lo que de verdad ve el LLM.
    """
    def f(v, suf="", dec=2, signo=False):
        if v is None: return "n/d"
        try:
            return (f"{v:+.{dec}f}" if signo else f"{round(v, dec)}") + suf
        except (TypeError, ValueError):
            return f"{v}{suf}"

    L = [f"FOTO DEL MERCADO — {s['fecha']}", ""]

    # PRECIOS Y TRAYECTORIA
    L.append("PRECIOS Y TRAYECTORIA:")
    L.append(f"  Precio ETH: {f(s['precio_eth'])} USD | Precio BTC: {f(s['precio_btc'])} USD | "
             f"Ratio ETH/BTC: {f(s['ratio_eth_btc'], dec=5)}")
    for titulo, comp, suf, dec in [
        ("Evolución precio ETH", s.get("comp_precio_eth"), " USD", 2),
        ("Evolución precio BTC", s.get("comp_precio_btc"), " USD", 2),
        ("Evolución ratio ETH/BTC", s.get("comp_ratio"), "", 5),
    ]:
        linea = _linea_comparativa(titulo, comp, suf, dec)
        if linea: L.append(linea)

    # CICLO / MÁXIMOS
    L.append("")
    L.append("POSICIÓN EN EL CICLO:")
    L.append(f"  Caída ETH desde su máximo histórico (drawdown): {f(s['drawdown_eth'], '%')} | "
             f"BTC: {f(s['drawdown_btc'], '%')}")
    L.append(f"  Caída ETH desde el máximo del último mes: {f(s['drawdown_eth_mes'], '%')} | "
             f"BTC: {f(s['drawdown_btc_mes'], '%')}")
    L.append(f"  Máximo histórico ETH: {f(s['ath_eth'])} USD el {s['ath_eth_fecha']} "
             f"(hace {s['dias_desde_ath_eth']} días)")
    L.append(f"  Máximo histórico BTC: {f(s['ath_btc'])} USD el {s['ath_btc_fecha']} "
             f"(hace {s['dias_desde_ath_btc']} días)")
    if s.get("ath_eth_ciclo") is not None:
        L.append(f"  Máximo ETH del ciclo actual (post-halving): {f(s['ath_eth_ciclo'])} USD "
                 f"(¿supera el ATH histórico? {'SÍ' if s['ath_eth_ciclo_supera'] else 'NO'})")
    if s.get("dias_desde_halving") is not None:
        L.append(f"  Días desde el último halving de Bitcoin: {s['dias_desde_halving']}")
    L.append(f"  Distancia a la media de 50 días: {f(s['dist_sma50'], '%')} | "
             f"a la de 200 días: {f(s['dist_sma200'], '%')}")

    # DOMINANCIAS
    L.append("")
    L.append("DOMINANCIAS DE MERCADO:")
    L.append(f"  BTC: {f(s['dom_btc'], '%')} | ETH: {f(s['dom_eth'], '%')} | "
             f"Resto (altcoins): {f(s['dom_alt'], '%')}")
    linea = _linea_comparativa("Evolución dominancia BTC", s.get("comp_dom_btc"), "%")
    if linea: L.append(linea)
    linea = _linea_comparativa("Evolución dominancia ETH", s.get("comp_dom_eth"), "%")
    if linea: L.append(linea)

    # SENTIMIENTO
    L.append("")
    L.append("SENTIMIENTO (MIEDO Y CODICIA):")
    L.append(f"  Índice actual: {f(s['fear_greed'], dec=0)} ({s['fear_greed_etiqueta']})")
    L.append(f"  Días seguidos en miedo extremo: {s['racha_miedo_extremo']}")
    L.append(f"  Media del último mes: {f(s['media_fg_mes'], dec=1)} | "
             f"mes anterior: {f(s['media_fg_mes_anterior'], dec=1)}")
    d14, d30 = s["fg_dist_14d"], s["fg_dist_30d"]
    L.append(f"  Reparto últimos 14 días → miedo extremo {d14['miedo_extremo']}, miedo {d14['miedo']}, "
             f"neutral {d14['neutral']}, codicia {d14['codicia']}, codicia extrema {d14['codicia_extrema']}")
    L.append(f"  Reparto últimos 30 días → miedo extremo {d30['miedo_extremo']}, miedo {d30['miedo']}, "
             f"neutral {d30['neutral']}, codicia {d30['codicia']}, codicia extrema {d30['codicia_extrema']}")
    linea = _linea_comparativa("Evolución del índice", s.get("comp_fear_greed"), "", 0)
    if linea: L.append(linea)

    # TÉCNICO
    L.append("")
    L.append("INDICADORES TÉCNICOS:")
    L.append(f"  RSI ETH (14d): {f(s['rsi_eth'])} | RSI BTC (14d): {f(s['rsi_btc'])}")
    L.append(f"  MFI ETH (14d): {f(s['mfi_eth'])} | MFI BTC (14d): {f(s['mfi_btc'])}")
    L.append(f"  Volatilidad ETH 7d: {f(s['vol_eth_7d'])} | 14d: {f(s['vol_eth_14d'])} | "
             f"30d: {f(s['vol_eth_30d'])} (percentil del último año: {f(s['vol_eth_percentil_1y'], '%', 1)})")
    L.append(f"  Estocástico %D ETH: {f(s['stoch_d_eth'])} | BTC: {f(s['stoch_d_btc'])}")
    L.append(f"  Bollinger %B ETH: {f(s['bb_pctb'], dec=3)} | anchura de bandas: {f(s['bb_width'], dec=3)}")

    # RETORNOS
    L.append("")
    L.append("RENDIMIENTOS RECIENTES:")
    L.append(f"  ETH hoy: {f(s['eth_ret_dia'], '%', signo=True)} | "
             f"acumulado 7d: {f(s['eth_cum_ret_7d'], '%', signo=True)} | "
             f"15d: {f(s['eth_cum_ret_15d'], '%', signo=True)} | "
             f"30d: {f(s['eth_cum_ret_30d'], '%', signo=True)}")
    L.append(f"  BTC 7d: {f(s['btc_ret_7d'], '%', signo=True)} | 30d: {f(s['btc_ret_30d'], '%', signo=True)}")

    # MACRO
    linea = _linea_comparativa("Evolución inflación EEUU", s.get("comp_inflacion"), "%")
    if linea: L.append(linea)
    linea = _linea_comparativa("Evolución tipos de interés EEUU", s.get("comp_fed_rate"), "%")
    if linea: L.append(linea)
    L.append("")
    L.append("CONTEXTO MACRO-FINANCIERO:")
    L.append(f"  Inflación interanual EEUU: {f(s['inflacion'], '%')} "
             f"(cambio 30d: {f(s['inflacion_chg30'], ' pp', signo=True)})")
    L.append(f"  Tipos de la Reserva Federal: {f(s['fed_rate'], '%')} "
             f"(cambio 30d: {f(s['fed_rate_chg30'], ' pp', signo=True)})")
    m = s.get("macro")
    if m:
        if m.get("dias_retraso") and m["dias_retraso"] > 0:
            L.append(f"  (Datos macro de mercado a fecha {m['fecha_datos']}, "
                     f"{m['dias_retraso']} día(s) por detrás del análisis cripto)")
        L.append(f"  Índice del dólar (DXY): {f(m.get('dxy'))} "
                 f"(30d: {f(m.get('dxy_chg_30d'), '%', signo=True)})")
        L.append(f"  Oro: {f(m.get('oro'))} USD/onza (30d: {f(m.get('oro_chg_30d'), '%', signo=True)})")
        L.append(f"  Nasdaq 100: {f(m.get('nasdaq'))} (30d: {f(m.get('nasdaq_chg_30d'), '%', signo=True)})")
        L.append(f"  Bono USA 10 años: {f(m.get('us10y'), '%')} "
                 f"(30d: {f(m.get('us10y_chg_30d_pp'), ' pp', signo=True)})")
        L.append(f"  Dólar/Yen (USD/JPY): {f(m.get('usdjpy'))}")
        if any(k in m for k in ("corr_btc_nasdaq_30d", "corr_btc_dxy_30d", "corr_btc_oro_30d")):
            L.append(f"  Correlación 30d de BTC → Nasdaq: {f(m.get('corr_btc_nasdaq_30d'))}, "
                     f"dólar: {f(m.get('corr_btc_dxy_30d'))}, oro: {f(m.get('corr_btc_oro_30d'))}")
    else:
        L.append("  (Bloque macro de mercado no disponible en este arranque)")

    # RÉGIMEN
    L.append("")
    L.append(f"RÉGIMEN DE MERCADO (HMM): {s['regimen']}")

    return "\n".join(L)


# ─── SEÑAL LSTM ──────────────────────────────────────────────────────────
def predecir_lstm():
    """Calcula la predicción de la LSTM para los próximos 3 días.

    En cristiano: carga la red entrenada, le da los últimos 30 días de datos y
    obtiene 3 retornos diarios estimados (uno por día). A partir de ahí calcula
    los precios estimados encadenando esos retornos (cada día parte del precio
    estimado del día anterior, NO del precio de hoy). Devuelve también el cambio
    acumulado a 3 días. RECORDATORIO: la dirección de esta señal es poco fiable.
    """
    ckpt = torch.load(DIR_MODELOS/"lstm_final.pt", map_location="cpu", weights_only=False)
    sc = joblib.load(DIR_MODELOS/"scaler_lstm.pkl")
    sx, sy = sc["sx"], sc["sy"]
    feats, regs = ckpt["feature_cols"], ckpt["regime_cols"]
    seq_len, arq = ckpt["seq_len"], ckpt["arquitectura"]
    modelo = LSTMRegressor(arq["n_features"], tuple(arq["hidden_sizes"]), arq["horizon"], arq["dropout"])
    modelo.load_state_dict(ckpt["state_dict"]); modelo.eval()
    ventana = _df.iloc[-seq_len:]
    Xc = sx.transform(ventana[feats].values)
    Xr = ventana[regs].values.astype(np.float32)
    X = torch.from_numpy(np.hstack([Xc, Xr]).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pred = modelo(X).numpy().ravel()
    retornos = sy.inverse_transform(pred.reshape(-1,1)).ravel()
    precio_hoy = float(ventana["eth_close"].iloc[-1]) if "eth_close" in ventana.columns else None
    precios = None
    if precio_hoy:
        precios, p = [], precio_hoy
        for r in retornos:
            p *= (1+r/100.0); precios.append(round(float(p),2))
    acum = (float(np.prod([1+r/100.0 for r in retornos]))-1)*100
    return {"fecha_base": str(ventana.index[-1].date()), "precio_hoy": precio_hoy,
            "horizonte_dias": int(ckpt["horizon"]),
            "retornos_diarios": [round(float(r),3) for r in retornos],
            "precios_estimados": precios, "retorno_acumulado": round(float(acum),3),
            "regimen_actual": _regimen_actual()}


def senal_a_texto(s):
    """Traduce la predicción de la LSTM a texto, recordando que es solo un apoyo."""
    if not s: return "Sin predicción de la LSTM."
    r = s["retornos_diarios"]; p = s.get("precios_estimados")
    t = f"""Predicción LSTM para {s['horizonte_dias']} días (base {s['fecha_base']}, {s['precio_hoy']} USD):
  - Retornos diarios (cada uno sobre el anterior): d1 {r[0]:+.2f}%, d2 {r[1]:+.2f}%, d3 {r[2]:+.2f}%"""
    if p:
        t += f"\n  - Precios estimados: d1≈{p[0]}, d2≈{p[1]}, d3≈{p[2]} USD"
    t += f"\n  - Variación acumulada 3d: {s['retorno_acumulado']:+.2f}%"
    t += """

CÓMO INTERPRETAR: la MAGNITUD es orientativa; la DIRECCIÓN es poco fiable (~50-52%, casi azar),
no la tomes como certeza. El RÉGIMEN del HMM puede ir con retraso: contrástalo con las noticias."""
    return t


# ─── NOTICIAS (muestreo ponderado por nota) ──────────────────────────────
def muestrear_noticias(n_eth=N_ETH, n_cripto=N_CRIPTO, n_macro=N_MACRO):
    """Elige unas pocas noticias por categoría, dando más probabilidad a las mejores.

    En cristiano: de todas las noticias del día, escoge unas 8 de Ethereum, 8 de
    cripto en general y 8 de macro. La elección es aleatoria pero PONDERADA por la
    nota (0-10): las noticias importantes salen más a menudo, pero cada pregunta
    puede traer noticias distintas, dando variedad sin perder calidad.
    """
    if _noticias_df is None or len(_noticias_df) == 0:
        return []
    elegidas = []
    for cat, n in [("eth", n_eth), ("cripto", n_cripto), ("macro", n_macro)]:
        sub = _noticias_df[_noticias_df["category"] == cat]
        if len(sub) == 0:
            continue
        pesos = sub["nota"].values + 0.1   # +0.1 para que incluso nota 0 tenga opción mínima
        k = min(n, len(sub))
        idx = np.random.choice(sub.index, size=k, replace=False, p=pesos/pesos.sum())
        elegidas += sub.loc[idx].to_dict("records")
    return elegidas


def noticias_a_texto(noticias):
    """Pone las noticias elegidas en una lista de texto para el prompt."""
    if not noticias:
        return "  (sin noticias)"
    return "\n".join(
        f"  [{n['date']}][{n['category']}] {n['source']}: {n['text']}" for n in noticias)


# ─── RETRIEVER EMBEBIDOS ─────────────────────────────────────────────────
def buscar_embebidos(pregunta, top_k=TOP_K_EMBEB):
    """Recupera los trozos de documento más parecidos a la pregunta.

    En cristiano: convierte la pregunta en números (embedding) y la compara con
    los trozos de los documentos largos para devolver los que más se parecen en
    significado. Así el modelo recibe justo el conocimiento que necesita.
    """
    if _emb_matrix is None or not _emb_chunks:
        return []
    q = np.array(embed_texts([pregunta], task_type="RETRIEVAL_QUERY")[0], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = _emb_matrix @ q
    idx = np.argsort(-sims)[:top_k]
    return [{"fuente": _emb_chunks[i]["fuente"], "text": _emb_chunks[i]["text"],
             "sim": float(sims[i])} for i in idx]


# ─── GENERACIÓN ──────────────────────────────────────────────────────────
def generar_respuesta(prompt, proveedor="gemini"):
    """Manda el prompt al modelo de lenguaje y devuelve su respuesta.

    En cristiano: envía todo el contexto montado al modelo (Gemini por defecto)
    y recoge el texto que genera. La opción de Claude está disponible pero
    inactiva (es de pago); se activa pasando proveedor="claude".
    """
    if proveedor == "gemini":
        return _client.models.generate_content(model=MODELO_LLM, contents=prompt).text
    elif proveedor == "claude":
        # Enchufable pero inactivo por defecto (API de pago).
        import anthropic
        cli = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        msg = cli.messages.create(model="claude-opus-4-8", max_tokens=2000,
            messages=[{"role":"user","content":prompt}])
        return msg.content[0].text
    raise ValueError(f"Proveedor desconocido: {proveedor}")


def construir_prompt(pregunta, snapshot, senal, embebidos, noticias, historial=""):
    """Monta el prompt final con instrucciones genéricas de proporcionalidad analítica."""
    emb_txt = "\n\n".join(f"  [{e['fuente']}] {e['text']}" for e in embebidos) or "  (sin documentos)"
    hist = f"\n<HISTORIAL>\n{historial}\n</HISTORIAL>\n" if historial else ""
    return f"""Eres un analista experto en mercados de criptomonedas, specialized en Ethereum.
Tu papel es裝 APOYAR LA DECISIÓN combinando señales cuantitativas (HMM + LSTM) con contexto
cualitativo (conocimiento experto + noticias). NO eres un oráculo: razonas con prudencia.

<CONOCIMIENTO_EXPERTO>
{_contexto_fijo}
</CONOCIMIENTO_EXPERTO>

<DOCUMENTOS_RECUPERADOS>
{emb_txt}
</DOCUMENTOS_RECUPERADOS>

<DATOS_DEL_DIA>
{snapshot_a_texto(snapshot)}
</DATOS_DEL_DIA>

<SENAL_LSTM>
{senal_a_texto(senal)}
</SENAL_LSTM>

<NOTICIAS>
{noticias_a_texto(noticias)}
</NOTICIAS>
{hist}
<PREGUNTA>
{pregunta}
</PREGUNTA>

INSTRUCCIONES DE RAZONAMIENTO Y RESPUESTA:
Lo primero, evalúa el ALCANCE y el FONDO de la <PREGUNTA> para responder con estricta proporcionalidad. Ve al grano, sé elástico y evita respuestas clónicas.

· CHARLA TRIVIAL o saludos ("buenos días", "¿qué tal?"): 
  Responde con brevedad y cortesía humana. Sin datos ni análisis.

· PREGUNTAS TEÓRICAS o de concepto ("¿qué es el staking?"): 
  Explica el funcionamiento y el concepto de forma didáctica. No uses el estado actual del mercado.

· PREGUNTAS CON ENFOQUE FINANCIERO, DE RIESGO O MERCADO:
  Aplica el marco mental del <MOTOR_DE_RAZONAMIENTO> (lenguaje probabilístico, uso del glosario descriptivo y separación de horizontes) pero ADAPTANDO la extensión y los bloques al fondo exacto de la consulta:
  
  1. PRINCIPIO DE RELEVANCIA TEMÁTICA: Responde exclusivamente sobre el tema solicitado. Si te preguntan por "factores macroeconómicos", limítate estrictamente a variables macro (Fed, inflación, bonos, Nasdaq, dólar, geopolítica). Está prohibido meter ruido de sentimiento cripto (como el índice de miedo y codicia o rachas de pánico) en respuestas puramente macroeconómicas.
  
  2. PRINCIPIO DE PROPORCIONALIDAD ESTRUCTURAL: La estructura rígida de 9 apartados (Bloque 14 del motor) y los Escenarios Probabilísticos quedan reservados ÚNICAMENTE para preguntas de horizonte amplio, estrategia temporal o planificación de carteras (ej: "¿qué hacer de aquí a final de año?", "haz un informe general"). Si la pregunta pide un aspecto acotado (ej: "¿qué riesgos hay?", "¿qué factores macro afectan?"), no uses la plantilla general; genera subtítulos dinámicos directamente relacionados con tu respuesta y ve al grano desde la primera línea.

En todos los casos: responde en español, responde unicamente con la información del contexto no busques en otras fuentes, usa SOLO los datos relevantes para esa pregunta (no vuelques todo el contexto), y sé honesto con la incertidumbre.

Está TOTALMENTE PROHIBIDO que incluyas etiquetas HTML (como <div>, <span>, <svg>, etc.) o clases CSS (como ef-response-wrapper, ef-rhead) en tu respuesta. No intentes imitar la estructura visual que veas en el historial. Tu salida debe ser Markdown limpio estándar."""

def responder(pregunta, proveedor="gemini", historial=""):
    """Función principal que llama la app: de la pregunta a la respuesta.

    En cristiano: orquesta todo el proceso. Saca la foto del mercado, pide la
    señal de la LSTM, recupera documentos y noticias, monta el prompt, se lo
    manda al modelo y devuelve la respuesta junto con las fuentes usadas.
    """
    snap = calcular_snapshot()
    try:
        senal = predecir_lstm()
    except Exception as e:
        senal = None; print(f"⚠️ señal LSTM: {e}")
    embebidos = buscar_embebidos(pregunta)
    noticias = muestrear_noticias()
    prompt = construir_prompt(pregunta, snap, senal, embebidos, noticias, historial)
    try:
        respuesta = generar_respuesta(prompt, proveedor=proveedor)
    except Exception as e:
        respuesta = f"❌ ERROR generando respuesta: {e}"
    fuentes = {"noticias": noticias, "embebidos": embebidos, "snapshot": snap, "senal": senal}
    return respuesta, fuentes


# Inicializar al importar.
# Va dentro de un try para que IMPORTAR rag nunca tumbe la app: si algo falla
# al arrancar (cuota de Google, un CSV que no carga, etc.), la app se levanta
# igual y el problema se ve al preguntar, no como una pantalla en negro.
try:
    inicializar()
except Exception as e:
    print(f"⚠️ Error durante la inicialización del RAG: {e}")
    print("   La app arrancará igualmente; algunas funciones pueden no estar disponibles.")