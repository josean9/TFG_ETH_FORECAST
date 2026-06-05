"""
Actualización diaria de noticias para el sistema RAG.

Descarga noticias de múltiples fuentes (RSS sin clave + APIs con clave), las deduplica,
filtra por calidad, las clasifica en categorías (eth / cripto / macro) y calcula una
NOTA DE RELEVANCIA 0-10 por noticia (recencia + fiabilidad de fuente + palabras clave).
Guarda TODAS las que pasan el filtro en data/news/noticias.csv (sobrescribe el del día
anterior → sin acumular, sin duplicados).

La APP, al responder, lee este CSV y hace un muestreo ponderado por la nota (8+8+8),
de modo que cada pregunta puede traer noticias algo distintas, priorizando las de nota alta.

Pensado para ejecutarse a diario (cron / GitHub Actions, en su propio workflow).
"""

import os
import re
import time
import hashlib
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
import feedparser
import pandas as pd

try:
    from palabras_clave import PALABRAS_CLAVE, TOPE_RELEVANCIA
except ImportError:
    # fallback mínimo si no se encuentra el archivo
    PALABRAS_CLAVE, TOPE_RELEVANCIA = {"ethereum": 2, "bitcoin": 2, "crypto": 1}, 12

# ─── RUTAS ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent      # src/data/ -> raíz
RUTA_SALIDA = BASE_DIR / "data" / "news" / "noticias.csv"

# ─── CLAVES DE API (desde entorno / .env) ─────────────────────────────────────
NEWSDATA_KEY = os.environ.get("NEWSDATA_KEY", "")
# (añade aquí otras claves si usas más APIs)

# ─── FUENTES RSS (sin clave) ──────────────────────────────────────────────────
RSS_FEEDS = {
    "CoinDesk":        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph":   "https://cointelegraph.com/rss",
    "Decrypt":         "https://decrypt.co/feed",
    "CryptoSlate":     "https://cryptoslate.com/feed/",
    "Bitcoinist":      "https://bitcoinist.com/feed/",
    "UToday":          "https://u.today/rss",
    "CoinJournal":     "https://coinjournal.net/news/feed/",
    "BitcoinMagazine": "https://bitcoinmagazine.com/feed",
    "TheBlock":        "https://www.theblock.co/rss.xml",
    "YahooFinance":    "https://finance.yahoo.com/news/rssindex",
    "MarketWatch":     "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "CNBCWorld":       "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "Investing":       "https://www.investing.com/rss/news_25.rss",
}

# ─── FIABILIDAD POR FUENTE (0-1, contribuye a la nota) ────────────────────────
FIABILIDAD_FUENTE = {
    "CoinDesk": 1.0, "Cointelegraph": 0.9, "TheBlock": 1.0, "Decrypt": 0.85,
    "BitcoinMagazine": 0.85, "CryptoSlate": 0.7, "Bitcoinist": 0.6, "UToday": 0.6,
    "CoinJournal": 0.7, "YahooFinance": 0.9, "MarketWatch": 0.95, "CNBCWorld": 0.95,
    "Investing": 0.85,
}
FIABILIDAD_DEFECTO = 0.6   # para fuentes de NewsData u otras no listadas

# ─── FILTROS DE CALIDAD ───────────────────────────────────────────────────────
FUENTES_BLACKLIST = {"openpr", "techbullion", "globenewswire", "prnewswire", "businesswire"}
SPAM = ["price prediction", "huge gains", "moonshot", "100x", "1000x", "before listing",
        "presale", "millionaire", "to the moon", "next big thing", "10x potential",
        "buy now", "get rich", "guaranteed"]
MEMES = ["pepeto", "shiba", "dogecoin", "pepe", "bonk", "wif", "floki"]

MACRO_KW = ["fed", "powell", "trump", "tariff", "inflation", "interest rate", "treasury",
            "regulation", "sec ", "federal reserve", "war", "china", "iran", "cpi",
            "recession", "economy", "fomc", "rate cut", "rate hike"]
ETH_KW = ["ethereum", "eth ", " eth", "vitalik", "merge", "staking", "layer 2", "rollup"]


def categoria(texto_lower):
    if any(k in texto_lower for k in ETH_KW):
        return "eth"
    if any(k in texto_lower for k in MACRO_KW):
        return "macro"
    return "cripto"   # resto de cripto (btc, altcoins, mercado general)


def pasa_calidad(noticia):
    tl, fl = noticia["text"].lower(), noticia["source"].lower()
    if fl in FUENTES_BLACKLIST:
        return False
    if any(s in tl for s in SPAM):
        return False
    if sum(1 for m in MEMES if m in tl) >= 1 and "bitcoin" not in tl and "ethereum" not in tl:
        return False
    if len(noticia["text"]) < 40:        # demasiado corta = poco informativa
        return False
    return True


def nota_relevancia(noticia, ahora):
    """Nota 0-10. Las palabras clave (relevancia de mercado) son decisivas:
    una noticia sin ningún término relevante se hunde, aunque sea reciente y de fuente fiable.
      - palabras clave: 0-6 (peso dominante)
      - recencia:       0-2.5
      - fiabilidad:     0-1.5
    """
    # 1) Palabras clave (0-6): suma de pesos del léxico, normalizada
    tl = noticia["text"].lower()
    suma = sum(peso for palabra, peso in PALABRAS_CLAVE.items() if palabra in tl)
    kw = min(suma / TOPE_RELEVANCIA, 1.0) * 6.0

    # Si no menciona NADA relevante, es ruido: nota muy baja (solo un resto por recencia)
    if suma == 0:
        return round(min(kw + 0.0, 10.0), 2)   # kw=0 -> nota 0

    # 2) Recencia (0-2.5): hoy=2.5, baja ~0.35/día
    try:
        dias = max((ahora - pd.Timestamp(noticia["date"])).days, 0)
    except Exception:
        dias = 3
    rec = max(2.5 - 0.35 * dias, 0.0)

    # 3) Fiabilidad de fuente (0-1.5)
    fia = FIABILIDAD_FUENTE.get(noticia["source"], FIABILIDAD_DEFECTO) * 1.5

    return round(min(kw + rec + fia, 10.0), 2)


# ─── DESCARGA ─────────────────────────────────────────────────────────────────
def descargar_rss(max_por_feed=40, max_dias=7):
    docs, ahora = [], pd.Timestamp.now()
    for nombre, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
        except Exception:
            continue
        for e in feed.entries[:max_por_feed]:
            titulo = (e.get("title", "") or "").strip()
            resumen = re.sub(r"<[^>]+>", "", e.get("summary", "") or e.get("description", "") or "")
            resumen = re.sub(r"\s+", " ", resumen).strip()
            fecha_raw = e.get("published", "") or e.get("updated", "")
            try:
                fdt = pd.Timestamp(parsedate_to_datetime(fecha_raw)).tz_localize(None)
                if (ahora - fdt).days > max_dias:
                    continue
                fecha = fdt.strftime("%Y-%m-%d")
            except Exception:
                fecha = ahora.strftime("%Y-%m-%d")
            texto = f"{titulo}. {resumen}"[:500].strip()
            if not titulo:
                continue
            docs.append({"text": texto, "title": titulo, "date": fecha, "source": nombre})
    return docs


def descargar_newsdata(query, paginas=2):
    if not NEWSDATA_KEY:
        return []
    docs, next_page, ahora = [], None, pd.Timestamp.now().strftime("%Y-%m-%d")
    for _ in range(paginas):
        params = {"apikey": NEWSDATA_KEY, "q": query, "language": "en"}
        if next_page:
            params["page"] = next_page
        try:
            data = requests.get("https://newsdata.io/api/1/latest", params=params, timeout=20).json()
        except Exception:
            break
        if data.get("status") != "success":
            break
        for art in data.get("results", []):
            titulo = (art.get("title", "") or "").strip()
            if not titulo or titulo == "[Removed]":
                continue
            desc = art.get("description", "") or ""
            texto = f"{titulo}. {desc}"[:500].strip()
            docs.append({"text": texto, "title": titulo,
                         "date": (art.get("pubDate", "") or "")[:10] or ahora,
                         "source": art.get("source_id", "newsdata")})
        next_page = data.get("nextPage")
        if not next_page:
            break
        time.sleep(1)
    return docs


# ─── DEDUPLICACIÓN ────────────────────────────────────────────────────────────
def _norm_titulo(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

def deduplicar(noticias):
    vistos_id, vistos_tit, out = set(), set(), []
    for n in noticias:
        nid = hashlib.md5(n["text"][:120].encode()).hexdigest()
        tit = _norm_titulo(n["title"])[:80]
        if nid in vistos_id or (tit and tit in vistos_tit):
            continue
        vistos_id.add(nid); vistos_tit.add(tit)
        out.append(n)
    return out


def main():
    print("=" * 60)
    print("Actualización de noticias para el RAG")
    print("=" * 60)
    ahora = pd.Timestamp.now()

    # 1) Descargar de todas las fuentes
    rss = descargar_rss()
    print(f"  RSS: {len(rss)} noticias")
    nd = []
    if NEWSDATA_KEY:
        nd = descargar_newsdata("bitcoin OR ethereum OR crypto") + \
             descargar_newsdata("federal reserve OR inflation OR tariff")
        print(f"  NewsData: {len(nd)} noticias")
    else:
        print("  NewsData: (sin clave, omitido)")

    todas = rss + nd
    print(f"  Total bruto: {len(todas)}")

    # 2) Deduplicar
    todas = deduplicar(todas)
    print(f"  Tras deduplicar: {len(todas)}")

    # 3) Filtrar calidad + clasificar + puntuar
    filtradas = []
    for n in todas:
        if not pasa_calidad(n):
            continue
        n["category"] = categoria(n["text"].lower())
        n["nota"] = nota_relevancia(n, ahora)
        filtradas.append(n)
    print(f"  Tras filtro de calidad: {len(filtradas)}")

    if not filtradas:
        print("⚠️  No quedaron noticias tras el filtro. No se sobrescribe el CSV.")
        return

    # 4) Guardar (ordenadas por nota desc) — sobrescribe
    df = pd.DataFrame(filtradas)[["date", "category", "nota", "source", "title", "text"]]
    df = df.sort_values("nota", ascending=False).reset_index(drop=True)
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_SALIDA, index=False)

    print(f"\n✓ Guardado en {RUTA_SALIDA}")
    print(f"  Total: {len(df)}  |  eth={sum(df.category=='eth')}, "
          f"cripto={sum(df.category=='cripto')}, macro={sum(df.category=='macro')}")
    print(f"  Nota media: {df.nota.mean():.2f}  |  máx: {df.nota.max()}  mín: {df.nota.min()}")
    print("=" * 60)


if __name__ == "__main__":
    main()