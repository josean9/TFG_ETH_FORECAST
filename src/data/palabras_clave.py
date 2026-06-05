"""
Léxico financiero ponderado para puntuar la relevancia de mercado de las noticias.

Cada palabra/expresión tiene un peso según su impacto potencial en el precio:
  - PESO 3 (alto impacto): eventos que mueven el mercado con fuerza.
  - PESO 2 (impacto medio): temas relevantes y recurrentes.
  - PESO 1 (contexto): términos de fondo, frecuentes pero menos decisivos.

La nota de relevancia de una noticia = suma de pesos de las palabras clave que contiene
(luego se normaliza a 0-10 en el script). Si no contiene ninguna, su relevancia es ~0.

Edita libremente: añade, quita o cambia de nivel según tu criterio.
Las comparaciones se hacen en minúsculas y sobre el texto completo (título + resumen).
"""

PALABRAS_CLAVE = {
    # ── PESO 3 — alto impacto (eventos que mueven el mercado) ──
    "etf": 3,
    "spot etf": 3,
    "sec": 3,
    "approval": 3,
    "approved": 3,
    "halving": 3,
    "hack": 3,
    "exploit": 3,
    "lawsuit": 3,
    "ban": 3,
    "regulation": 3,
    "interest rate": 3,
    "rate cut": 3,
    "rate hike": 3,
    "fed": 3,
    "powell": 3,
    "fomc": 3,
    "inflation": 3,
    "cpi": 3,
    "all-time high": 3,
    "ath": 3,
    "crash": 3,
    "liquidation": 3,
    "default": 3,
    "bankruptcy": 3,

    # ── PESO 2 — impacto medio (temas relevantes y recurrentes) ──
    "ethereum": 2,
    "eth": 2,
    "bitcoin": 2,
    "btc": 2,
    "merge": 2,
    "staking": 2,
    "upgrade": 2,
    "fork": 2,
    "layer 2": 2,
    "rollup": 2,
    "gas fees": 2,
    "whale": 2,
    "institutional": 2,
    "blackrock": 2,
    "grayscale": 2,
    "treasury": 2,
    "tariff": 2,
    "recession": 2,
    "bull market": 2,
    "bear market": 2,
    "rally": 2,
    "sell-off": 2,
    "selloff": 2,
    "volatility": 2,
    "adoption": 2,
    "futures": 2,
    "options": 2,
    "open interest": 2,
    "stablecoin": 2,
    "defi": 2,
    "trump": 2,
    "vitalik": 2,

    # ── PESO 1 — contexto (frecuentes, menos decisivos) ──
    "crypto": 1,
    "blockchain": 1,
    "altcoin": 1,
    "market": 1,
    "price": 1,
    "trading": 1,
    "exchange": 1,
    "binance": 1,
    "coinbase": 1,
    "wallet": 1,
    "mining": 1,
    "nft": 1,
    "web3": 1,
    "token": 1,
    "dollar": 1,
    "gold": 1,
    "nasdaq": 1,
    "stocks": 1,
    "economy": 1,
    "investors": 1,
    "bullish": 1,
    "bearish": 1,
    "support": 1,
    "resistance": 1,
    "forecast": 1,
}

# Peso máximo teórico de referencia para normalizar a 0-10 (ajustable).
# No es la suma de todo: es un tope razonable a partir del cual se considera "muy relevante".
TOPE_RELEVANCIA = 12