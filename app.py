"""
App de chat (Gradio) para el sistema de apoyo a la decisión sobre Ethereum.
"EtherForecast AI" — dashboard de tres columnas.

Disposición (como en la maqueta):
  ┌──────────────┬───────────────────────────┬──────────────────┐
  │ Conversaciones│      EtherForecast AI      │ Estado del mercado│
  │  (lateral)    │  título · ejemplos · chat  │  (panel derecho) │
  └──────────────┴───────────────────────────┴──────────────────┘

Notas de diseño:
  - La columna de estado del mercado queda pegada a la derecha del todo.
  - La lateral de conversaciones es visual; "Nueva conversación" sí funciona
    (vacía el chat).
  - Iconos de los ejemplos de colores; cada uno con su tinte.
  - Respuesta del bot maquetada como ficha: cabecera con el sesgo direccional y
    el régimen, "Resumen de la predicción", y secciones con icono (Contexto RAG,
    Señal LSTM, Noticias).
  - Barra de scroll de la conversación oculta (sigue funcionando con rueda).
  - Caja de texto con el botón de enviar integrado de Gradio (flecha centrada,
    sin borde, sin clip).

Compatible con Gradio 6.16.0 (theme/css en launch(); submit_btn del Textbox
dispara el evento .submit() al hacer clic; SVG en el mensaje con
sanitize_html=False y el texto del modelo escapado).

Despliegue: Hugging Face Spaces. Secrets: GEMINI_API_KEY, NEWSDATA_KEY.
"""

import os
import html
import gradio as gr
import rag

# ─── CONFIG ───────────────────────────────────────────────────────────────
PROVEEDOR = "gemini"
MAX_TURNOS_MEMORIA = 5
LSTM_ACCURACY_PCT = 51.3
LSTM_HORIZONTE_DIAS = 3

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Paleta de acento
COL = {
    "blue": "#5b9bff", "green": "#16c784", "cyan": "#22d3ee",
    "purple": "#a78bfa", "amber": "#f0b90b", "red": "#ff6b5b",
}

# (clave_icono, color, pregunta)
EJEMPLOS = [
    ("prob",  "blue",   "¿Cuál es la probabilidad de subida de ETH esta semana?"),
    ("sent",  "green",  "Analiza el sentimiento actual del mercado."),
    ("news",  "cyan",   "Resume las noticias más relevantes de las últimas 24 horas."),
    ("macro", "purple", "¿Qué factores macroeconómicos afectan ahora a Ethereum?"),
    ("ciclo", "amber",  "Explica el contexto actual del ciclo de mercado."),
    ("risk",  "red",    "¿Qué riesgos existen ahora mismo?"),
]


# ─── ICONOS (SVG de línea estilo lucide) ──────────────────────────────────
PATHS = {
    "prob":     '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "sent":     '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/>'
                '<line x1="9" x2="9.01" y1="9" y2="9"/><line x1="15" x2="15.01" y1="9" y2="9"/>',
    "news":     '<path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Z"/>'
                '<path d="M4 22a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/>'
                '<path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8Z"/>',
    "macro":    '<circle cx="12" cy="12" r="10"/>'
                '<path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
    "ciclo":    '<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/>'
                '<line x1="6" x2="6" y1="20" y2="14"/>',
    "risk":     '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6'
                'a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5'
                'a1 1 0 0 1 1 1z"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "history":  '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
                '<path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "resumen":  '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
                '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>',
    "up":       '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "down":     '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "neu":      '<line x1="5" x2="19" y1="12" y2="12"/>',
    "search":   '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "info":     '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "sliders":  '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/>'
                '<line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/>'
                '<line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/>'
                '<line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/>'
                '<line x1="18" x2="22" y1="16" y2="16"/>',
    "user":     '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
}


def _svg(body, stroke="currentColor", size=18, sw=1.9):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


# Iconos en línea (heredan color por CSS) para dentro de los mensajes y la lateral.
ICONO = {k: _svg(v) for k, v in PATHS.items()}

ETH_INLINE = ('<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" '
              'fill="currentColor"><path d="M12 2 5.5 12.2 12 16l6.5-3.8L12 2Z" opacity=".8"/>'
              '<path d="M12 17.2 5.5 13.4 12 22l6.5-8.6L12 17.2Z" opacity=".5"/></svg>')

ETH_AVATAR_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 417">'
    '<path fill="#8a93c4" d="M127.96 0l-2.79 9.5v275.668l2.79 2.785 127.962-75.638z"/>'
    '<path fill="#c3cdf0" d="M127.96 0L0 212.32l127.96 75.638V154.158z"/>'
    '<path fill="#8a93c4" d="M127.96 312.187l-1.575 1.92v98.199l1.575 4.6L256 236.587z"/>'
    '<path fill="#c3cdf0" d="M127.96 416.905v-104.72L0 236.585z"/>'
    '<path fill="#62699a" d="M127.96 287.958l127.96-75.637-127.96-58.162z"/>'
    '<path fill="#8a93c4" d="M0 212.32l127.96 75.638v-133.8z"/></svg>'
)


def _preparar_iconos():
    """Escribe a /assets los SVG que Gradio necesita por ruta (iconos de botón y avatar)."""
    os.makedirs(ASSETS, exist_ok=True)
    for clave, color, _ in EJEMPLOS:
        ruta = os.path.join(ASSETS, f"ic_{clave}.svg")
        with open(ruta, "w", encoding="utf-8") as f:   # reescribe por si cambia el color
            f.write(_svg(PATHS[clave], stroke=COL[color], size=20, sw=1.8))
    ruta_eth = os.path.join(ASSETS, "eth.svg")
    if not os.path.exists(ruta_eth):
        with open(ruta_eth, "w", encoding="utf-8") as f:
            f.write(ETH_AVATAR_SVG)


def _icono_ejemplo(clave):
    return os.path.join(ASSETS, f"ic_{clave}.svg")


def _ruta_avatar():
    return os.path.join(ASSETS, "eth.svg")


# ─── HISTORIAL → TEXTO ─────────────────────────────────────────────────────
def _a_texto(contenido):
    if contenido is None:
        return ""
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, dict):
        return contenido.get("text", "") or ""
    if isinstance(contenido, (list, tuple)):
        partes = []
        for bloque in contenido:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict):
                partes.append(bloque.get("text", "") or "")
            elif isinstance(bloque, (list, tuple)):
                partes.append(_a_texto(bloque))
        return " ".join(p for p in partes if p)
    return str(contenido)


def _construir_historial(history):
    if not history:
        return ""
    recientes = history[-(MAX_TURNOS_MEMORIA * 2):]
    lineas = []
    for turno in recientes:
        if isinstance(turno, dict):
            rol = "Usuario" if turno.get("role") == "user" else "Asistente"
            lineas.append(f"{rol}: {_a_texto(turno.get('content', ''))}")
    return "\n".join(lineas)


# ─── MAQUETADO DE LA RESPUESTA ─────────────────────────────────────────────
def _escapar_cuerpo(t):
    """Neutraliza HTML del texto del modelo sin tocar el Markdown."""
    if not isinstance(t, str):
        t = str(t or "")
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _seccion(icono, titulo, cuerpo_html, color):
    return ("<div class='ef-rsec'>"
            f"<div class='ef-rsec-h ic-{color}'><span class='ef-rsec-ico'>{ICONO[icono]}</span>"
            f"{html.escape(titulo)}</div>"
            f"<div class='ef-rsec-b'>{cuerpo_html}</div></div>")


def _chips_noticias(noticias):
    vistos = []
    for n in noticias:
        src = None
        if isinstance(n, dict):
            for k in ("fuente", "source", "medio", "feed", "origen"):
                if n.get(k):
                    src = str(n[k]).strip()
                    break
        elif isinstance(n, str):
            src = n.strip()
        if src and src not in vistos:
            vistos.append(src)
    vistos = vistos[:8]
    if not vistos:
        return f"{len(noticias)} titulares incorporados al contexto de esta respuesta."
    chips = "".join(f"<span class='ef-src'>{html.escape(s)}</span>" for s in vistos)
    return f"<div class='ef-srcwrap'>{chips}</div>"


def _sesgo_direccional(senal):
    """Sesgo a 3 días según el signo de la señal del LSTM (honesto: 'sesgo', no certeza)."""
    if not senal:
        return ("Neutral", "neu")
    r = senal.get("retornos_diarios", []) or []
    if not r:
        return ("Neutral", "neu")
    sub = r[:3]
    m = sum(sub) / max(len(sub), 1)
    if m > 0.05:
        return ("Alcista", "up")
    if m < -0.05:
        return ("Bajista", "down")
    return ("Neutral", "neu")


def construir_respuesta(mensaje, historial_txt):
    """Llama al RAG y devuelve la respuesta maquetada (cabecera + cuerpo MD + pie)."""
    respuesta, fuentes = rag.responder(mensaje, proveedor=PROVEEDOR, historial=historial_txt)
    snap = fuentes.get("snapshot", {}) or {}
    senal = fuentes.get("senal")
    noticias = fuentes.get("noticias", []) or []
    embebidos = fuentes.get("embebidos", []) or []
    regimen = snap.get("regimen", "—")

    # Sesgo direccional + confianza del régimen (si rag la expone; si no, el nombre).
    direccion, cls = _sesgo_direccional(senal)
    prob = None
    for k in ("regimen_prob", "prob_regimen", "confianza_regimen", "prob_estado"):
        if snap.get(k) is not None:
            prob = snap.get(k)
            break
    if prob is not None:
        try:
            p = float(prob)
            p = p * 100 if p <= 1 else p
            conf_label, conf_val = "Confianza del régimen", f"{p:.0f}%"
        except (TypeError, ValueError):
            conf_label, conf_val = "Régimen (HMM)", str(regimen)
    else:
        conf_label, conf_val = "Régimen (HMM)", str(regimen)

    cabecera = (
        "<div class='ef-rhead'>"
        f"<span class='ef-rhead-eth'>{ETH_INLINE}</span>"
        "<div class='ef-rhead-main'>"
        "<span class='ef-rhead-label'>Sesgo direccional · 3 días</span>"
        f"<span class='ef-dir ef-dir-{cls}'>{ICONO[cls]}{direccion}</span>"
        "</div>"
        "<div class='ef-rhead-conf'>"
        f"<span class='ef-conf-label'>{html.escape(conf_label)}</span>"
        f"<span class='ef-conf-val'>{html.escape(conf_val)}</span>"
        "</div></div>"
        f"<div class='ef-rsec-h ic-blue ef-resumen-h'><span class='ef-rsec-ico'>{ICONO['resumen']}</span>"
        "Resumen de la predicción</div>"
    )

    cuerpo = _escapar_cuerpo(respuesta)

    secciones = [_seccion(
        "history", "Contexto histórico (RAG)",
        f"{len(embebidos)} fragmentos del corpus y {len(noticias)} noticias en el contexto de esta respuesta.",
        "blue")]
    if senal:
        r = senal.get("retornos_diarios", [0, 0, 0])
        senal_txt = " / ".join(f"{x:+.1f}%" for x in r[:3])
        secciones.append(_seccion(
            "activity", "Señal del modelo LSTM",
            f"{senal_txt} a 3 días. "
            f"<span class='ef-warn'>Dirección poco fiable (≈ azar · accuracy {LSTM_ACCURACY_PCT}%).</span>",
            "amber"))
    if noticias:
        secciones.append(_seccion("news", f"Noticias utilizadas ({len(noticias)})",
                                  _chips_noticias(noticias), "teal"))

    pie = "<div class='ef-rfoot'>" + "".join(secciones) + "</div>"
    return f"{cabecera}\n\n{cuerpo}\n\n{pie}"


# ─── EVENTOS DEL CHAT ──────────────────────────────────────────────────────
def agregar_usuario(mensaje, history):
    if not mensaje or not mensaje.strip():
        return history, ""
    history = (history or []) + [{"role": "user", "content": mensaje.strip()}]
    return history, ""


def responder_bot(history):
    if not history or history[-1].get("role") != "user":
        return history
    mensaje = _a_texto(history[-1].get("content", ""))
    historial_txt = _construir_historial(history[:-1])
    try:
        texto = construir_respuesta(mensaje, historial_txt)
    except Exception as e:
        texto = f"❌ Ha ocurrido un error generando la respuesta: {e}"
    return history + [{"role": "assistant", "content": texto}]


def nueva_conversacion():
    return []


# ─── PANEL DERECHO ─────────────────────────────────────────────────────────
def _fmt_grande(n):
    if n is None:
        return "n/d"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "n/d"
    for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(n) >= div:
            return f"${n/div:.1f}{suf}"
    return f"${n:,.0f}"


def _color_cambio(v):
    return "#16c784" if (v is not None and v >= 0) else "#ea3943"


def _len_seguro(x):
    if x is None:
        return 0
    try:
        return len(x)
    except Exception:
        return 0


def _datos_panel():
    d = {"precio": None, "cambio": None, "fg": None, "fg_etq": "n/d",
         "vol": None, "vol_pct": None, "dom_btc": None, "dom_eth": None,
         "regimen": "n/d", "drawdown": None, "fecha": "n/d",
         "mcap": None, "volumen": None, "noticias": 0, "docs": 0}
    try:
        s = rag.calcular_snapshot()
        d.update({
            "precio": s.get("precio_eth"), "cambio": s.get("eth_ret_dia"),
            "fg": s.get("fear_greed"), "fg_etq": s.get("fear_greed_etiqueta", "n/d"),
            "vol": s.get("vol_eth_30d"), "vol_pct": s.get("vol_eth_percentil_1y"),
            "dom_btc": s.get("dom_btc"), "dom_eth": s.get("dom_eth"),
            "regimen": s.get("regimen", "n/d"), "drawdown": s.get("drawdown_eth"),
            "fecha": s.get("fecha", "n/d"),
        })
    except Exception as e:
        print(f"⚠️ No se pudo construir el panel de datos: {e}")
    try:
        fila = rag._df.iloc[-1]
        d["mcap"] = float(fila["eth_mcap"]) if "eth_mcap" in fila else None
        d["volumen"] = float(fila["eth_volume"]) if "eth_volume" in fila else None
    except Exception:
        pass
    d["noticias"] = _len_seguro(getattr(rag, "_noticias_df", None))
    d["docs"] = _len_seguro(getattr(rag, "_emb_chunks", None))
    return d


def construir_panel_html():
    d = _datos_panel()
    precio = f"${d['precio']:,.2f}" if d["precio"] is not None else "n/d"
    cambio = d["cambio"]
    cambio_txt = f"{cambio:+.2f}% (24h)" if cambio is not None else "n/d"
    cambio_col = _color_cambio(cambio)
    fg_col = "#16c784" if (d["fg"] is not None and d["fg"] >= 55) else \
             ("#ea3943" if (d["fg"] is not None and d["fg"] < 45) else "#f0b90b")

    def fila(label, valor, color="#e8edf5"):
        return (f"<div class='ef-row'><span class='ef-row-l'>{label}</span>"
                f"<span class='ef-row-v' style='color:{color}'>{valor}</span></div>")

    vol_txt = (f"{d['vol']:.1f}" if d["vol"] is not None else "n/d")
    if d["vol_pct"] is not None:
        vol_txt += f" (p{d['vol_pct']:.0f})"

    return f"""
<div class='ef-panel'>
  <div class='ef-card'>
    <div class='ef-card-head'>ESTADO DEL MERCADO <span class='ef-live'>● En vivo</span></div>
    <div class='ef-price-block'>
      <div class='ef-coin'>Ethereum (ETH)</div>
      <div class='ef-price'>{precio}</div>
      <div class='ef-change' style='color:{cambio_col}'>{cambio_txt}</div>
    </div>
    {fila("Fear &amp; Greed", f"{d['fg']} ({d['fg_etq']})" if d['fg'] is not None else "n/d", fg_col)}
    {fila("Volatilidad (30d)", vol_txt)}
    {fila("Dominancia BTC", f"{d['dom_btc']:.1f}%" if d['dom_btc'] is not None else "n/d")}
    {fila("Dominancia ETH", f"{d['dom_eth']:.1f}%" if d['dom_eth'] is not None else "n/d")}
    {fila("Capitalización ETH", _fmt_grande(d['mcap']))}
    {fila("Volumen 24h", _fmt_grande(d['volumen']))}
    {fila("Drawdown desde ATH", f"{d['drawdown']:.1f}%" if d['drawdown'] is not None else "n/d", "#ea3943")}
  </div>
  <div class='ef-card'>
    <div class='ef-card-head'>MODELO LSTM <span class='ef-live'>● Activo</span></div>
    {fila("Última actualización", d['fecha'])}
    {fila("Accuracy direccional", f"{LSTM_ACCURACY_PCT}%")}
    {fila("Horizonte de predicción", f"{LSTM_HORIZONTE_DIAS} días")}
    {fila("Dirección", "Poco fiable (≈ azar)", "#f0b90b")}
  </div>
  <div class='ef-card'>
    <div class='ef-card-head'>SISTEMA RAG <span class='ef-live'>● Activo</span></div>
    {fila("Régimen (HMM)", d['regimen'], "#5b9bff")}
    {fila("Noticias analizadas", str(d['noticias']))}
    {fila("Documentos indexados", str(d['docs']))}
  </div>
  <div class='ef-card ef-note'>
    <div class='ef-note-title'>ⓘ Información importante</div>
    <div class='ef-note-body'>Los datos y análisis no constituyen asesoramiento
    financiero. La dirección del precio no es predecible con fiabilidad; el sistema
    aporta contexto razonado, no certezas.</div>
  </div>
</div>
"""


# ─── LATERAL DE CONVERSACIONES (visual) ────────────────────────────────────
def _item(texto, activo=False):
    cls = "ef-side-item ef-side-active" if activo else "ef-side-item"
    return f"<div class='{cls}'>{html.escape(texto)}</div>"


def _fitem(icono, texto):
    return f"<div class='ef-side-fitem'>{ICONO[icono]}<span>{html.escape(texto)}</span></div>"


SIDEBAR_HTML = f"""
<div class='ef-side'>
  <div class='ef-side-search'>{ICONO['search']}<span>Buscar conversaciones…</span></div>
  <div class='ef-side-scroll'>
    <div class='ef-side-group'>Hoy</div>
    {_item("¿Probabilidad de subida de ETH…", activo=True)}
    {_item("Análisis del sentimiento actual")}
    {_item("Noticias relevantes últimas 24 h")}
    <div class='ef-side-group'>Ayer</div>
    {_item("Contexto macroeconómico actual")}
    {_item("Ciclo de mercado actual")}
    {_item("Comparativa ETH vs BTC")}
    <div class='ef-side-group'>Esta semana</div>
    {_item("Escenario bajista para ETH")}
    {_item("Impacto de los ETF en Ethereum")}
  </div>
  <div class='ef-side-foot'>
    {_fitem("info", "Información del modelo")}
    {_fitem("sliders", "Configuración")}
    {_fitem("user", "Cuenta")}
  </div>
</div>
"""

LOGO_HTML = (f"<div class='ef-side-logo'><span class='ef-side-logo-ico'>{ETH_INLINE}</span>"
             "<span class='ef-side-logo-txt'>EtherForecast AI</span></div>")

CABECERA_HTML = """
<div class='ef-header'>
  <div class='ef-title'>Ether<span class='ef-title-accent'>Forecast</span> AI</div>
  <div class='ef-subtitle'>Análisis del mercado de Ethereum mediante HMM + LSTM + RAG</div>
</div>
"""

DISCLAIMER_HTML = """
<div class='ef-foot'>Las respuestas se generan con modelos estadísticos e inteligencia artificial y no
constituyen asesoramiento financiero. Los mercados son inherentemente impredecibles.</div>
"""


# ─── CSS ───────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 100% !important; background:#0a0e1a !important; padding:12px 16px !important; }
footer { display:none !important; }

/* ── Fila principal: 3 columnas que se encogen, nunca se apilan ── */
.ef-main-row { flex-wrap:nowrap !important; align-items:stretch !important; gap:14px !important; }
.ef-main-row > * { flex-shrink:1 !important; }

/* ── Columnas ── */
.ef-sidecol { background:#0c111e !important; border:1px solid #1a2233 !important; border-radius:16px !important;
  padding:14px 12px !important; display:flex !important; flex-direction:column !important; }
.ef-centercol { padding:0 4px !important; }
.ef-panelcol { padding:0 !important; }

/* Reparte la altura de la lateral: logo y botón fijos, lista crece, pie abajo */
.ef-sidecol > .ef-sb-logo, .ef-sidecol > .ef-new { flex:0 0 auto !important; }
.ef-sb-body { flex:1 1 auto !important; min-height:0 !important; display:flex !important; flex-direction:column !important; }
.ef-sb-body > * { flex:1 1 auto !important; min-height:0 !important; display:flex !important; flex-direction:column !important; }

/* ── Lateral ── */
.ef-side-logo { display:flex; align-items:center; gap:8px; color:#e8edf5; font-weight:700; font-size:1rem; padding:2px 6px 10px; }
.ef-side-logo-ico { display:inline-flex; color:#5b9bff; flex:none; }
.ef-side-logo { min-width:0; }
.ef-side-logo-txt { min-width:0; overflow-wrap:anywhere; }
.ef-new button { background:#2f6bff !important; color:#fff !important; border:none !important;
  border-radius:12px !important; font-weight:600 !important; width:100% !important; box-shadow:none !important;
  height:46px !important; min-height:46px !important; max-height:46px !important; flex:none !important; }
.ef-new button:hover { background:#3f78ff !important; }
.ef-side-search { display:flex; align-items:center; gap:8px; color:#6b7488; background:#0f1626;
  border:1px solid #1c2740; border-radius:10px; padding:9px 11px; font-size:0.83rem; margin:12px 2px; min-width:0; }
.ef-side-search svg { color:#5b6378; flex:none; }
.ef-side-search span { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ef-side { display:flex !important; flex-direction:column !important; flex:1 1 auto !important; min-height:0 !important; width:100%; }
.ef-side-scroll { flex:1 1 auto !important; min-height:0 !important; overflow-y:auto; scrollbar-width:none; }
.ef-side-scroll::-webkit-scrollbar { display:none; }
.ef-side-group { color:#5b6378; font-size:0.71rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; padding:11px 8px 4px; }
.ef-side-item { color:#aeb8cc; font-size:0.85rem; padding:8px 10px; border-radius:8px; cursor:default;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ef-side-item:hover { background:#141c2e; }
.ef-side-active { background:#16213a !important; color:#e8edf5 !important; }
.ef-side-foot { border-top:1px solid #1a2233; margin-top:auto; padding-top:8px; flex:0 0 auto !important; }
.ef-side-fitem { display:flex; align-items:center; gap:9px; color:#8a93a6; font-size:0.85rem; padding:8px; border-radius:8px; }
.ef-side-fitem:hover { background:#141c2e; }
.ef-side-fitem svg { color:#6b7488; flex:none; }
.ef-side-fitem { min-width:0; }
.ef-side-fitem span { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* ── Cabecera centro ── */
.ef-header { text-align:center; padding: 0 0 12px 0; }
.ef-title { font-size: 2rem; font-weight: 800; letter-spacing:-0.5px; color:#e8edf5; }
.ef-title-accent { color:#5b9bff; }
.ef-subtitle { color:#8a93a6; font-size:0.9rem; margin-top:4px; }

/* ── Tarjetas de ejemplo (icono de color + texto) ── */
.ef-chips-row { gap:11px !important; flex-wrap:wrap !important; justify-content:center !important; margin-bottom:12px !important; }
.ef-ex { flex:0 1 250px !important; }
.ef-ex button {
  display:flex !important; align-items:center !important; gap:11px !important;
  justify-content:flex-start !important; text-align:left !important;
  background:#121829 !important; border:1px solid #222b40 !important;
  color:#c3ccdd !important; border-radius:14px !important;
  padding:12px 14px !important; font-weight:500 !important; font-size:0.88rem !important;
  min-height:0 !important; box-shadow:none !important; line-height:1.35 !important;
  transition: border-color .15s ease, background .15s ease, transform .1s ease !important;
}
.ef-ex button:hover { border-color:#5b9bff !important; background:#161d31 !important; transform:translateY(-1px) !important; }
.ef-ex button img { width:20px !important; height:20px !important; flex:none !important;
  border-radius:9px !important; padding:6px !important; box-sizing:content-box !important; }
.ef-ex-prob  button img { background:rgba(91,155,255,.14) !important; }
.ef-ex-sent  button img { background:rgba(22,199,132,.14) !important; }
.ef-ex-news  button img { background:rgba(34,211,238,.14) !important; }
.ef-ex-macro button img { background:rgba(167,139,250,.14) !important; }
.ef-ex-ciclo button img { background:rgba(240,185,11,.14) !important; }
.ef-ex-risk  button img { background:rgba(255,107,91,.14) !important; }

/* ── Chat ── */
.ef-chat { border-radius:18px !important; border:1px solid #222b40 !important; background:#0e1422 !important; }
.ef-chat .bubble-wrap { background:transparent !important; padding:14px 16px !important;
  scrollbar-width:none !important; -ms-overflow-style:none !important; }
.ef-chat .bubble-wrap::-webkit-scrollbar { width:0 !important; height:0 !important; display:none !important; }
.ef-chat .message-row { padding:9px 0 !important; }
.ef-chat .flex-wrap { background:transparent !important; border:none !important; box-shadow:none !important; }
.ef-chat .avatar-container { width:34px !important; height:34px !important; background:#0f1626 !important; border:1px solid #222b40 !important; }
.ef-chat .avatar-image { object-fit:contain !important; padding:5px !important; }
.ef-chat .message.user { background:#2f6bff !important; color:#fff !important; border:none !important; box-shadow:none !important;
  border-radius:18px 18px 4px 18px !important; padding:11px 15px !important; max-width:80% !important; }
.ef-chat .message.bot { background:#101728 !important; color:#dde4f0 !important; border:1px solid #1c2740 !important;
  box-shadow:none !important; border-radius:4px 16px 16px 16px !important; padding:0 !important; max-width:94% !important; }
.ef-chat .message.bot .message-content { padding:14px 18px !important; font-size:0.95rem !important; line-height:1.65 !important; }
.ef-chat .message.bot .message-content p { margin:0 0 9px 0 !important; }
.ef-chat .message.bot .message-content strong { color:#fff !important; }

/* Cabecera de la ficha: sesgo + confianza */
.ef-rhead { display:flex; align-items:center; gap:12px; padding-bottom:12px; margin-bottom:12px; border-bottom:1px solid #1c2740; }
.ef-rhead-eth { display:inline-flex; align-items:center; justify-content:center; color:#8a93c4;
  background:#0f1626; border:1px solid #222b40; border-radius:50%; width:30px; height:30px; flex:none; }
.ef-rhead-main { display:flex; flex-direction:column; gap:1px; }
.ef-rhead-label { color:#8a93a6; font-size:0.71rem; text-transform:uppercase; letter-spacing:.04em; }
.ef-dir { display:flex; align-items:center; gap:5px; font-weight:700; font-size:1.05rem; }
.ef-dir-up { color:#16c784; } .ef-dir-down { color:#ea3943; } .ef-dir-neu { color:#8a93a6; }
.ef-rhead-conf { margin-left:auto; text-align:right; }
.ef-conf-label { display:block; color:#8a93a6; font-size:0.71rem; text-transform:uppercase; letter-spacing:.04em; }
.ef-conf-val { color:#5b9bff; font-weight:700; font-size:0.95rem; }
.ef-resumen-h { margin-bottom:6px !important; }

/* Pie de la ficha: secciones */
.ef-rfoot { margin-top:13px !important; padding-top:12px !important; border-top:1px solid #1c2740 !important; }
.ef-rsec { margin-top:11px !important; } .ef-rsec:first-child { margin-top:0 !important; }
.ef-rsec-h { display:flex; align-items:center; gap:7px; font-size:0.86rem; font-weight:700; color:#e8edf5; margin-bottom:3px; }
.ef-rsec-ico { display:inline-flex; }
.ic-blue .ef-rsec-ico { color:#5b9bff; } .ic-amber .ef-rsec-ico { color:#f0b90b; } .ic-teal .ef-rsec-ico { color:#16c784; }
.ef-rsec-b { color:#aeb8cc; font-size:0.85rem; line-height:1.5; }
.ef-warn { color:#f0b90b; }
.ef-srcwrap { display:flex; flex-wrap:wrap; gap:6px; margin-top:5px; }
.ef-src { background:#0f1626; border:1px solid #222b40; color:#aeb8cc; font-size:0.74rem; padding:3px 10px; border-radius:999px; }

.ef-chat .placeholder-content { height:100% !important; display:flex !important; align-items:center !important; justify-content:center !important; }
.ef-chat .placeholder { text-align:center !important; color:#aeb8cc !important; }

/* ── Caja de entrada ── */
.ef-inputrow { margin-top:8px !important; }
.ef-input { position:relative !important; }
.ef-input textarea { background:#0f1626 !important; border:1px solid #222b40 !important; border-radius:16px !important;
  padding:15px 18px !important; font-size:0.95rem !important; color:#e8edf5 !important; box-shadow:none !important;
  min-height:54px !important; scrollbar-width:none !important; }
.ef-input textarea::-webkit-scrollbar { display:none !important; }
.ef-input textarea:focus { border-color:#5b9bff !important; box-shadow:0 0 0 3px rgba(91,155,255,.15) !important; outline:none !important; }
.ef-input .input-container { align-items:center !important; padding-right:6px !important; }
.ef-input .submit-button { background:#2f6bff !important; border:none !important; border-radius:11px !important;
  width:40px !important; height:40px !important; min-width:40px !important;
  margin-left:12px !important; margin-right:2px !important; }
.ef-input .submit-button:hover { background:#3f78ff !important; }
.ef-input .submit-button svg { color:#fff !important; }

/* ── Panel derecho ── */
.ef-live { color:#16c784; font-size:0.7rem; font-weight:600; }
.ef-card-head { color:#c3ccdd; font-size:0.73rem; font-weight:700; letter-spacing:.06em; display:flex; justify-content:space-between; align-items:center; margin-bottom:9px; }
.ef-price-block { margin-bottom:11px; }
.ef-coin { color:#c3ccdd; font-size:0.82rem; }
.ef-price { color:#e8edf5; font-size:1.8rem; font-weight:800; line-height:1.1; margin:2px 0; }
.ef-change { font-size:0.87rem; font-weight:600; }
.ef-row { display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-top:1px solid #1a2233; font-size:0.85rem; }
.ef-row-l { color:#8a93a6; } .ef-row-v { color:#e8edf5; font-weight:600; }
.ef-note { background:#0f1626; }
.ef-note-title { color:#5b9bff; font-size:0.78rem; font-weight:700; margin-bottom:5px; }
.ef-note-body { color:#8a93a6; font-size:0.78rem; line-height:1.45; }

/* ── Pie legal ── */
.ef-foot { text-align:center; color:#5b6378; font-size:0.75rem; padding:14px 12px 2px; line-height:1.5; }
"""


def _tema_oscuro():
    return gr.themes.Base(
        primary_hue="blue", neutral_hue="slate", radius_size="lg",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0a0e1a", body_text_color="#e8edf5",
        block_background_fill="#121829", block_border_color="#222b40",
        block_label_text_color="#8a93a6",
        input_background_fill="#0f1626", input_border_color="#222b40",
        button_primary_background_fill="#2f6bff", button_primary_text_color="#ffffff",
    )


# ─── INTERFAZ ──────────────────────────────────────────────────────────────
_preparar_iconos()

with gr.Blocks(title="EtherForecast AI", fill_height=True) as demo:
    with gr.Row(equal_height=False, elem_classes=["ef-main-row"]):

        # ── IZQUIERDA: conversaciones (visual) ──
        with gr.Column(scale=11, min_width=150, elem_classes=["ef-sidecol"]):
            gr.HTML(LOGO_HTML, elem_classes=["ef-sb-logo"])
            nueva = gr.Button("＋  Nueva conversación", elem_classes=["ef-new"])
            gr.HTML(SIDEBAR_HTML, elem_classes=["ef-sb-body"])

        # ── CENTRO: EtherForecast AI ──
        with gr.Column(scale=34, min_width=320, elem_classes=["ef-centercol"]):
            gr.HTML(CABECERA_HTML)

            botones_ejemplo = []
            with gr.Row(elem_classes=["ef-chips-row"]):
                for clave, color, pregunta in EJEMPLOS:
                    b = gr.Button(pregunta, icon=_icono_ejemplo(clave),
                                  elem_classes=["ef-ex", f"ef-ex-{clave}"])
                    botones_ejemplo.append((b, pregunta))

            chatbot = gr.Chatbot(
                height=540, show_label=False, elem_classes=["ef-chat"],
                layout="bubble", avatar_images=(None, _ruta_avatar()),
                sanitize_html=False, render_markdown=True, line_breaks=True,
                placeholder="<div style='text-align:center'>"
                            "<div style='font-size:2rem; margin-bottom:6px'>⟠</div>"
                            "<strong style='font-size:1.05rem'>EtherForecast AI</strong><br>"
                            "<span style='color:#8a93a6'>Pregunta sobre Ethereum, noticias, "
                            "contexto de mercado o predicciones.</span></div>",
            )

            with gr.Row(elem_classes=["ef-inputrow"]):
                caja = gr.Textbox(
                    show_label=False, autofocus=True, elem_classes=["ef-input"],
                    placeholder="Pregunta sobre Ethereum, noticias, contexto o predicciones...",
                    lines=1, max_lines=6, submit_btn=True,
                )

            gr.HTML(DISCLAIMER_HTML)

        # ── DERECHA: estado del mercado (pegado a la derecha) ──
        with gr.Column(scale=13, min_width=220, elem_classes=["ef-panelcol"]):
            panel = gr.HTML(construir_panel_html())

    # ── Eventos ──
    caja.submit(agregar_usuario, [caja, chatbot], [chatbot, caja]).then(
        responder_bot, chatbot, chatbot)

    for boton, pregunta in botones_ejemplo:
        boton.click(lambda h, t=pregunta: agregar_usuario(t, h),
                    inputs=chatbot, outputs=[chatbot, caja]).then(
                    responder_bot, chatbot, chatbot)

    nueva.click(nueva_conversacion, None, chatbot)


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=_tema_oscuro(), css=CSS, allowed_paths=[ASSETS])