"""
App de chat (Gradio) para el sistema de apoyo a la decisión sobre Ethereum.
"EtherForecast AI" — interfaz tipo dashboard.

El envío del mensaje se hace en DOS PASOS: primero se muestra la pregunta del
usuario al instante, y después se calcula la respuesta. Así la interfaz no se
queda congelada mientras el RAG trabaja (lo que provocaba "conexión perdida").

Compatible con Gradio 6.x: theme y css se pasan en launch(); gr.Chatbot usa el
formato "messages" por defecto (no se pasa type=).

Despliegue: Hugging Face Spaces.
  - Secrets: GEMINI_API_KEY, NEWSDATA_KEY, (ANTHROPIC_API_KEY opcional)
  - Local:  python app.py
"""

import gradio as gr
import rag

# ─── CONFIG ───────────────────────────────────────────────────────────────
PROVEEDOR = "gemini"
MAX_TURNOS_MEMORIA = 5

LSTM_ACCURACY_PCT = 51.3
LSTM_HORIZONTE_DIAS = 3

EJEMPLOS = [
    "📈  ¿Cuál es la probabilidad de subida de ETH esta semana?",
    "🙂  Analiza el sentimiento actual del mercado.",
    "📰  Resume las noticias más relevantes de las últimas 24 horas.",
    "🌐  ¿Qué factores macroeconómicos afectan ahora a Ethereum?",
    "📊  Explica el contexto actual del ciclo de mercado.",
    "🛡️  ¿Qué riesgos existen ahora mismo?",
]


# ─── HISTORIAL → TEXTO PARA EL PROMPT ─────────────────────────────────────
def _a_texto(contenido):
    """Convierte el 'content' de un mensaje a texto plano.

    En cristiano: en Gradio 6 el contenido de un mensaje ya no es siempre un
    texto suelto; puede venir envuelto como una lista de bloques, p.ej.
    [{'text': 'hola', 'type': 'text'}]. Esta función lo desenvuelve y devuelve
    siempre un string limpio, venga en el formato que venga. Sin esto, ese
    contenido envuelto llegaba a Gemini como si fuera la pregunta y lo rechazaba.
    """
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
                partes.append(_a_texto(bloque))   # por si viene doblemente anidado
        return " ".join(p for p in partes if p)
    return str(contenido)


def _construir_historial(history):
    """Convierte el historial del chat en texto para pasarlo al prompt."""
    if not history:
        return ""
    recientes = history[-(MAX_TURNOS_MEMORIA * 2):]   # cada turno = user + assistant
    lineas = []
    for turno in recientes:
        if isinstance(turno, dict):
            rol = "Usuario" if turno.get("role") == "user" else "Asistente"
            lineas.append(f"{rol}: {_a_texto(turno.get('content', ''))}")
    return "\n".join(lineas)


def _responder_con_pie(mensaje, historial_txt):
    """Llama al RAG y añade el pie de transparencia (régimen, señal, fuentes)."""
    respuesta, fuentes = rag.responder(mensaje, proveedor=PROVEEDOR, historial=historial_txt)
    snap = fuentes.get("snapshot", {}) or {}
    senal = fuentes.get("senal")
    pie = f"\n\n---\n*Régimen actual (HMM): **{snap.get('regimen', '?')}**"
    if senal:
        r = senal["retornos_diarios"]
        pie += f" · Señal LSTM 3d: {r[0]:+.1f}% / {r[1]:+.1f}% / {r[2]:+.1f}% (dirección poco fiable)"
    pie += (f" · {len(fuentes.get('noticias', []))} noticias"
            f" · {len(fuentes.get('embebidos', []))} docs*")
    return respuesta + pie


# ─── ENVÍO EN DOS PASOS ───────────────────────────────────────────────────
def agregar_usuario(mensaje, history):
    """Paso 1: muestra la pregunta del usuario al instante y vacía la caja."""
    if not mensaje or not mensaje.strip():
        return history, ""
    history = (history or []) + [{"role": "user", "content": mensaje.strip()}]
    return history, ""


def responder_bot(history):
    """Paso 2: calcula la respuesta para el último mensaje del usuario.

    En cristiano: mira el último mensaje (la pregunta), llama al RAG y añade la
    respuesta al chat. Si algo falla, muestra el error dentro del chat en lugar
    de tumbar la app.
    """
    if not history or history[-1].get("role") != "user":
        return history
    mensaje = _a_texto(history[-1].get("content", ""))
    historial_txt = _construir_historial(history[:-1])
    try:
        texto = _responder_con_pie(mensaje, historial_txt)
    except Exception as e:
        texto = f"❌ Ha ocurrido un error generando la respuesta: {e}"
    return history + [{"role": "assistant", "content": texto}]


# ─── PANEL DERECHO (datos reales del snapshot) ────────────────────────────
def _fmt_grande(n):
    """Formatea un número grande como 515.2B / 18.7B / 1.2M."""
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
    """Longitud a prueba de None y de DataFrame (NO usar 'or []' con DataFrames)."""
    if x is None:
        return 0
    try:
        return len(x)
    except Exception:
        return 0


def _datos_panel():
    """Recoge los datos del panel derecho desde el snapshot y el estado del RAG."""
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
    """Construye el HTML del panel derecho con los datos reales."""
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


CABECERA_HTML = """
<div class='ef-header'>
  <div class='ef-title'>Ether<span class='ef-title-accent'>Forecast</span> AI</div>
  <div class='ef-subtitle'>Análisis del mercado de Ethereum mediante HMM + LSTM + RAG</div>
</div>
"""


# ─── CSS (bordes más redondeados y suaves) ────────────────────────────────
CSS = """
.gradio-container { max-width: 100% !important; background:#0a0e1a !important; }
footer { display:none !important; }

.ef-header { text-align:center; padding: 6px 0 18px 0; }
.ef-title { font-size: 2.2rem; font-weight: 800; letter-spacing:-0.5px; color:#e8edf5; }
.ef-title-accent { color:#5b9bff; }
.ef-subtitle { color:#8a93a6; font-size:0.95rem; margin-top:6px; }

/* ── Chips de ejemplo ── */
.ef-chips { flex-wrap: wrap !important; gap: 8px !important; justify-content: center; margin-bottom: 4px; }
.ef-ejemplo button {
  background:#121829 !important; border:1px solid #222b40 !important;
  color:#aeb8cc !important; border-radius:999px !important;
  padding:8px 16px !important; font-weight:500 !important;
  font-size:0.84rem !important; min-height:0 !important; width:auto !important;
  box-shadow:none !important;
  transition: border-color .15s ease, color .15s ease !important;
}
.ef-ejemplo button:hover { border-color:#5b9bff !important; color:#e8edf5 !important; }

/* ── Chat estilo conversación ── */
.ef-chat { border-radius:20px !important; border:1px solid #222b40 !important; background:#0e1422 !important; }
.ef-chat .message {
  border:none !important; box-shadow:none !important;
  font-size:0.95rem !important; line-height:1.65 !important;
}
/* Usuario: burbuja azul a la derecha */
.ef-chat .message-row.user-row { justify-content: flex-end !important; }
.ef-chat .message.user {
  background:#2f6bff !important; color:#ffffff !important;
  border-radius:18px 18px 4px 18px !important;
  padding:11px 16px !important; max-width:78% !important;
}
/* Asistente: texto limpio a la izquierda, sin caja */
.ef-chat .message.bot {
  background:transparent !important; color:#dde4f0 !important;
  padding:6px 4px !important; max-width:100% !important;
}
/* El pie de transparencia (régimen/señal) más discreto */
.ef-chat .message.bot em { color:#7c8699; font-size:0.78rem; }
.ef-chat .message.bot hr { border-color:#1a2233 !important; margin:10px 0 !important; }
/* Iconos (copiar/compartir) solo al pasar el ratón */
.ef-chat .icon-button-wrapper { opacity:0 !important; transition:opacity .15s ease; }
.ef-chat .message-row:hover .icon-button-wrapper { opacity:1 !important; }

/* ── Placeholder del chat centrado ── */
.ef-chat .placeholder-content { height:100% !important; }
.ef-chat .placeholder {
  display:flex !important; align-items:center !important;
  justify-content:center !important; height:100% !important;
  text-align:center !important; color:#aeb8cc !important;
}

/* ── Caja de entrada estilo Claude/ChatGPT (botón dentro) ── */
.ef-inputwrap {
  position:relative !important;
  background:#0f1626 !important;
  border:1px solid #222b40 !important;
  border-radius:18px !important;
  padding:4px !important;
  transition: border-color .15s ease, box-shadow .15s ease;
}
.ef-inputwrap:focus-within {
  border-color:#5b9bff !important;
  box-shadow:0 0 0 3px rgba(91,155,255,.15) !important;
}
.ef-input, .ef-input > * {
  background:transparent !important; border:none !important; box-shadow:none !important;
}
.ef-input textarea {
  background:transparent !important; border:none !important;
  box-shadow:none !important; resize:none !important;
  padding:12px 64px 12px 16px !important;
  font-size:0.95rem !important;
  scrollbar-width:none !important;            /* Firefox */
}
.ef-input textarea::-webkit-scrollbar { display:none !important; }  /* Chrome/Edge */

.ef-send {
  position:absolute !important; right:12px !important; bottom:12px !important;
  width:38px !important; min-width:38px !important; max-width:38px !important;
  flex:none !important;
}
.ef-send button {
  width:38px !important; height:34px !important; min-width:0 !important;
  padding:0 !important; border-radius:10px !important;
  font-size:0.95rem !important; line-height:1 !important;
  display:flex !important; align-items:center !important; justify-content:center !important;
}


.ef-live { color:#16c784; font-size:0.7rem; font-weight:600; }
.ef-price-block { margin-bottom:14px; }
.ef-coin { color:#c3ccdd; font-size:0.85rem; }
.ef-price { color:#e8edf5; font-size:1.95rem; font-weight:800; line-height:1.1; margin:3px 0; }
.ef-change { font-size:0.9rem; font-weight:600; }
.ef-row {
  display:flex; justify-content:space-between; align-items:center;
  padding:8px 0; border-top:1px solid #1a2233; font-size:0.88rem;
}
.ef-row-l { color:#8a93a6; }
.ef-row-v { color:#e8edf5; font-weight:600; }
.ef-note { background:#0f1626; }
.ef-note-title { color:#5b9bff; font-size:0.8rem; font-weight:700; margin-bottom:6px; }
.ef-note-body { color:#8a93a6; font-size:0.8rem; line-height:1.45; }

/* Matar el recuadro interior y el padding por defecto del scroll */
.ef-chat .bubble-wrap { background:transparent !important; padding:10px 14px !important; }

/* Aire entre turnos */
.ef-chat .message-row { padding:7px 0 !important; }
.ef-chat .flex-wrap { background:transparent !important; box-shadow:none !important; }

/* El contenedor real del texto del bot, para tipografía limpia */
.ef-chat .message.bot .message-content { font-size:0.95rem !important; line-height:1.65 !important; }

/* Iconos copiar/like: discretos, solo al hover (nombre correcto en 6.16) */
.ef-chat .message-buttons { opacity:0 !important; transition:opacity .15s ease; }
.ef-chat .message-row:hover .message-buttons { opacity:1 !important; }
"""


# ─── TEMA OSCURO ──────────────────────────────────────────────────────────
def _tema_oscuro():
    """Tema oscuro tipo dashboard (fondo navy, acento azul, esquinas redondeadas)."""
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


# ─── INTERFAZ ─────────────────────────────────────────────────────────────
with gr.Blocks(title="EtherForecast AI", fill_height=True) as demo:

    gr.HTML(CABECERA_HTML)

    with gr.Row(equal_height=False):

        with gr.Column(scale=3):
            botones_ejemplo = []
            with gr.Row(elem_classes=["ef-chips"]):
                for texto in EJEMPLOS:
                    b = gr.Button(texto, elem_classes=["ef-ejemplo"], size="sm")
                    botones_ejemplo.append((b, texto))

            chatbot = gr.Chatbot(
                height=520, show_label=False, elem_classes=["ef-chat"],
                placeholder="<div style='text-align:center'>"
                            "<div style='font-size:2rem; margin-bottom:6px'>⟠</div>"
                            "<strong style='font-size:1.05rem'>EtherForecast AI</strong><br>"
                            "<span style='color:#8a93a6'>Pregunta sobre Ethereum, noticias, "
                            "contexto de mercado o predicciones.</span></div>",
            )

            with gr.Group(elem_classes=["ef-inputwrap"]):
                caja = gr.Textbox(
                    show_label=False, autofocus=True, elem_classes=["ef-input"],
                    placeholder="Pregunta sobre Ethereum, noticias, contexto o predicciones...",
                    lines=1, max_lines=6,
                )
                enviar = gr.Button("➤", variant="primary", elem_classes=["ef-send"])


        with gr.Column(scale=1, min_width=300):
            panel = gr.HTML(construir_panel_html())

    # ── Eventos (dos pasos: mostrar pregunta → calcular respuesta) ─────
    caja.submit(agregar_usuario, [caja, chatbot], [chatbot, caja]).then(
        responder_bot, chatbot, chatbot)
    enviar.click(agregar_usuario, [caja, chatbot], [chatbot, caja]).then(
        responder_bot, chatbot, chatbot)

    for boton, texto in botones_ejemplo:
        pregunta = texto.split("  ", 1)[-1]   # quita el emoji inicial
        boton.click(lambda h, t=pregunta: agregar_usuario(t, h),
                    inputs=chatbot, outputs=[chatbot, caja]).then(
                    responder_bot, chatbot, chatbot)


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=_tema_oscuro(), css=CSS)