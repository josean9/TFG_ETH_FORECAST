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

.ef-ejemplo button {
  background:#121829 !important; border:1px solid #222b40 !important;
  color:#c3ccdd !important; text-align:left !important; border-radius:18px !important;
  padding:16px 18px !important; font-weight:500 !important; min-height:62px !important;
  box-shadow:none !important;
  transition: border-color .18s ease, background .18s ease, transform .1s ease !important;
}
.ef-ejemplo button:hover {
  border-color:#5b9bff !important; background:#161d31 !important; transform: translateY(-1px);
}

.ef-chat { border-radius:20px !important; border:1px solid #222b40 !important; background:#0e1422 !important; }
.ef-input textarea { border-radius:16px !important; background:#0f1626 !important; border:1px solid #222b40 !important; }
.ef-send button { border-radius:16px !important; font-weight:600 !important; }

.ef-panel { display:flex; flex-direction:column; gap:16px; }
.ef-card { background:#121829; border:1px solid #222b40; border-radius:18px; padding:18px 20px; }
.ef-card-head {
  font-size:0.72rem; letter-spacing:0.8px; color:#8a93a6; font-weight:700;
  display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;
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
            with gr.Row():
                for texto in EJEMPLOS[:3]:
                    b = gr.Button(texto, elem_classes=["ef-ejemplo"])
                    botones_ejemplo.append((b, texto))
            with gr.Row():
                for texto in EJEMPLOS[3:]:
                    b = gr.Button(texto, elem_classes=["ef-ejemplo"])
                    botones_ejemplo.append((b, texto))

            chatbot = gr.Chatbot(
                height=440, show_label=False, elem_classes=["ef-chat"],
                placeholder="<strong>EtherForecast AI</strong><br>"
                            "Pregunta sobre Ethereum, noticias, contexto de mercado o predicciones.",
            )

            with gr.Row():
                caja = gr.Textbox(
                    show_label=False, scale=9, autofocus=True, elem_classes=["ef-input"],
                    placeholder="Pregunta sobre Ethereum, noticias, contexto o predicciones...",
                )
                enviar = gr.Button("Enviar ➤", variant="primary", scale=1, elem_classes=["ef-send"])

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