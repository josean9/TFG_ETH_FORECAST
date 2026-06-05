"""
App de chat (Gradio) para el sistema de apoyo a la decisión sobre Ethereum.
"EtherForecast AI" — interfaz tipo dashboard.

Diseño:
  - Cabecera con título.
  - Tarjetas de preguntas de ejemplo (clicables).
  - Chat central (gr.Chatbot montado a mano para poder tener panel lateral).
  - Panel derecho con datos REALES del snapshot (precio, % del día, Fear&Greed,
    volatilidad, dominancias, régimen) + estado del modelo LSTM y del sistema RAG.

Toda la lógica de análisis vive en rag.py; esta app solo es la interfaz.

Compatible con Gradio 6.x:
  - theme y css se pasan en launch() (no en Blocks).
  - gr.Chatbot usa el formato "messages" (lista de dicts {role, content}), que es
    el valor por defecto en Gradio 6, por eso no se pasa type=.

Despliegue: Hugging Face Spaces.
  - Secrets: GEMINI_API_KEY, NEWSDATA_KEY, (ANTHROPIC_API_KEY opcional)
  - Local:  python app.py
"""

import gradio as gr
import rag

# ─── CONFIG ───────────────────────────────────────────────────────────────
PROVEEDOR = "gemini"          # "gemini" (por defecto) o "claude"
MAX_TURNOS_MEMORIA = 5        # turnos previos que se recuerdan en el prompt

# Accuracy direccional REAL de la LSTM en test (del entrenamiento). Es un dato
# fijo del modelo, por eso es una constante y no se recalcula en vivo.
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
def _construir_historial(history):
    """Convierte el historial del chat en texto para pasarlo al prompt.

    En cristiano: coge los últimos turnos de la conversación y los escribe como
    'Usuario: ... / Asistente: ...' para que el modelo recuerde de qué se hablaba.
    Tolera el formato de dicts {role, content} de Gradio 6.
    """
    if not history:
        return ""
    recientes = history[-(MAX_TURNOS_MEMORIA * 2):]   # *2 porque cada turno = user + assistant
    lineas = []
    for turno in recientes:
        if isinstance(turno, dict):
            rol = "Usuario" if turno.get("role") == "user" else "Asistente"
            lineas.append(f"{rol}: {turno.get('content', '')}")
        elif isinstance(turno, (list, tuple)) and len(turno) == 2:
            if turno[0]: lineas.append(f"Usuario: {turno[0]}")
            if turno[1]: lineas.append(f"Asistente: {turno[1]}")
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


def responder_chat(mensaje, history):
    """Función principal: añade la pregunta y la respuesta al historial.

    Devuelve el historial actualizado y vacía la caja de texto.
    """
    if not mensaje or not mensaje.strip():
        return history, ""
    historial_txt = _construir_historial(history)
    try:
        texto = _responder_con_pie(mensaje.strip(), historial_txt)
    except Exception as e:
        texto = f"❌ Ha ocurrido un error generando la respuesta: {e}"
    history = (history or []) + [
        {"role": "user", "content": mensaje.strip()},
        {"role": "assistant", "content": texto},
    ]
    return history, ""


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


def _datos_panel():
    """Recoge los datos para el panel derecho desde el snapshot y el estado del RAG.

    En cristiano: lee la foto del mercado más reciente (la del snapshot) y los
    contadores del sistema (noticias cargadas, documentos indexados) para
    mostrarlos en las tarjetas de la derecha. Si algo falla, devuelve valores
    'n/d' sin romper la app.
    """
    d = {"precio": None, "cambio": None, "fg": None, "fg_etq": "n/d",
         "vol": None, "vol_pct": None, "dom_btc": None, "dom_eth": None,
         "regimen": "n/d", "drawdown": None, "fecha": "n/d",
         "mcap": None, "volumen": None,
         "noticias": 0, "docs": 0}
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
    # Market cap y volumen: del último día del df (si está disponible)
    try:
        fila = rag._df.iloc[-1]
        d["mcap"] = float(fila["eth_mcap"]) if "eth_mcap" in fila else None
        d["volumen"] = float(fila["eth_volume"]) if "eth_volume" in fila else None
    except Exception:
        pass
    # Estado del sistema
    try:
        d["noticias"] = len(getattr(rag, "_noticias_df", []) or [])
        d["docs"] = len(getattr(rag, "_emb_chunks", []) or [])
    except Exception:
        pass
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

    panel = f"""
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
    {fila("Drawdown desde ATH", f"{d['drawdown']:.1f}%" if d['drawdown'] is not None else "n/d", cambio_col if False else "#ea3943")}
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
    return panel


CABECERA_HTML = """
<div class='ef-header'>
  <div class='ef-title'>Ether<span class='ef-title-accent'>Forecast</span> AI</div>
  <div class='ef-subtitle'>Análisis del mercado de Ethereum mediante HMM + LSTM + RAG</div>
</div>
"""


# ─── CSS ──────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 100% !important; }

.ef-header { text-align:center; padding: 8px 0 14px 0; }
.ef-title { font-size: 2.1rem; font-weight: 800; letter-spacing:-0.5px; color:#e8edf5; }
.ef-title-accent { color:#5b9bff; }
.ef-subtitle { color:#8a93a6; font-size:0.95rem; margin-top:4px; }

/* Tarjetas de ejemplo (botones) */
.ef-ejemplo button {
  background:#121829 !important; border:1px solid #232c43 !important;
  color:#c3ccdd !important; text-align:left !important; border-radius:12px !important;
  padding:14px 16px !important; font-weight:500 !important; min-height:60px !important;
  transition: border-color .15s ease, background .15s ease !important;
}
.ef-ejemplo button:hover {
  border-color:#5b9bff !important; background:#161d31 !important;
}

/* Panel derecho */
.ef-panel { display:flex; flex-direction:column; gap:14px; }
.ef-card {
  background:#121829; border:1px solid #232c43; border-radius:14px;
  padding:16px 18px;
}
.ef-card-head {
  font-size:0.72rem; letter-spacing:0.8px; color:#8a93a6; font-weight:700;
  display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;
}
.ef-live { color:#16c784; font-size:0.7rem; font-weight:600; }
.ef-price-block { margin-bottom:14px; }
.ef-coin { color:#c3ccdd; font-size:0.85rem; }
.ef-price { color:#e8edf5; font-size:1.9rem; font-weight:800; line-height:1.1; margin:2px 0; }
.ef-change { font-size:0.9rem; font-weight:600; }
.ef-row {
  display:flex; justify-content:space-between; align-items:center;
  padding:7px 0; border-top:1px solid #1b2334; font-size:0.88rem;
}
.ef-row-l { color:#8a93a6; }
.ef-row-v { color:#e8edf5; font-weight:600; }
.ef-note { background:#0f1626; }
.ef-note-title { color:#5b9bff; font-size:0.8rem; font-weight:700; margin-bottom:6px; }
.ef-note-body { color:#8a93a6; font-size:0.8rem; line-height:1.4; }
"""


# ─── TEMA OSCURO ──────────────────────────────────────────────────────────
def _tema_oscuro():
    """Tema oscuro tipo dashboard (fondo navy, acento azul)."""
    return gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0a0e1a",
        body_text_color="#e8edf5",
        block_background_fill="#121829",
        block_border_color="#232c43",
        block_label_text_color="#8a93a6",
        input_background_fill="#0f1626",
        input_border_color="#232c43",
        button_primary_background_fill="#2f6bff",
        button_primary_text_color="#ffffff",
    )


# ─── INTERFAZ ─────────────────────────────────────────────────────────────
with gr.Blocks(title="EtherForecast AI", fill_height=True) as demo:

    gr.HTML(CABECERA_HTML)

    with gr.Row(equal_height=False):

        # ── Columna central: chat ─────────────────────────────────────
        with gr.Column(scale=3):

            # Tarjetas de ejemplo (2 filas de 3)
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
                height=460,
                show_label=False,
                avatar_images=(None, None),
                placeholder="<strong>EtherForecast AI</strong><br>"
                            "Pregunta sobre Ethereum, noticias, contexto de mercado o predicciones.",
            )

            with gr.Row():
                caja = gr.Textbox(
                    show_label=False, scale=9, autofocus=True,
                    placeholder="Pregunta sobre Ethereum, noticias, contexto o predicciones...",
                )
                enviar = gr.Button("Enviar ➤", variant="primary", scale=1)

        # ── Columna derecha: panel de datos ───────────────────────────
        with gr.Column(scale=1, min_width=300):
            panel = gr.HTML(construir_panel_html())

    # ── Eventos ───────────────────────────────────────────────────────
    caja.submit(responder_chat, [caja, chatbot], [chatbot, caja])
    enviar.click(responder_chat, [caja, chatbot], [chatbot, caja])

    # Botones de ejemplo: cada uno envía su texto (el default t=texto captura el valor)
    for boton, texto in botones_ejemplo:
        boton.click(lambda h, t=texto: responder_chat(t.split("  ", 1)[-1], h),
                    inputs=chatbot, outputs=[chatbot, caja])


if __name__ == "__main__":
    demo.launch(theme=_tema_oscuro(), css=CSS)