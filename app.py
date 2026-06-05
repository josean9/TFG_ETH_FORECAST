"""
App de chat (Gradio) para el sistema de apoyo a la decisión sobre Ethereum.

Interfaz conversacional que usa el módulo rag.py para responder, manteniendo
memoria de la conversación (las preguntas-respuestas previas se pasan al prompt).

Despliegue: pensada para Hugging Face Spaces (gradio).
  - Variables de entorno (Secrets en HF): GEMINI_API_KEY, NEWSDATA_KEY, (ANTHROPIC_API_KEY opcional)
  - Ejecutar en local:  python app.py
"""

import gradio as gr
import rag

# Proveedor de generación: "gemini" (por defecto) o "claude"
PROVEEDOR = "gemini"

# Cuántos turnos previos recordar (para no saturar el prompt)
MAX_TURNOS_MEMORIA = 5


def _construir_historial(history):
    """Convierte el historial de Gradio en texto para el prompt.
    Tolera ambos formatos: lista de dicts {role, content} (Gradio 5+)
    o lista de pares [usuario, asistente] (Gradio 4)."""
    if not history:
        return ""
    recientes = history[-MAX_TURNOS_MEMORIA:]
    lineas = []
    for turno in recientes:
        if isinstance(turno, dict):
            rol = "Usuario" if turno.get("role") == "user" else "Asistente"
            lineas.append(f"{rol}: {turno.get('content','')}")
        elif isinstance(turno, (list, tuple)) and len(turno) == 2:
            if turno[0]:
                lineas.append(f"Usuario: {turno[0]}")
            if turno[1]:
                lineas.append(f"Asistente: {turno[1]}")
    return "\n".join(lineas)


def responder_chat(mensaje, history):
    """Función que Gradio llama en cada turno."""
    historial_txt = _construir_historial(history)
    respuesta, fuentes = rag.responder(mensaje, proveedor=PROVEEDOR, historial=historial_txt)

    # Añadir un pie con las fuentes usadas (transparencia)
    snap = fuentes.get("snapshot", {})
    senal = fuentes.get("senal")
    pie = f"\n\n---\n*Régimen actual (HMM): **{snap.get('regimen','?')}** · "
    if senal:
        r = senal["retornos_diarios"]
        pie += f"Señal LSTM 3d: {r[0]:+.1f}% / {r[1]:+.1f}% / {r[2]:+.1f}% (dirección poco fiable) · "
    pie += f"{len(fuentes.get('noticias',[]))} noticias · {len(fuentes.get('embebidos',[]))} docs*"

    return respuesta + pie


# ─── INTERFAZ ────────────────────────────────────────────────────────────
DESCRIPCION = """
# 🔷 Asistente de análisis de Ethereum

Sistema de **apoyo a la decisión** que combina:
**HMM** (régimen de mercado) · **LSTM** (señal cuantitativa a 3 días) · **RAG** (noticias + conocimiento experto).

⚠️ *No es asesoramiento financiero. La dirección del precio no es predecible con fiabilidad;
el sistema aporta contexto razonado, no certezas.*
"""

EJEMPLOS = [
    "¿Cómo ves Ethereum esta semana?",
    "¿Qué estrategia tendría sentido en el régimen actual?",
    "¿Las noticias recientes son favorables o desfavorables para ETH?",
    "¿Es buen momento para hacer DCA?",
]

with gr.Blocks(title="Asistente Ethereum", theme=gr.themes.Soft(
        primary_hue="indigo", secondary_hue="slate")) as demo:
    gr.Markdown(DESCRIPCION)
    gr.ChatInterface(
        fn=responder_chat,
        examples=EJEMPLOS,
    )

if __name__ == "__main__":
    demo.launch()
