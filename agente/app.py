"""Interfaz web del agente (Streamlit).

Correr desde la carpeta Solucion:
    streamlit run agente/app.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import altair as alt
import pandas as pd
import streamlit as st

from agente import herramientas as H
from agente import sin_llm
from agente.nucleo import crear_agente, resultado_json

st.set_page_config(page_title="Agente de costos de equipos", layout="wide")

COLORES = {"Equipo1": "#9467bd", "Equipo2": "#d62728", "X": "#1f77b4", "Y": "#ff7f0e", "Z": "#2ca02c"}
EJEMPLOS = [
    "¿Qué materias primas determinan el precio de cada equipo y cuáles son ruido?",
    "¿Cuánto costará el Equipo 1 en febrero de 2024 y con qué rango?",
    "Según las noticias actuales del aluminio, ¿cuánto costaría hoy el Equipo 2? Compáralo con el pronóstico.",
    "Un proveedor me cotiza el Equipo 1 en 520 con precios de agosto 2023. ¿Es caro?",
    "Si el acero sube 30%, ¿qué pasa con cada equipo?",
    "Voy a comprar 40 Equipo1 en enero 2024 y 25 Equipo2 en marzo 2024. Hazme el Excel del presupuesto con contingencia.",
]

# estado
if "agente" not in st.session_state:
    try:
        st.session_state.agente = crear_agente()
    except Exception:
        st.session_state.agente = None
if "chat" not in st.session_state:
    st.session_state.chat = []

agente = st.session_state.agente

# barra lateral
with st.sidebar:
    st.header("Agente de costos")
    modos = ["Agente con IA (Gemini)", "Modo sin IA (reglas)"] if agente else ["Modo sin IA (reglas)"]
    modo = st.radio("Modo", modos)
    if not agente:
        st.warning("Sin API key: el agente responde con reglas. Agrega GEMINI_API_KEY en .env.")
    st.caption("Datos del modelo: ene-2010 a ago-2023 | Pronóstico: sep-2023 a ago-2024")
    if st.button("Nueva conversación", width="stretch"):
        st.session_state.chat = []
        if agente:
            agente.reiniciar()
        st.rerun()
    st.subheader("Preguntas de ejemplo")
    for q in EJEMPLOS:
        if st.button(q, width="stretch"):
            st.session_state.pendiente = q
    with st.expander("Memoria guardada"):
        notas = H.leer_memoria()
        if notas:
            for n in notas[-10:]:
                st.write(f"- {n['fecha']}: {n['nota']}")
        else:
            st.caption("El agente guarda aquí datos útiles (planes de compra, decisiones).")

tab_chat, tab_pron, tab_agente, tab_metodo = st.tabs(
    ["Agente", "Pronóstico", "IA vs. Agente", "Metodología"])

# chat
with tab_chat:
    st.subheader("Pregúntale al agente sobre los costos de los equipos")
    for msg in st.session_state.chat:
        with st.chat_message(msg["rol"]):
            st.markdown(msg["texto"].replace("$", r"\$"))
            if msg.get("pasos"):
                with st.expander(f"Detalle de ejecución: {len(msg['pasos'])} herramientas - {msg['modelo']}"):
                    for i, p in enumerate(msg["pasos"], 1):
                        st.markdown(f"**{i}. `{p.herramienta}`** `{p.argumentos}`")
                        st.code(resultado_json(p), language="json")
            for archivo in msg.get("archivos", []):
                st.download_button(f"Descargar {Path(archivo).name}", Path(archivo).read_bytes(),
                                   file_name=Path(archivo).name, key=archivo)

    pregunta = st.chat_input("Escribe tu pregunta...") or st.session_state.pop("pendiente", None)
    if pregunta:
        st.session_state.chat.append({"rol": "user", "texto": pregunta})
        with st.spinner("El agente está pensando y usando sus herramientas..."):
            try:
                if agente and modo.startswith("Agente"):
                    r = agente.preguntar(pregunta)
                else:
                    r = sin_llm.responder(pregunta)
            except Exception as e:
                r = sin_llm.responder(pregunta)
                r.texto = f"La IA no respondió ({type(e).__name__}); uso el modo sin IA.\n\n" + r.texto
        archivos = [p.resultado["archivo"] for p in r.pasos if p.herramienta == "generar_reporte"
                    and "archivo" in p.resultado]
        st.session_state.chat.append({"rol": "assistant", "texto": r.texto, "pasos": r.pasos,
                                      "modelo": r.modelo, "archivos": archivos})
        st.rerun()

# pronóstico interactivo
with tab_pron:
    serie = st.selectbox("Serie", ["Equipo1", "Equipo2", "X", "Y", "Z"])
    hist = H.D["mensual"][[serie]].loc["2019":].reset_index()
    hist.columns = ["fecha", "valor"]
    pr = H.D["pronostico"].query("serie == @serie").copy()
    pr["fecha"] = pd.to_datetime(pr["mes"]) + pd.offsets.MonthEnd(0)
    color = COLORES[serie]

    base = alt.Chart(pr).encode(x=alt.X("fecha:T", title=None))
    banda90 = base.mark_area(opacity=0.15, color=color).encode(y=alt.Y("P5:Q", title="Precio"), y2="P95:Q")
    banda50 = base.mark_area(opacity=0.3, color=color).encode(y="P25:Q", y2="P75:Q")
    mediana = base.mark_line(strokeDash=[6, 4], color=color, strokeWidth=2).encode(
        y="P50:Q", tooltip=["mes", alt.Tooltip("P5:Q", format=",.0f"), alt.Tooltip("P50:Q", format=",.0f"),
                            alt.Tooltip("P95:Q", format=",.0f")])
    linea = alt.Chart(hist).mark_line(color=color, strokeWidth=2).encode(
        x="fecha:T", y="valor:Q", tooltip=[alt.Tooltip("fecha:T", format="%b-%Y"), alt.Tooltip("valor:Q", format=",.1f")])
    st.altair_chart((banda90 + banda50 + linea + mediana).properties(height=380), width="stretch")
    st.caption("Línea continua: histórico (promedio mensual). Discontinua: mediana. "
               "Banda oscura: 50% de probabilidad. Banda clara: 90%.")
    st.dataframe(pr.drop(columns=["serie", "fecha"]).set_index("mes").round(1), width="stretch")

# explicación IA vs agente
with tab_agente:
    st.subheader("¿En qué se diferencia un modelo de IA de un agente de IA?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Sistema de IA convencional** (modelo)
- Recibe una entrada y produce una salida: **predice, clasifica o genera**.
- Es **pasivo**: hace una sola cosa cuando se le pide.
- Ejemplo en esta prueba: el modelo de pronóstico. Le das los datos y te da bandas de precio.
- No decide qué hacer, no usa herramientas, no recuerda, no actúa en el mundo.
""")
    with c2:
        st.markdown("""
**Agente de IA**
- Tiene un **objetivo** y decide por sí mismo **cómo** alcanzarlo.
- **Autonomía:** elige qué pasos dar y en qué orden (ciclo percibir, decidir, actuar y observar).
- **Herramientas:** usa funciones externas (consultar el pronóstico, buscar noticias, calcular escenarios).
- **Memoria:** recuerda la conversación (corto plazo) y guarda notas entre sesiones (largo plazo).
- **Acción:** produce efectos reales (genera el Excel del presupuesto).
""")
    st.markdown("#### Cómo funciona este agente (código propio, sin frameworks)")
    st.graphviz_chart("""
digraph {
  rankdir=TB; nodesep=0.5; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#eeeeee", fontname="Helvetica", fontsize=13, width=3.2];
  edge [fontname="Helvetica", fontsize=11];
  U [label="Pregunta del usuario"]; P [label="1. Percibir\\n(instrucciones + memoria + historial)"];
  D [label="2. Decidir\\n(LLM: ¿respondo o uso herramienta?)", fillcolor="#d9edf7"];
  A [label="3. Actuar\\n(ejecuta herramientas en Python)"]; O [label="4. Observar\\n(resultados al LLM)"];
  R [label="Respuesta final\\n+ traza + archivos", fillcolor="#dff0d8"];
  T [label="Herramientas:\\npronóstico, presupuesto, escenarios\\nprecio justo, noticias, memoria, reporte", shape=note];
  U -> P -> D; D -> A [label=" necesita datos"]; A -> O; O -> D [label=" vuelve a decidir"]; D -> R [label=" ya sabe"];
  {rank=same; A; T} A -> T [style=dashed, arrowhead=none];
}""")
    st.info("Cada respuesta del chat muestra su **traza**: qué herramientas eligió el agente, con qué argumentos y qué "
            "obtuvo. Así la autonomía es visible y auditable.")

# metodología
with tab_metodo:
    r = H.resumen_modelo()
    st.subheader("Resultados del análisis")
    m = H.D["modelo"]
    st.markdown(f"""
| | Equipo 1 | Equipo 2 |
|---|---|---|
| Modelo | {m.loc['Equipo1','X']:.2f}*X + {m.loc['Equipo1','Y']:.2f}*Y + {m.loc['Equipo1','Z']:.2f}*Z | {m.loc['Equipo2','X']:.2f}*X + {m.loc['Equipo2','Y']:.2f}*Y + {m.loc['Equipo2','Z']:.2f}*Z |
| Determinante principal | Y | Z, luego Y |
| Ruido | Z | Ninguno |
""")
    st.markdown(f"- **Validación:** {r['validacion']}\n- **Pronóstico:** {r['pronostico']}\n"
                f"- **Horizonte:** {r['horizonte_recomendado']}")
    st.caption("Identificación (notebook 01, sección 4): X = petróleo Brent y Z = aluminio, confirmados contra series públicas; Y = acero (aprox.).")
