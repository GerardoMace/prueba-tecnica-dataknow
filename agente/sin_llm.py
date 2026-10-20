"""Modo de respaldo basado en reglas cuando no hay conexión al modelo."""
import re

from . import herramientas as H
from .nucleo import Paso, Respuesta


def _serie(texto):
    t = texto.lower()
    if "equipo 2" in t or "equipo2" in t:
        return "Equipo2"
    for s in ["X", "Y", "Z"]:
        if re.search(rf"\b{s.lower()}\b", t):
            return s
    return "Equipo1"


def _mes(texto):
    m = re.search(r"(20\d\d)-(\d\d)", texto)
    return m.group(0) if m else None


def responder(pregunta):
    t = pregunta.lower()
    pasos = []

    def usar(nombre, **args):
        res = H.ejecutar(nombre, args)
        pasos.append(Paso(nombre, args, res, 0))
        return res

    if any(k in t for k in ["noticia", "mercado", "actual", "hoy"]):
        s = _serie(pregunta)
        insumo = s if s in "XYZ" else ("Y" if s == "Equipo1" else "Z")
        r = usar("buscar_noticias", insumo_o_tema=insumo)
        lineas = [f"- {n['fecha']} - {n['titular']}" for n in r.get("noticias", [])]
        texto = (f"Titulares recientes sobre {H.NOMBRES[insumo]}:\n" + "\n".join(lineas)
                 if lineas else "No pude consultar noticias en este momento.")
    elif any(k in t for k in ["sube", "baja", "escenario", "si el", "si la"]):
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", pregunta)]
        pct = nums[0] if nums else 20
        insumo = _serie(pregunta) if _serie(pregunta) in "XYZ" else "Y"
        r = usar("simular_escenario", **{f"cambio_{insumo.lower()}_pct": pct})
        texto = "\n".join(f"- {e}: {v['cambio_%']:+.1f}% (de {v['precio_base']:,.0f} a {v['precio_escenario']:,.0f})"
                          for e, v in r["equipos"].items())
        texto = f"Si {insumo} cambia {pct:+.0f}%:\n{texto}"
    elif any(k in t for k in ["pronóstico", "pronostico", "proyección", "proyeccion", "cuánto costará", "precio"]):
        s, mes = _serie(pregunta), _mes(pregunta)
        r = usar("consultar_pronostico", serie=s, mes=mes)
        filas = r.get("meses", [])[:6]
        texto = f"Pronóstico de {s} (P50 central, rango 90% P5-P95):\n" + "\n".join(
            f"- {f['mes']}: {f['P50']:,.0f}  (P5 {f['P5']:,.0f} - P95 {f['P95']:,.0f})" for f in filas)
    else:
        r = usar("resumen_modelo")
        texto = ("Resumen del modelo:\n"
                 f"- Equipo1: {r['interpretacion']['Equipo1']}\n"
                 f"- Equipo2: {r['interpretacion']['Equipo2']}\n"
                 f"- Validación: {r['validacion']}\n"
                 f"- Horizonte: {r['horizonte_recomendado']}")
    texto += "\n\n_(Modo sin IA: respuestas por reglas. Activa la API key para preguntas libres.)_"
    return Respuesta(texto=texto, pasos=pasos, modelo="reglas (sin LLM)")
