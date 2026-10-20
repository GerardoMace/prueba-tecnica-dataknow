"""Funciones de consulta y cálculo para el agente."""
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

from src.datos import PROCESSED, RAIZ

REPORTES = RAIZ / "outputs" / "reportes"
MEMORIA = RAIZ / "agente" / "memoria.json"

NOMBRES = {"X": "petróleo Brent", "Y": "acero (identificación aproximada)", "Z": "aluminio"}
BUSQUEDAS = {"X": "Brent crude oil price", "Y": "steel price", "Z": "aluminium price LME"}


def _cargar():
    return {
        "modelo": pd.read_csv(PROCESSED / "modelo_relacion.csv", index_col=0),
        "pronostico": pd.read_csv(PROCESSED / "pronostico.csv"),
        "presupuesto": pd.read_csv(PROCESSED / "presupuesto_6m.csv"),
        "escenarios": pd.read_csv(PROCESSED / "escenarios.csv", index_col=0),
        "error": pd.read_csv(PROCESSED / "error_por_horizonte.csv", index_col=0),
        "mensual": pd.read_csv(PROCESSED / "historico_mensual.csv", index_col=0, parse_dates=True),
    }


D = _cargar()


def _redondear(obj):
    if isinstance(obj, dict):
        return {k: _redondear(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redondear(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 2)
    return obj


def resumen_modelo():
    """Modelo de cada equipo, métricas de validación y horizonte recomendado."""
    m = D["modelo"]
    ultimo = D["mensual"].iloc[-1]
    return _redondear({
        "datos_hasta": D["mensual"].index[-1].strftime("%Y-%m"),
        "pesos": {e: {i: m.loc[e, i] for i in ["X", "Y", "Z"]} for e in m.index},
        "participacion_en_costo_%": {
            e: {i: 100 * m.loc[e, i] * ultimo[i] / sum(m.loc[e, j] * ultimo[j] for j in "XYZ") for i in "XYZ"}
            for e in m.index},
        "interpretacion": {
            "Equipo1": "0.20*X + 0.80*Y. Y explica ~97% del costo; Z no influye (p=0.999).",
            "Equipo2": "1/3 de X, Y y Z. Z explica ~76% del costo, Y ~21%, X ~3%.",
        },
        "validacion": "R² > 0.99; entrenado 2010-2019 y probado 2020-2023: error promedio 1.5%, sin sesgo. "
                      "Cinco métodos (OLS, cointegración, NNLS, elasticidades, Lasso) coinciden.",
        "pronostico": "Caminata aleatoria + Monte Carlo (10,000 escenarios, cambios históricos en bloques), "
                      "bandas calibradas: la del 90% contuvo al real 90% de las veces en 2016-2023.",
        "horizonte_recomendado": "6 meses con re-pronóstico mensual; 7-12 meses solo como referencia de riesgo.",
        "error_esperado_por_horizonte_%": D["error"].loc[[1, 3, 6, 12]].to_dict(orient="index"),
        "insumos": NOMBRES,
    })


def consultar_pronostico(serie, mes=None):
    """Bandas de probabilidad (P5..P95) de una serie para un mes (AAAA-MM) o todos los meses."""
    p = D["pronostico"]
    p = p[p["serie"].str.lower() == serie.lower()]
    if p.empty:
        return {"error": f"Serie '{serie}' no existe. Opciones: Equipo1, Equipo2, X, Y, Z."}
    if mes:
        p = p[p["mes"] == mes]
        if p.empty:
            return {"error": f"Sin pronóstico para {mes}. El pronóstico cubre 2023-09 a 2024-08."}
    return _redondear({
        "serie": serie,
        "ultimo_valor_observado": float(D["mensual"][p["serie"].iloc[0]].iloc[-1]),
        "nota": "P50 = escenario central; P5-P95 = 90% de probabilidad; P25-P75 = 50%.",
        "meses": p.drop(columns="serie").to_dict(orient="records"),
    })


def calcular_presupuesto(compras, nivel="P80"):
    """Presupuesto de una lista de compras [{equipo, mes, unidades}] con el percentil elegido."""
    p = D["pronostico"]
    filas, total, total_base = [], 0.0, 0.0
    for c in compras:
        fila = p[(p["serie"] == c["equipo"]) & (p["mes"] == c["mes"])]
        if fila.empty:
            return {"error": f"Sin pronóstico para {c['equipo']} en {c['mes']} (cubre 2023-09 a 2024-08)."}
        precio, base = float(fila[nivel].iloc[0]), float(fila["P50"].iloc[0])
        u = float(c["unidades"])
        filas.append({**c, "precio_unitario": precio, "subtotal": precio * u,
                      "subtotal_P50": base * u, "reserva": (precio - base) * u})
        total += precio * u
        total_base += base * u
    return _redondear({"nivel": nivel, "detalle": filas, "total": total, "total_P50": total_base,
                       "reserva_contingencia": total - total_base,
                       "nota": f"{nivel}: probabilidad de no excederlo = {nivel[1:]}%."})


def simular_escenario(cambio_x_pct=0, cambio_y_pct=0, cambio_z_pct=0):
    """Impacto en el precio de los equipos si los insumos cambian en los % indicados (vs. ago-2023)."""
    m, ultimo = D["modelo"], D["mensual"].iloc[-1]
    cambios = {"X": cambio_x_pct, "Y": cambio_y_pct, "Z": cambio_z_pct}
    out = {}
    for e in m.index:
        antes = sum(m.loc[e, i] * ultimo[i] for i in "XYZ")
        despues = sum(m.loc[e, i] * ultimo[i] * (1 + cambios[i] / 100) for i in "XYZ")
        out[e] = {"precio_base": antes, "precio_escenario": despues, "cambio_%": 100 * (despues / antes - 1)}
    return _redondear({"cambios_insumos_%": cambios, "equipos": out})


def costo_con_precios(x=None, y=None, z=None,
                      cotizacion=None, equipo=None):
    """Precio justo de los equipos con precios de insumos dados (p. ej. precios actuales de noticias).
    Si se da una cotización de proveedor, dice si está dentro del rango de mercado."""
    m, ultimo = D["modelo"], D["mensual"].iloc[-1]
    precios = {"X": x if x is not None else ultimo["X"], "Y": y if y is not None else ultimo["Y"],
               "Z": z if z is not None else ultimo["Z"]}
    out = {"precios_insumos_usados": precios,
           "insumos_no_dados_se_toman_de": D["mensual"].index[-1].strftime("%Y-%m"), "equipos": {}}
    for e in m.index:
        justo = sum(m.loc[e, i] * precios[i] for i in "XYZ")
        sd = m.loc[e, "residuo_std_diario_%"] / 100
        r = {"precio_justo": justo, "rango_95%": [justo * (1 - 1.96 * sd), justo * (1 + 1.96 * sd)]}
        if cotizacion is not None and (equipo is None or equipo == e):
            dif = 100 * (cotizacion / justo - 1)
            r["cotizacion"] = cotizacion
            r["diferencia_%"] = dif
            r["veredicto"] = ("cara: negociar" if cotizacion > r["rango_95%"][1]
                              else "barata: verificar calidad" if cotizacion < r["rango_95%"][0]
                              else "dentro del rango de mercado")
        out["equipos"][e] = r
    return _redondear(out)


def ubicar_en_pronostico(serie, valor, mes="2024-08"):
    """En qué percentil del pronóstico cae un valor observado (para contrastar precios actuales con el modelo)."""
    p = D["pronostico"]
    fila = p[(p["serie"] == serie) & (p["mes"] == mes)]
    if fila.empty:
        return {"error": "Serie o mes no válidos."}
    f = fila.iloc[0]
    qs = [("P5", 5), ("P10", 10), ("P25", 25), ("P50", 50), ("P75", 75), ("P90", 90), ("P95", 95)]
    pos = "por debajo de P5" if valor < f["P5"] else "por encima de P95" if valor > f["P95"] else None
    if pos is None:
        for (a, qa), (b, qb) in zip(qs, qs[1:]):
            if f[a] <= valor <= f[b]:
                pos = f"entre {a} y {b}"
                break
    return _redondear({"serie": serie, "mes": mes, "valor": valor, "ubicacion": pos,
                       "bandas": {k: float(f[k]) for k, _ in qs},
                       "lectura": "Fuera de P5-P95 = el mercado salió del rango previsto; conviene re-entrenar con datos nuevos."})


def buscar_noticias(insumo_o_tema, max_resultados=6):
    """Titulares recientes de Google News (fuente externa, gratuita, sin API key)."""
    q = BUSQUEDAS.get(insumo_o_tema.upper(), insumo_o_tema)
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(q + " when:30d")
           + "&hl=en-US&gl=US&ceid=US:en")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raiz = ET.fromstring(urllib.request.urlopen(req, timeout=15).read())
    except Exception as e:
        return {"error": f"No se pudieron consultar noticias: {e}"}
    items = []
    for it in raiz.iter("item"):
        fecha = parsedate_to_datetime(it.findtext("pubDate"))
        items.append({"fecha": fecha.strftime("%Y-%m-%d"), "titular": it.findtext("title"),
                      "fuente": it.findtext("source"), "link": it.findtext("link")})
    items.sort(key=lambda i: i["fecha"], reverse=True)
    return {"busqueda": q, "consultado": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "noticias": items[:max_resultados]}


def guardar_nota(nota):
    """Memoria de largo plazo: guarda un dato importante del usuario para futuras conversaciones."""
    notas = leer_memoria()
    notas.append({"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "nota": nota})
    MEMORIA.write_text(json.dumps(notas, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"guardado": True, "total_notas": len(notas)}


def leer_memoria():
    try:
        return json.loads(MEMORIA.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def generar_reporte(compras, nivel="P80", titulo="Presupuesto de equipos"):
    """Acción: genera un archivo Excel con el presupuesto y sus supuestos, listo para finanzas."""
    pres = calcular_presupuesto(compras, nivel)
    if "error" in pres:
        return pres
    REPORTES.mkdir(parents=True, exist_ok=True)
    ruta = REPORTES / f"presupuesto_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    with pd.ExcelWriter(ruta) as xw:
        pd.DataFrame(pres["detalle"]).to_excel(xw, sheet_name="Detalle", index=False)
        pd.DataFrame([{"titulo": titulo, "nivel": nivel, "total": pres["total"], "total_P50": pres["total_P50"],
                       "reserva_contingencia": pres["reserva_contingencia"],
                       "metodo": "Modelo de insumos + Monte Carlo calibrado; datos hasta 2023-08"}]
                     ).to_excel(xw, sheet_name="Resumen", index=False)
    return {"archivo": str(ruta), "total": pres["total"], "reserva_contingencia": pres["reserva_contingencia"]}


_COMPRAS = {
    "type": "array",
    "items": {"type": "object", "properties": {
        "equipo": {"type": "string", "enum": ["Equipo1", "Equipo2"]},
        "mes": {"type": "string", "description": "AAAA-MM, entre 2023-09 y 2024-08"},
        "unidades": {"type": "number"}}, "required": ["equipo", "mes", "unidades"]},
}
_NIVEL = {"type": "string", "enum": ["P50", "P75", "P80", "P90", "P95"],
          "description": "Percentil: P50 central, P80 con contingencia, P90 conservador"}

HERRAMIENTAS = [
    {"name": "resumen_modelo",
     "description": "Explica qué insumos determinan cada equipo (modelos, % del costo), la validación del modelo, "
                    "el método de pronóstico y el horizonte recomendado. Úsala para preguntas sobre metodología o resultados.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "consultar_pronostico",
     "description": "Bandas de probabilidad del precio pronosticado de Equipo1, Equipo2, X, Y o Z por mes (2023-09 a 2024-08).",
     "parameters": {"type": "object", "properties": {
         "serie": {"type": "string", "enum": ["Equipo1", "Equipo2", "X", "Y", "Z"]},
         "mes": {"type": "string", "description": "Opcional, AAAA-MM"}}, "required": ["serie"]}},
    {"name": "calcular_presupuesto",
     "description": "Presupuesto total de un plan de compras con reserva de contingencia.",
     "parameters": {"type": "object", "properties": {"compras": _COMPRAS, "nivel": _NIVEL}, "required": ["compras"]}},
    {"name": "simular_escenario",
     "description": "Qué pasa con el precio de los equipos si X, Y o Z suben/bajan cierto %.",
     "parameters": {"type": "object", "properties": {
         "cambio_x_pct": {"type": "number"}, "cambio_y_pct": {"type": "number"}, "cambio_z_pct": {"type": "number"}}}},
    {"name": "costo_con_precios",
     "description": "Precio justo de cada equipo dados precios de insumos (por ejemplo, precios actuales encontrados "
                    "en noticias) y, opcional, si una cotización de proveedor es cara, barata o de mercado. "
                    "Los insumos que no se den se toman automáticamente del último mes con datos (ago-2023): "
                    "para evaluar con precios de ago-2023 basta con pasar la cotización y el equipo.",
     "parameters": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
         "cotizacion": {"type": "number"}, "equipo": {"type": "string", "enum": ["Equipo1", "Equipo2"]}}}},
    {"name": "ubicar_en_pronostico",
     "description": "Ubica un precio observado (p. ej. actual) dentro de las bandas pronosticadas para saber si el "
                    "mercado salió del rango previsto.",
     "parameters": {"type": "object", "properties": {
         "serie": {"type": "string", "enum": ["Equipo1", "Equipo2", "X", "Y", "Z"]},
         "valor": {"type": "number"}, "mes": {"type": "string", "description": "AAAA-MM; por defecto 2024-08"}},
         "required": ["serie", "valor"]}},
    {"name": "buscar_noticias",
     "description": "Titulares recientes (últimos 30 días) de un insumo (X, Y, Z) o de cualquier tema de mercado, "
                    "con fecha y fuente. Úsala para contexto externo: tendencias, precios actuales, economía.",
     "parameters": {"type": "object", "properties": {
         "insumo_o_tema": {"type": "string", "description": "X, Y, Z o un tema libre en inglés"}},
         "required": ["insumo_o_tema"]}},
    {"name": "guardar_nota",
     "description": "Guarda en memoria de largo plazo un dato del usuario o del proyecto que sirva en futuras "
                    "conversaciones (plan de compras, preferencias, decisiones).",
     "parameters": {"type": "object", "properties": {"nota": {"type": "string"}}, "required": ["nota"]}},
    {"name": "generar_reporte",
     "description": "Acción: crea un Excel con el presupuesto de un plan de compras para enviarlo a finanzas.",
     "parameters": {"type": "object", "properties": {
         "compras": _COMPRAS, "nivel": _NIVEL, "titulo": {"type": "string"}}, "required": ["compras"]}},
]

FUNCIONES = {
    "resumen_modelo": resumen_modelo,
    "consultar_pronostico": consultar_pronostico,
    "calcular_presupuesto": calcular_presupuesto,
    "simular_escenario": simular_escenario,
    "costo_con_precios": costo_con_precios,
    "ubicar_en_pronostico": ubicar_en_pronostico,
    "buscar_noticias": buscar_noticias,
    "guardar_nota": guardar_nota,
    "generar_reporte": generar_reporte,
}


def ejecutar(nombre, argumentos):
    """Ejecuta una herramienta por nombre. Los errores se regresan como dato para que el agente se corrija."""
    if nombre not in FUNCIONES:
        return {"error": f"Herramienta desconocida: {nombre}"}
    try:
        return FUNCIONES[nombre](**argumentos)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
