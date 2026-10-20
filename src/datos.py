from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"
EXTERNOS = RAIZ / "data" / "external"

INSUMOS = ["X", "Y", "Z"]
EQUIPOS = ["Equipo1", "Equipo2"]


def _limpiar_serie(df, nombre):
    s = (
        df.dropna(subset=["Date", "Price"])
        .drop_duplicates(subset="Date", keep="last")
        .set_index("Date")
        .sort_index()["Price"]
        .astype(float)
    )
    s.name = nombre
    return s


def cargar_insumo(nombre, carpeta=RAW):
    ruta = carpeta / f"{nombre}.csv"
    if nombre == "Y":
        df = pd.read_csv(ruta, sep=";", decimal=",", encoding="utf-8-sig")
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    else:
        df = pd.read_csv(ruta, parse_dates=["Date"])
    return _limpiar_serie(df, nombre)


def cargar_insumos(carpeta=RAW):
    return {n: cargar_insumo(n, carpeta) for n in INSUMOS}


def cargar_historico(carpeta=RAW):
    df = pd.read_csv(carpeta / "historico_equipos.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset="Date").set_index("Date").sort_index()
    return df.rename(columns=lambda c: c.replace("Price_", ""))


def validar_historico(hist, insumos):
    filas = []
    for n, s in insumos.items():
        comun = hist.index.intersection(s.index)
        dif = (hist.loc[comun, n] - s.loc[comun]).abs()
        filas.append({
            "insumo": n,
            "fechas_historico": len(hist),
            "fechas_en_crudo": len(comun),
            "coinciden_%": round(100 * (dif < 1e-6).mean(), 2),
            "error_max": dif.max(),
        })
    return pd.DataFrame(filas).set_index("insumo")


def reporte_calidad(s):
    ret = np.log(s).diff()
    rachas = s.groupby((s.diff() != 0).cumsum()).size()
    return {
        "inicio": s.index.min().date(),
        "fin": s.index.max().date(),
        "observaciones": len(s),
        "nulos": int(s.isna().sum()),
        "precios_<=0": int((s <= 0).sum()),
        "dias_repetidos": int((s.diff() == 0).sum()),
        "racha_max_sin_cambio": int(rachas.max()),
        "saltos_>10%": int((ret.abs() > 0.10).sum()),
        "fines_de_semana": int((s.index.dayofweek >= 5).sum()),
    }


def frecuencia_real_Y(y):
    sin_cambio = y.diff() == 0
    return (100 * sin_cambio.groupby(y.index.year).mean()).round(1)


def racha_mas_larga(s):
    grupo = (s.diff() != 0).cumsum()
    tam = s.groupby(grupo).size()
    tramo = s[grupo == tam.idxmax()]
    return {"inicio": tramo.index.min().date(), "fin": tramo.index.max().date(),
            "dias": len(tramo), "precio": tramo.iloc[0]}


def saltos_equipos(hist, umbral=0.10):
    filas = []
    cambio = hist.pct_change() * 100
    for e in ["Equipo1", "Equipo2"]:
        ret = np.log(hist[e]).diff()
        for fecha in ret[ret.abs() > umbral].index:
            filas.append({"fecha": fecha.date(), "equipo": e, "equipo_%": cambio.loc[fecha, e],
                          **{f"{i}_%": cambio.loc[fecha, i] for i in INSUMOS}})
    return pd.DataFrame(filas).set_index(["equipo", "fecha"])


def a_mensual(df):
    return df.resample("ME").mean()


def guardar_procesados(hist):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    hist.to_csv(PROCESSED / "historico_diario.csv")
    a_mensual(hist).to_csv(PROCESSED / "historico_mensual.csv")


def cargar_externa(archivo):
    df = pd.read_csv(EXTERNOS / archivo, na_values=".")
    s = df.set_index(pd.to_datetime(df.iloc[:, 0])).iloc[:, 1].astype(float).dropna()
    s.name = archivo.replace(".csv", "")
    return s


def comparar_con_referencia(serie, referencia):
    m = pd.concat([serie.resample("MS").mean(), referencia.resample("MS").mean()], axis=1, join="inner").dropna()
    cambios = np.log(m).diff().dropna()
    return {
        "desde": m.index.min().strftime("%Y-%m"),
        "hasta": m.index.max().strftime("%Y-%m"),
        "meses": len(m),
        "corr_niveles": m.corr().iloc[0, 1],
        "corr_cambios_mensuales": cambios.corr().iloc[0, 1],
        "proporcion_mediana": (m.iloc[:, 0] / m.iloc[:, 1]).median(),
    }
