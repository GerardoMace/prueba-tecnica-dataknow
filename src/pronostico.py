import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

INSUMOS = ["X", "Y", "Z"]
MODELOS = ["Ingenuo", "ARIMA(1,1,0)", "ETS amortiguado", "VAR(1)"]


def _ingenuo(log_s, h):
    return np.repeat(log_s.iloc[-1], h)


def _arima(log_s, h):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(log_s.values, order=(1, 1, 0)).fit()
    return res.forecast(h)


def _ets(log_s, h):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ExponentialSmoothing(log_s.values, trend="add", damped_trend=True).fit()
    return res.forecast(h)


def _var(log_df, h):
    d = log_df.diff().dropna()
    res = VAR(d.values).fit(1)
    cambios = res.forecast(d.values[-1:], h)
    return pd.DataFrame(log_df.values[-1] + np.cumsum(cambios, axis=0), columns=log_df.columns)


def pronostico_puntual(log_df, h):
    out = {
        "Ingenuo": pd.DataFrame({c: _ingenuo(log_df[c], h) for c in log_df}),
        "ARIMA(1,1,0)": pd.DataFrame({c: _arima(log_df[c], h) for c in log_df}),
        "ETS amortiguado": pd.DataFrame({c: _ets(log_df[c], h) for c in log_df}),
        "VAR(1)": _var(log_df, h),
    }
    return {k: np.exp(v) for k, v in out.items()}


def backtest(mensual, inicio, h_max=12, ventana_min=60):
    log_df = np.log(mensual[INSUMOS])
    origenes = [i for i in range(len(log_df)) if log_df.index[i] >= pd.Timestamp(inicio)
                and i >= ventana_min and i + 1 < len(log_df)]
    filas = []
    for i in origenes:
        h = min(h_max, len(log_df) - 1 - i)
        preds = pronostico_puntual(log_df.iloc[:i + 1], h)
        reales = mensual[INSUMOS].iloc[i + 1:i + 1 + h].reset_index(drop=True)
        for modelo, p in preds.items():
            err = 100 * (p.values - reales.values) / reales.values
            for k in range(h):
                for j, c in enumerate(INSUMOS):
                    filas.append({"origen": log_df.index[i], "h": k + 1, "modelo": modelo,
                                  "insumo": c, "error_%": err[k, j]})
    return pd.DataFrame(filas)


def resumen_backtest(bt):
    return (bt.assign(abs_err=bt["error_%"].abs())
              .pivot_table(index=["insumo", "h"], columns="modelo", values="abs_err", aggfunc="mean")
              [MODELOS])


def simular(mensual, h, n=5000, bloque=3,
            semilla=42, escala=1.0, dinamica="caminata"):
    rng = np.random.default_rng(semilla)
    log_df = np.log(mensual[INSUMOS])
    d = log_df.diff().dropna().values
    if dinamica == "VAR":
        res = VAR(d).fit(1)
        A, c, resid = res.coefs[0], res.intercept, res.resid
    else:
        A, c, resid = np.zeros((3, 3)), np.zeros(3), d - d.mean(axis=0)
    resid = resid * escala
    T = len(resid)

    sims = np.empty((n, h, 3))
    for s in range(n):
        idx = []
        while len(idx) < h:
            ini = rng.integers(0, T - bloque + 1)
            idx.extend(range(ini, ini + bloque))
        choques = resid[idx[:h]]
        nivel = log_df.values[-1].copy()
        cambio = d[-1].copy()
        for k in range(h):
            cambio = c + A @ cambio + choques[k]
            nivel = nivel + cambio
            sims[s, k] = nivel
    return np.exp(sims)


def a_equipos(sims, pesos, ruido_mensual_pct,
              semilla=7):
    rng = np.random.default_rng(semilla)
    out = {}
    for e in pesos.index:
        w = pesos.loc[e, INSUMOS].values
        base = sims @ w
        ruido = rng.normal(0, ruido_mensual_pct[e] / 100, size=base.shape)
        out[e] = base * (1 + ruido)
    return out


def bandas(trayectorias, fechas,
           cuantiles=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)):
    q = np.quantile(trayectorias, cuantiles, axis=0).T
    return pd.DataFrame(q, index=fechas, columns=[f"P{int(x * 100)}" for x in cuantiles])


def cobertura_backtest(mensual, inicio, h, n=1000,
                       paso=3, escala=1.0, dinamica="caminata"):
    filas = []
    idx = [i for i in range(len(mensual)) if mensual.index[i] >= pd.Timestamp(inicio)
           and i + h < len(mensual)][::paso]
    for i in idx:
        sims = simular(mensual.iloc[:i + 1], h, n=n, semilla=i, escala=escala, dinamica=dinamica)
        real = mensual[INSUMOS].iloc[i + 1:i + 1 + h].values
        p05, p95 = np.quantile(sims, [0.05, 0.95], axis=0)
        p25, p50, p75 = np.quantile(sims, [0.25, 0.50, 0.75], axis=0)
        for k in range(h):
            for j, c in enumerate(INSUMOS):
                filas.append({"origen": mensual.index[i], "h": k + 1, "insumo": c,
                              "dentro_90": p05[k, j] <= real[k, j] <= p95[k, j],
                              "dentro_50": p25[k, j] <= real[k, j] <= p75[k, j],
                              "error_mediana_%": 100 * abs(p50[k, j] / real[k, j] - 1)})
    return pd.DataFrame(filas)


def calibrar_escala(mensual, inicio, h, objetivo=0.90,
                    escalas=np.arange(1.0, 2.01, 0.1), n=500,
                    dinamica="caminata"):
    filas = []
    for e in escalas:
        cb = cobertura_backtest(mensual, inicio, h, n=n, escala=e, dinamica=dinamica)
        filas.append({"escala": round(float(e), 2), "cobertura_90": cb["dentro_90"].mean(),
                      "cobertura_50": cb["dentro_50"].mean(),
                      "error_mediana_%": cb["error_mediana_%"].mean()})
    return pd.DataFrame(filas).set_index("escala")
