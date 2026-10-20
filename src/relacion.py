import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import nnls
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, coint

INSUMOS = ["X", "Y", "Z"]


def ols_niveles(df, equipo, insumos=INSUMOS):
    X = sm.add_constant(df[insumos])
    return sm.OLS(df[equipo], X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})


def tabla_ols(res):
    return pd.DataFrame({
        "coeficiente": res.params,
        "error_std": res.bse,
        "p_valor": res.pvalues,
        "significativo_5%": res.pvalues < 0.05,
    })


def vif(df, insumos=INSUMOS):
    X = sm.add_constant(df[insumos]).values
    return pd.Series([variance_inflation_factor(X, i + 1) for i in range(len(insumos))],
                     index=insumos, name="VIF")


def prueba_cointegracion(df, equipo, insumos=INSUMOS):
    t, p, _ = coint(df[equipo], df[insumos])
    return {"estadistico": t, "p_valor": p, "cointegrados": p < 0.05}


def adf(s):
    est, p, *_ = adfuller(s.dropna(), autolag="AIC")
    return {"estadistico": est, "p_valor": p, "estacionaria": p < 0.05}


def pesos_nnls(df, equipo, insumos=INSUMOS):
    w, _ = nnls(df[insumos].values, df[equipo].values)
    return pd.Series(w, index=insumos, name=equipo)


def participacion_costo(df, pesos):
    aporte = df[pesos.index].mean() * pesos
    return (100 * aporte / aporte.sum()).rename("% del costo")


def elasticidades(df_mensual, equipo, insumos=INSUMOS):
    r = np.log(df_mensual).diff().dropna()
    return sm.OLS(r[equipo], sm.add_constant(r[insumos])).fit(cov_type="HC1")


def lasso_seleccion(df_mensual, equipo, insumos=INSUMOS):
    from sklearn.model_selection import TimeSeriesSplit
    r = np.log(df_mensual).diff().dropna()
    Xs = StandardScaler().fit_transform(r[insumos])
    ys = (r[equipo] - r[equipo].mean()) / r[equipo].std()
    m = LassoCV(cv=TimeSeriesSplit(5), random_state=0).fit(Xs, ys)
    return pd.Series(m.coef_, index=insumos, name=equipo)


def pesos_moviles(df_mensual, equipo, ventana=36,
                  insumos=INSUMOS):
    filas = {}
    for fin in range(ventana, len(df_mensual) + 1):
        tramo = df_mensual.iloc[fin - ventana:fin]
        filas[tramo.index[-1]] = pesos_nnls(tramo, equipo, insumos)
    return pd.DataFrame(filas).T


def validacion_fuera_muestra(df, equipo, corte,
                             insumos=INSUMOS):
    train, test = df[:corte], df[corte:].iloc[1:]
    w = pesos_nnls(train, equipo, insumos)
    pred = test[insumos] @ w
    err = test[equipo] - pred
    return {
        "pesos": w,
        "pred": pred,
        "MAPE_%": 100 * (err.abs() / test[equipo]).mean(),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "sesgo_%": 100 * (err / test[equipo]).mean(),
    }


def diagnostico_residuos(residuo, precio):
    lb = acorr_ljungbox(residuo, lags=[10], return_df=True)
    pct = 100 * residuo / precio
    return {
        "residuo_std_%": pct.std(),
        "residuo_media_%": pct.mean(),
        "ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
        "ruido_blanco": float(lb["lb_pvalue"].iloc[0]) > 0.05,
        "estacionario_ADF": adf(residuo)["estacionaria"],
    }
