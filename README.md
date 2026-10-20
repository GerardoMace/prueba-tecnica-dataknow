# Prueba Técnica - Data Scientist (DataKnow)

- **Autor:** José Gerardo Martínez Celaya  
- **Caso:** Estimación y pronóstico del costo de equipos críticos a partir de materias primas
- **Demo en línea del agente:** https://agente-costos-dataknow.streamlit.app/
## Descripción del proyecto

Este repositorio contiene la solución para estimar el costo de dos equipos de construcción (Equipo 1 y Equipo 2) en función de tres materias primas (X, Y, Z), proyectar sus precios con intervalos de confianza mediante simulación Monte Carlo y exponer los resultados a través de un agente conversacional en Streamlit conectado a Gemini.

## Organización de carpetas

- `data/raw/`: archivos originales proporcionados en la prueba.
- `data/external/`: series históricas públicas (FRED y SHFE) utilizadas para contrastar las materias primas.
- `data/processed/`: tablas limpias y resultados exportados por los notebooks.
- `notebooks/`:
  - `01_limpieza_y_calidad.ipynb`: revisión de datos, cambio de frecuencia en Y y cruce con series externas.
  - `02_relacion_insumos.ipynb`: pruebas de cointegración, estimación de pesos (OLS, NNLS, Lasso) y validación fuera de muestra.
  - `03_pronostico.ipynb`: backtesting de modelos de series de tiempo, simulación Monte Carlo y cálculo de presupuesto.
- `src/`: funciones auxiliares para carga de datos, regresiones, simulación y gráficas.
- `agente/`: código del agente conversacional y aplicación en Streamlit (`app.py`).
- `arquitectura/`: diagrama de la arquitectura propuesta en Azure (`.png` y `.drawio`).
- `documentos/`: documento escrito y diapositivas del caso.

## Instalación y ejecución

1. Instalar las librerías necesarias (Python 3.11+):

```bash
pip install -r requirements.txt
```

2. Los notebooks dentro de la carpeta `notebooks/` están numerados en orden secuencial (01, 02 y 03) y pueden abrirse y ejecutarse desde Jupyter o VS Code.

3. Para iniciar la interfaz web del agente, configurar la variable `GEMINI_API_KEY` en un archivo `.env` en la raíz del proyecto y ejecutar:

```bash
streamlit run agente/app.py
```

Si no se configura una clave de API, la aplicación se ejecuta en el modo de respaldo basado en reglas locales.
