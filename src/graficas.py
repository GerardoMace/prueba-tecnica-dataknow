import matplotlib.pyplot as plt

from .datos import RAIZ

FIGURAS = RAIZ / "outputs" / "figuras"

COLORES = {
    "X": "#1f77b4",
    "Y": "#ff7f0e",
    "Z": "#2ca02c",
    "Equipo1": "#9467bd",
    "Equipo2": "#d62728",
}


def estilo():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (10, 4)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def guardar(fig, nombre):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / f"{nombre}.png", bbox_inches="tight")
