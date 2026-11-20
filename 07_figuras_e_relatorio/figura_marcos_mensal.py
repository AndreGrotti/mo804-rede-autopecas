# Figura do relatório final: rede principal, série mensal com ajuste, contrafactual e os marcos do período
# (versão mensal da Figura D1 do relatório consolidado; marcos de scripts/intervencao_grafico_marcos.py).
# Uso: python figura_marcos_mensal.py <pasta_downloads> <pasta_saida>    Lê intervencao_mensal_completa_contrafactual.csv
import os, sys
AQUI = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(AQUI, "figuras_mensais.py"), encoding="utf-8").read()
     .split("# ------------------------------------------------------------------ F1")[0])
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

C = pd.read_csv(os.path.join(DL, "intervencao_mensal_completa_contrafactual.csv"), sep=";")
C["x"] = [xm(p.replace("-", "")) for p in C.periodo]
OLIVA, ROXO, VERDE_E, AZUL_E = "#8c8c00", "#7b3fa0", "#2e7d32", "#1565c0"
# (início, fim ou None, código, descrição, cor): com fim = barra na base; sem fim = linha pontilhada
MARCOS = [
    ("2018-01-01", "2018-12-31", "a", "fiscalização notificada às sementes (ao longo de 2018)", VERDE_E),
    ("2018-05-21", None, "b", "greve dos caminhoneiros (21 a 31/mai/2018)", ROXO),
    ("2018-04-25", "2019-12-31", "c", "crise cambial e recessão na Argentina, principal destino dos veículos exportados (2018–19)", OLIVA),
    ("2019-12-31", None, "1", "China comunica o surto de Wuhan (31/dez/2019)", VERM),
    ("2020-01-25", None, "2", "fábricas chinesas paradas após o Ano Novo Lunar: falta de componentes (fim de jan–fev/2020)", VERM),
    ("2020-02-26", None, "3", "primeiro caso confirmado no Brasil (26/fev/2020)", VERM),
    ("2020-03-11", None, "4", "OMS declara a pandemia (11/mar/2020)", VERM),
    ("2020-03-24", None, "5", "quarentena em SP; montadoras param (fim de mar/2020)", VERM),
    ("2021-03-01", "2022-06-30", "d", "falta de semicondutores: paradas nas montadoras (2021–22)", AZUL_E),
]
def marca(ax, rotulos=True):
    tr = ax.get_xaxis_transform(); alt_l = [0.94, 0.94, 0.86, 0.78, 0.70, 0.62]; alt_f = [0.03, 0.08, 0.13]; jl = jf = 0
    for ini, fim, cod, _, cor in MARCOS:
        x0 = pd.Timestamp(ini)
        if fim:
            y = alt_f[jf % 3]; jf += 1
            ax.plot([x0, pd.Timestamp(fim)], [y, y], transform=tr, lw=3.2, color=cor, alpha=0.6, solid_capstyle="butt", zorder=1)
            if rotulos: ax.text(x0, y + 0.012, cod, transform=tr, fontsize=6.5, color=cor, fontweight="bold")
        elif cod.isdigit():
            continue                               # marcos 1–5: uma faixa só, desenhada abaixo
        else:
            ax.axvline(x0, color=cor, lw=0.7, ls=":", zorder=1)
            if rotulos: ax.text(x0, alt_l[jl % len(alt_l)], cod, transform=tr, fontsize=6.5, color=cor, ha="center",
                                fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.3))
            jl += 1
    # marcos 1–5 (31/dez/2019 a 24/mar/2020) numa faixa cinza única, com rótulo no topo; detalhes na legenda da figura
    ax.axvspan(pd.Timestamp("2019-12-31"), pd.Timestamp("2020-03-24"), color="0.55", alpha=0.18, lw=0, zorder=0)
    if rotulos: ax.text(pd.Timestamp("2020-02-10"), 0.94, "1–5", transform=tr, fontsize=6.5, color=VERM, ha="center",
                        fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.3))

pain = [("log_vertices", "Estabelecimentos (log)", 1), ("grau_medio", "Grau médio", 1),
        ("densidade_x1000", "Densidade (por mil)", 1), ("transitividade", "Transitividade", 1)]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
for ax, (m, tit, k) in zip(axs.flat, pain):
    c = C[C.metrica == m]
    choque(ax)
    ax.plot(c.x, c.observado * k, "-", color="0.35", lw=0.7, zorder=3)
    ax.plot(c.x, c.ajustado * k, "-", color=AZUL, lw=1.0, zorder=4)
    ax.plot(c.x, c.sem_intervencao * k, "--", color=LAR, lw=1.2, zorder=4)
    marca(ax); ax.set_title(tit); eixo_anos(ax, passo=1)
    for t in ax.get_xticklabels(): t.set_fontsize(6.5)
    v = np.concatenate([c.observado, c.sem_intervencao.dropna()]) * k
    lo, hi = np.nanmin(v), np.nanmax(v); r = hi - lo
    ax.set_ylim(lo - 0.32 * r, hi + 0.30 * r)          # folga embaixo para as barras e em cima para os números
leit = [Line2D([], [], color="0.35", lw=0.8, label="observado"), Line2D([], [], color=AZUL, lw=1.0, label="ajustado pelo modelo"),
        Line2D([], [], color=LAR, lw=1.2, ls="--", label="contrafactual (até 2 anos)"),
        Patch(color=VERM, alpha=0.15, label="janela do choque (mar–jun/2020)")]
fig.legend(handles=leit, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.035))
fig.tight_layout(rect=(0, 0.03, 1, 1))
salva(fig, "f2m_marcos")
