# Gráficos com marcos — rode DEPOIS da célula principal (usa S, contraf, varred, crits, DATA, TRANSICAO, HORIZONTE)
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

METRICAS  = list(S.columns)   # ou, por exemplo: ["grau_medio", "transitividade", "densidade_x1000"]
SALVAR_FIG = "/home/candre/dados/rede/disciplina/serie/resultados/intervencao_marcos.png"   # None = não grava

# (início, fim ou None, código, descrição, cor). Fim = None -> linha pontilhada; com fim -> barra na base.
MARCOS = [
    ("2018-01-01", "2018-12-31", "a", "notificação das 35 sementes (ao longo de 2018)", "tab:green"),
    ("2018-05-21", None,         "b", "greve dos caminhoneiros (21 a 31/mai/2018)", "tab:purple"),
    ("2018-04-25", "2019-12-31", "c", "crise cambial e recessão na Argentina, principal destino dos veículos (2018–19)", "tab:olive"),
    ("2019-12-31", None,         "1", "China comunica o surto de Wuhan (31/dez/2019)", "tab:red"),
    ("2020-01-25", None,         "2", "fábricas chinesas paradas após o Ano Novo Lunar: falta de componentes (fim jan–fev/2020)", "tab:red"),
    ("2020-02-26", None,         "3", "1º caso confirmado no Brasil (26/fev/2020)", "tab:red"),
    ("2020-03-11", None,         "4", "OMS declara pandemia (11/mar/2020)", "tab:red"),
    ("2020-03-24", None,         "5", "quarentena em SP; montadoras param (fim de mar/2020)", "tab:red"),
    ("2021-03-01", "2022-06-30", "d", "falta de semicondutores: paradas nas montadoras (2021–22, aprox.)", "tab:blue"),
]

def meio(p): return p.start_time + (p.end_time - p.start_time) / 2   # ponto no meio do trimestre

def marca(ax):
    tr = ax.get_xaxis_transform()            # x em datas, y em fração da altura do painel
    alt_l, alt_f = [0.95, 0.86, 0.77], [0.03, 0.08, 0.13]
    jl = jf = 0
    for ini, fim, cod, _, cor in MARCOS:
        x0 = pd.Timestamp(ini)
        if fim:
            y = alt_f[jf % 3]; jf += 1
            ax.plot([x0, pd.Timestamp(fim)], [y, y], transform=tr, lw=4, color=cor, alpha=0.55,
                    solid_capstyle="butt", zorder=1)
            ax.text(x0, y + 0.015, cod, transform=tr, fontsize=7, color=cor, fontweight="bold")
        else:
            ax.axvline(x0, color=cor, lw=0.8, ls=":", zorder=1)
            ax.text(x0, alt_l[jl % 3], cod, transform=tr, fontsize=7, color=cor, ha="center",
                    fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.4))
            jl += 1

n = len(METRICAS)
fig, axs = plt.subplots(n, 2, figsize=(14, 2.7 * n), gridspec_kw={"width_ratios": [1.5, 1]},
                        squeeze=False, sharex=True)
ini_choque, fim_choque = periodo(TRANSICAO[0]).start_time, periodo(TRANSICAO[-1]).end_time
for (ax, ax2), col in zip(axs, METRICAS):
    c = contraf[contraf.metrica == col]; x = [meio(p) for p in c.index]
    ax.axvspan(ini_choque, fim_choque, color="tab:red", alpha=0.08, zorder=0)
    ax.plot(x, c["observado"], "o-", ms=3, color="0.3", label="observado", zorder=3)
    ax.plot(x, c["ajustado"], color="tab:blue", label="ajustado", zorder=2)
    ax.plot(x, c["sem_intervencao"], "--", color="tab:orange", label=f"sem pandemia (até +{HORIZONTE} tri)", zorder=2)
    marca(ax); ax.set_title(col, fontsize=9)

    t = varred[varred.metrica == col]
    ax2.axvspan(ini_choque, fim_choque, color="tab:red", alpha=0.08, zorder=0)
    ax2.bar([meio(periodo(d)) for d in t.data], t["wald"], width=60, zorder=2,
            color=["tab:red" if d == DATA else "0.6" for d in t.data])
    ax2.axhline(crits[col], ls="--", color="k", lw=0.8)
    marca(ax2); ax2.set_title("força da quebra em cada data (tracejado: limite 5%)", fontsize=8)

for ax in axs.flat:
    ax.set_xlim(pd.Timestamp("2016-12-01"), pd.Timestamp("2026-02-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=7, labelbottom=True)
plt.tight_layout()

# Legenda 1 (à esquerda, acima dos painéis): como ler as curvas e a varredura
leitura = [
    Line2D([], [], marker="o", ms=3, color="0.3", label="observado"),
    Line2D([], [], color="tab:blue", label="ajustado pelo modelo"),
    Line2D([], [], ls="--", color="tab:orange", label=f"sem pandemia (contrafactual, até +{HORIZONTE} tri)"),
    Patch(color="tab:red", alpha=0.15, label=f"janela do choque ({TRANSICAO[0]}–{TRANSICAO[-1]})"),
    Patch(color="0.6", label="varredura: força da quebra (Wald) em cada data"),
    Patch(color="tab:red", label=f"varredura: data testada ({DATA})"),
    Line2D([], [], ls="--", color="k", lw=0.8, label="varredura: limite de 5% (sup-Wald)"),
]
# Legenda 2 (à direita): os marcos — barra grossa = período; linha pontilhada = data
marcos_h = [Line2D([], [], color=cor, lw=6, alpha=0.55, solid_capstyle="butt", label=f"{cod}: {txt}") if fim
            else Line2D([], [], color=cor, lw=1.2, ls=":", label=f"{cod}: {txt}")
            for ini, fim, cod, txt, cor in MARCOS]
fig.legend(handles=leitura, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8, frameon=False,
           title="Como ler", title_fontsize=9, alignment="left")
fig.legend(handles=marcos_h, loc="lower left", bbox_to_anchor=(0.30, 1.0), fontsize=8, frameon=False,
           title="Marcos (barra grossa na base = período; linha pontilhada = data)", title_fontsize=9,
           alignment="left")
fig.text(0.01, 0.0, "Cada ponto fica no meio do seu trimestre. Na varredura, 2020T1–2020T3 têm a mesma altura "
         "por construção (os pulsos absorvem esses trimestres).", va="top", fontsize=8)
if SALVAR_FIG:
    fig.savefig(SALVAR_FIG, dpi=150, bbox_inches="tight"); print("Gravado:", SALVAR_FIG)
plt.show()
