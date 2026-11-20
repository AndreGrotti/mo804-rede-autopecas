# Gera as figuras do relatório final em SVG (texto como texto), no estilo das figuras do Jupyter do projeto:
# pontos por trimestre, linha clara de média móvel de 4 trimestres, moldura completa, um tique por ano.
# Uso: python figuras.py <pasta_scripts> <pasta_downloads> <pasta_saida>
import os, sys, io, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.ticker as mticker
plt.show = lambda *a, **k: None
CEL, DL, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"svg.fonttype": "none", "font.family": "Segoe UI", "font.size": 8, "axes.titlesize": 9.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.linewidth": 0.7, "lines.linewidth": 1.1, "lines.markersize": 3, "figure.dpi": 100,
                     "legend.framealpha": 0.9, "legend.edgecolor": "0.8"})
AZUL, LAR, VERDE, CINZA, VERM = "#1f77b4", "#ff7f0e", "#2ca02c", "0.45", "#d62728"

def salva(fig, nome):
    plt.rcParams["svg.hashsalt"] = nome
    b = io.StringIO(); fig.savefig(b, format="svg", bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    s = b.getvalue(); s = s[s.index("<svg"):]
    s = re.sub(r'<svg([^>]*?) width="[^"]+" height="[^"]+"', r'<svg\1 width="100%"', s, count=1)
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<[^>]+>", lambda m: re.sub(r"(\d+\.\d)\d+", r"\1", m.group(0)), s)   # coordenadas com 1 casa; texto intacto
    # marcadores: a cor de preenchimento vai para a definição de cada marcador, e cada ponto fica só com a posição
    estilos = {}
    for m in re.finditer(r'<use xlink:href="#(m[0-9a-f]+)" x="[^"]+" y="[^"]+" style="([^"]*)"/>', s):
        estilos.setdefault(m.group(1), set()).add(m.group(2))
    for mid, st in estilos.items():
        if len(st) != 1: continue
        st = st.pop(); fill = re.search(r"fill:\s*([^;]+)", st)
        s = re.sub(rf'(<path id="{mid}" d="[^"]*" style=")', rf'\1fill: {fill.group(1).strip()}; ' if fill else r'\1', s)
        s = s.replace(f' style="{st}"/>', "/>") if all(len(v) == 1 for v in estilos.values()) else \
            re.sub(rf'(<use xlink:href="#{mid}" x="[^"]+" y="[^"]+") style="[^"]*"/>', r"\1/>", s)
    s = re.sub(r"\n\s+", "\n", s)
    open(os.path.join(OUT, nome + ".svg"), "w", encoding="utf-8").write(s)
    print(nome, round(len(s) / 1024), "KB")

def per(t): return pd.Period(str(t).replace("T", "Q"), "Q")
def meio(p): return p.start_time + (p.end_time - p.start_time) / 2
def choque(ax, rotulo=False):
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"), color=VERM, alpha=0.12, lw=0, zorder=0)
def eixo_anos(ax, ini="2016-10-01", fim="2026-03-01", passo=1):
    ax.set_xlim(pd.Timestamp(ini), pd.Timestamp(fim))
    ax.xaxis.set_major_locator(mdates.YearLocator(passo)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=0)
    for t in ax.get_xticklabels(): t.set_fontsize(7)
def serie(ax, x, y, cor, rot, ls="-", mm=True):
    ax.plot(x, y, ls, color=cor, marker="o", ms=2.6, lw=1.0, label=rot, zorder=3)
    if mm: ax.plot(x, pd.Series(np.asarray(y, float)).rolling(4).mean(), "-", color=cor, lw=2.2, alpha=0.4, zorder=2)

# ------------------------------------------------------------------ F1: a rede nos 36 trimestres
A = pd.read_csv(os.path.join(DL, "A_metricas (2).csv"), sep=";")
A["x"] = [meio(per(t)) for t in A.trimestre]
ver = {("trimestre", "completa"): ("rede principal", AZUL, "-"), ("trimestre", "so_SP"): ("só SP", LAR, "-"),
       ("12meses", "completa"): ("nós dos 12 meses", VERDE, "--")}
pain = [("vertices", "Estabelecimentos (nós)", 1), ("arestas", "Ligações (arestas)", 1), ("grau_medio", "Grau médio", 1),
        ("densidade", "Densidade (por mil)", 1000), ("transitividade", "Transitividade", 1),
        ("assortatividade", "Assortatividade de grau", 1)]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.9))
for ax, (c, tit, k) in zip(axs.flat, pain):
    choque(ax)
    for (r, v), (rot, cor, ls) in ver.items():
        d = A[(A.regra == r) & (A.versao == v)].sort_values("x")
        serie(ax, d.x, d[c] * k, cor, rot, ls)
    if c == "transitividade":
        d = A[(A.regra == "trimestre") & (A.versao == "completa")].sort_values("x")
        serie(ax, d.x, d.transitividade_aleat, "0.5", "aleatória (mesmos graus)")
    ax.set_title(tit); eixo_anos(ax, passo=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", ".") if abs(v) >= 1000 else f"{v:g}"))
h, l = axs[1][1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.2); salva(fig, "f1_rede")

# ------------------------------------------------------------------ F2/F3: análise de intervenção
CRIT = {"completa": {"log_vertices": 31.6, "log_arestas": 26.1, "grau_medio": 24.5, "densidade_x1000": 26.0,
                     "assortatividade": 22.6, "transitividade": 33.7},
        "so_SP": {"log_vertices": 29.6, "log_arestas": 23.1, "grau_medio": 22.3, "densidade_x1000": 22.2,
                  "assortatividade": 30.1, "transitividade": 23.4}}
def roda(versao):
    src = open(os.path.join(CEL, "intervencao_pandemia.py"), encoding="utf-8").read()
    for a, b in {'CAMINHO_CSV = "/home/candre/dados/rede/disciplina/serie/resultados/A_metricas.csv"':
                 f"CAMINHO_CSV = {os.path.join(DL, 'A_metricas (2).csv')!r}",
                 'VERSAO      = "completa"': f"VERSAO      = {versao!r}", "RODAR_TESTE = True": "RODAR_TESTE = False",
                 "NSIM        = 999": "NSIM        = 19", "GRAFICOS    = True": "GRAFICOS    = False",
                 'SALVAR_CSV  = "/home/candre/dados/rede/disciplina/serie/resultados/intervencao_coef.csv"': "SALVAR_CSV  = None"}.items():
        assert a in src, a; src = src.replace(a, b)
    ns = {"__name__": "__main__"}
    out = io.StringIO(); so = sys.stdout; sys.stdout = out
    try: exec(compile(src, "intervencao_pandemia.py", "exec"), ns)
    finally: sys.stdout = so
    return ns
MET = [("log_vertices", "Estabelecimentos (log)"), ("grau_medio", "Grau médio"),
       ("densidade_x1000", "Densidade (por mil)"), ("transitividade", "Transitividade")]
MARCOS = [("2019-12-31", "1"), ("2020-03-24", "5")]
for versao, nome in (("completa", "f2_intervencao_principal"), ("so_SP", "f3_intervencao_soSP")):
    ns = roda(versao); cf, vr, coef = ns["contraf"], ns["varred"], ns["coef"]
    q = coef[coef.termo == "degrau (nível)"].set_index("metrica").coef
    print(versao, "degraus:", {k: round(float(v), 4) for k, v in q.items()})
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.3), gridspec_kw={"height_ratios": [1.3, 1]})
    for j, (c, tit) in enumerate(MET):
        a1, a2 = axs[0][j], axs[1][j]
        d = cf[cf.metrica == c]; x = [meio(p) for p in d.index]
        choque(a1)
        a1.plot(x, d.observado, "o-", ms=2.6, lw=0.9, color="0.25", label="observado", zorder=3)
        a1.plot(x, d.ajustado, color=AZUL, lw=1.3, label="ajustado pelo modelo", zorder=2)
        a1.plot(x, d.sem_intervencao, "--", color=LAR, lw=1.5, label="sem o choque (contrafactual, até 2 anos)", zorder=2)
        a1.set_title(tit); eixo_anos(a1, passo=2)
        t = vr[vr.metrica == c]
        xs = [meio(per(s)) for s in t.data]
        a2.bar(xs, t.wald, width=62, color=[VERM if s == "2020T2" else "0.62" for s in t.data], zorder=2)
        a2.axhline(CRIT[versao][c], ls="--", color="k", lw=0.9, zorder=3)
        choque(a2); eixo_anos(a2, "2017-12-01", "2024-09-01", passo=2)
        if j == 0: a2.set_ylabel("força da quebra (Wald)")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h + [plt.Rectangle((0, 0), 1, 1, color="0.62"), plt.Rectangle((0, 0), 1, 1, color=VERM),
                    plt.Line2D([], [], ls="--", color="k", lw=0.9)],
               l + ["varredura: quebra em cada data", "data testada (2020T2)", "limite de 5% (sup-Wald)"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=1.0); salva(fig, nome)

# ------------------------------------------------------------------ F4: as relações através do choque
T = pd.read_csv(os.path.join(DL, "A_arestas_painel_trimestral.csv"), sep=";")
T = T[(T.painel == "todos") & (T.defasagem == 4)].copy(); T["x"] = [meio(per(t)) for t in T.trimestre]
Y = pd.read_csv(os.path.join(DL, "A_arestas_painel_anual.csv"), sep=";")
Y = Y[(Y.painel == "todos") & (Y.ano_base == 2017) & (Y.tipo == "todas")].sort_values("ano")
V = pd.read_csv(os.path.join(DL, "A_arestas_painel_2020.csv"), sep=";"); V = V[V.painel == "todos"].sort_values("ano")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
a = axs[0]; choque(a)
a.plot(T.x, T.retencao, "o-", ms=2.6, color=AZUL, label="em número")
a.plot(T.x, T.retencao_valor, "o--", ms=2.6, color=LAR, label="em valor")
a.set_title("Pares do trimestre que se repetem\num ano depois (no trimestre-base)"); eixo_anos(a, "2016-10-01", "2025-03-01", 2)
a.legend(loc="center right")
a = axs[1]
a.axvspan(2019.5, 2020.5, color=VERM, alpha=0.12, lw=0)
a.plot(Y.ano, Y.retencao, "o-", ms=3, color=AZUL, label="em número")
a.plot(Y.ano, Y.retencao_valor, "o--", ms=3, color=LAR, label="em valor")
a.plot(Y.ano, Y.retencao_acaso, ":", color=CINZA, lw=1.4, label="modelo aleatório")
a.set_ylim(0, 1.05); a.set_title("Pares de 2017 que voltam\na negociar em cada ano"); a.set_xticks([2018, 2020, 2022, 2024])
a.legend(loc="lower left")
a = axs[2]
cores = [VERM if y == 2020 else "0.55" for y in V.ano]
a.bar(V.ano - 0.2, V.ausentes_T2, 0.4, color=cores, alpha=0.5, label="pares de T1 ausentes em T2")
a.bar(V.ano + 0.2, V.dos_ausentes_voltam_T3_ou_T4, 0.4, color=cores, label="dos ausentes: voltam em T3 ou T4")
a.set_title("Ausentes no 2º trimestre e\nretorno no mesmo ano"); a.set_xticks(list(V.ano)[::2]); a.set_ylim(0, 0.8)
a.legend(loc="upper left", fontsize=6.8)
fig.tight_layout(w_pad=1.2); salva(fig, "f4_relacoes")

# ------------------------------------------------------------------ F5: as sementes através do choque
M = pd.read_csv(os.path.join(DL, "B_medianas (2).csv"), sep=";"); M["x"] = [meio(per(t)) for t in M.trimestre]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5))
for a, (c, tit) in zip(axs, [("parceiros", "Parceiros por semente"), ("herfindahl", "Herfindahl do valor"),
                             ("cinco_maiores", "Participação dos 5 maiores"), ("frac_fora_v", "Valor com parceiros\nde fora de SP")]):
    choque(a)
    for dr, cor in (("fornecedores", AZUL), ("clientes", LAR)):
        d = M[M.direcao == dr].sort_values("x")
        serie(a, d.x, d[c], cor, dr)
    a.set_title(tit); eixo_anos(a, passo=2)
axs[0].legend(loc="center right")
fig.tight_layout(w_pad=0.8); salva(fig, "f5_sementes")

# ------------------------------------------------------------------ F6: vínculos fortes e fracos através do choque
P = pd.read_csv(os.path.join(DL, "C_presenca_relativa (1).csv"), sep=";", encoding="utf-8")
P = P[(P.peso == "relacoes") & (P.faixa_tipo == "valor")].copy(); P["x"] = [meio(per(t)) for t in P.trimestre]
fx = [("menos de 1%", "0.5", "menos de 1% do valor da semente"), ("1% a 10%", LAR, "de 1% a 10%"), ("10% ou mais", AZUL, "10% ou mais")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for a, dr in zip(axs, ("fornecedores", "clientes")):
    choque(a)
    for f, cor, rot in fx:
        d = P[(P.direcao == dr) & (P.faixa == f)].sort_values("x")
        a.plot(d.x, d.relativa, "o-", ms=2.6, lw=1.0, color=cor, label=rot)
    a.axhline(1, color="k", lw=0.6); a.set_title(f"Relações com {dr}"); eixo_anos(a, "2017-10-01", "2026-03-01")
axs[0].set_ylabel("presença ÷ presença em 2017T4"); axs[0].legend(loc="lower left")
fig.tight_layout(w_pad=1.0); salva(fig, "f6_vinculos")

# ------------------------------------------------------------------ F7: a fronteira e a quebra de 2018T3
a = A[(A.regra == "trimestre") & (A.versao == "completa")].sort_values("x").set_index("trimestre")
s = A[(A.regra == "trimestre") & (A.versao == "so_SP")].sort_values("x").set_index("trimestre")
F = pd.read_csv(os.path.join(DL, "A_fronteira (2).csv"), sep=";"); F["x"] = [meio(per(t)) for t in F.trimestre]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax = axs[0]; choque(ax)
for d, cor, rot in ((a, AZUL, "rede principal"), (s, LAR, "só SP")):
    serie(ax, d.x, 100 * d.densidade / d.densidade.loc["2017T1":"2018T2"].mean(), cor, rot)
ax.axvline(meio(per("2018T3")), color="k", ls=":", lw=1)
ax.set_title("Densidade\n(índice: 2017T1–2018T2 = 100)"); eixo_anos(ax, passo=2); ax.legend(loc="lower left")
ax = axs[1]; choque(ax)
serie(ax, a.x, a.vertices * a.frac_nos_fora_SP, AZUL, "fora de SP")
ax.axvline(meio(per("2018T3")), color="k", ls=":", lw=1)
ax.set_title("Estabelecimentos de fora\nde SP no círculo"); eixo_anos(ax, passo=2)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", ".")))
ax = axs[2]; choque(ax)
serie(ax, F.x, 100 * F.frac_fora_n, AZUL, "em número")
serie(ax, F.x, 100 * F.frac_fora_v, LAR, "em valor")
ax.set_title("Parceiros das sementes\nfora de SP (%)"); eixo_anos(ax, passo=2); ax.legend(loc="center right")
fig.tight_layout(w_pad=1.0); salva(fig, "f7_fronteira")
