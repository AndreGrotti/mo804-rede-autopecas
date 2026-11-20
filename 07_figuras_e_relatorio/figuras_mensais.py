# Figuras 1–3 do relatório final na série MENSAL, em SVG (texto como texto), no mesmo estilo das demais.
# Uso: python figuras_mensais.py <pasta_downloads> <pasta_saida>
# Lê: "M_metricas (1).csv" e intervencao_mensal_{completa,so_SP}_{contrafactual,varredura}.csv
import os, sys, io, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.ticker as mticker
DL, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"svg.fonttype": "none", "font.family": "Segoe UI", "font.size": 8, "axes.titlesize": 9.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.linewidth": 0.7, "lines.linewidth": 1.0, "figure.dpi": 100,
                     "legend.framealpha": 0.9, "legend.edgecolor": "0.8"})
AZUL, LAR, VERDE, VERM = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"

def _n(v):  # décimos inteiros -> texto curto ("12", "-3.4", ".5")
    s = f"{v / 10:.1f}".removesuffix(".0")
    if s == "-0": s = "0"
    return "." + s[2:] if s.startswith("0.") else ("-." + s[3:] if s.startswith("-0.") else s)

def compacta(d):
    # caminhos só com M/L/Z: primeiro ponto absoluto, demais relativos (l), em décimos de pixel, sem drift
    tk = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:e[-+]?\d+)?", d)
    if not tk or any(t.isalpha() and t not in "MLZz" for t in tk): return d
    pts, cmd, nums = [], None, []          # pts: ("M"|"L", x, y) ou ("Z",)
    def fecha():
        for k in range(0, len(nums) - 1, 2):
            pts.append(("M" if (cmd == "M" and k == 0) else "L", round(float(nums[k]) * 10), round(float(nums[k + 1]) * 10)))
    for t in tk:
        if t.isalpha():
            if cmd: fecha()
            cmd, nums = t.upper(), []
            if cmd == "Z": pts.append(("Z",)); cmd = None
        else: nums.append(t)
    if cmd: fecha()
    out, px, py, ult, x0, y0 = [], 0, 0, None, 0, 0
    for p in pts:
        if p[0] == "Z": out.append("z"); ult = "z"; px, py = x0, y0; continue
        c, x, y = p
        if c == "M": out.append("M" + _n(x) + " " + _n(y)); ult = "M"; x0, y0 = x, y
        else: out.append(("l" if ult != "l" else "") + _n(x - px) + " " + _n(y - py)); ult = "l"
        px, py = x, y
    s = " ".join(out)
    return re.sub(r" (?=-)|(?<=[Mlz]) ", "", s).replace(" l", "l").replace(" z", "z")

def salva(fig, nome):
    plt.rcParams["svg.hashsalt"] = nome
    b = io.StringIO(); fig.savefig(b, format="svg", bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    s = b.getvalue(); s = s[s.index("<svg"):]
    s = re.sub(r'<svg([^>]*?) width="[^"]+" height="[^"]+"', r'<svg\1 width="100%"', s, count=1)
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<[^>]+>", lambda m: re.sub(r"(\d+\.\d)\d+", r"\1", m.group(0)), s)
    estilos = {}
    for m in re.finditer(r'<use xlink:href="#(m[0-9a-f]+)" x="[^"]+" y="[^"]+" style="([^"]*)"/>', s):
        estilos.setdefault(m.group(1), set()).add(m.group(2))
    for mid, st in estilos.items():
        if len(st) != 1: continue
        st = st.pop(); fill = re.search(r"fill:\s*([^;]+)", st)
        s = re.sub(rf'(<path id="{mid}" d="[^"]*" style=")', rf'\1fill: {fill.group(1).strip()}; ' if fill else r'\1', s)
        s = re.sub(rf'(<use xlink:href="#{mid}" x="[^"]+" y="[^"]+") style="[^"]*"/>', r"\1/>", s)
    s = re.sub(r'(<path[^>]*? d=")([^"]*)(")', lambda m: m.group(1) + compacta(m.group(2)) + m.group(3), s)
    s = re.sub(r"\n\s+", "\n", s)
    open(os.path.join(OUT, nome + ".svg"), "w", encoding="utf-8").write(s)
    print(nome, round(len(s) / 1024), "KB")

def xm(p): return pd.Period(str(p), "M").start_time + pd.Timedelta(days=14)
def choque(ax):
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"), color=VERM, alpha=0.12, lw=0, zorder=0)
def eixo_anos(ax, ini="2016-11-01", fim="2026-02-01", passo=2):
    ax.set_xlim(pd.Timestamp(ini), pd.Timestamp(fim))
    ax.xaxis.set_major_locator(mdates.YearLocator(passo)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for t in ax.get_xticklabels(): t.set_fontsize(7)
def serie(ax, x, y, cor, rot, mm=True):
    ax.plot(x, y, "-", color=cor, lw=0.7, label=rot, zorder=3)
    if mm: ax.plot(x, pd.Series(np.asarray(y, float)).rolling(12, center=True).mean(), "-", color=cor, lw=2.2, alpha=0.4, zorder=2)
milhar = mticker.FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", ".") if abs(v) >= 1000 else f"{v:g}")

# ------------------------------------------------------------------ F1: a rede nos 108 meses
# 19/11: só SP pela regra nova de UF (emitente > Receita > destinatário), versão "so_SP_nova" do M_metricas_v2.csv
M = pd.read_csv(os.path.join(DL, "M_metricas_v2.csv"), sep=";")
M = M[M.versao != "so_SP"].replace({"versao": {"so_SP_nova": "so_SP"}})
M["x"] = [xm(p) for p in M.periodo]
ver = {"completa": ("rede principal", AZUL), "so_SP": ("só SP", LAR), "painel_todos": ("núcleo persistente (painel)", VERDE)}
pain = [("vertices", "Estabelecimentos (nós)", 1), ("arestas", "Ligações (arestas)", 1), ("grau_medio", "Grau médio", 1),
        ("densidade", "Densidade (por mil)", 1000), ("transitividade", "Transitividade", 1),
        ("assortatividade", "Assortatividade de grau", 1)]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.9))
for ax, (c, tit, k) in zip(axs.flat, pain):
    choque(ax)
    for v, (rot, cor) in ver.items():
        d = M[M.versao == v].sort_values("x")
        serie(ax, d.x, d[c] * k, cor, rot)
    if c == "transitividade":
        d = M[M.versao == "completa"].sort_values("x")
        serie(ax, d.x, d.transitividade_aleat, "0.5", "aleatória (mesmos graus)")
    ax.set_title(tit); eixo_anos(ax); ax.yaxis.set_major_formatter(milhar)
h, l = axs[1][1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.2); salva(fig, "f1_rede")

# ------------------------------------------------------------------ F2/F3: análise de intervenção mensal
MET = [("log_vertices", "Estabelecimentos (log)"), ("grau_medio", "Grau médio"),
       ("densidade_x1000", "Densidade (por mil)"), ("transitividade", "Transitividade")]
for arq, nome in (("intervencao_mensal_completa", "f2_intervencao_principal"),
                  ("intervencao_soSP_nova_mensal", "f3_intervencao_soSP")):      # só SP: célula I1, regra nova
    cf = pd.read_csv(os.path.join(DL, f"{arq}_contrafactual.csv"), sep=";")
    vr = pd.read_csv(os.path.join(DL, f"{arq}_varredura.csv"), sep=";")
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.3), gridspec_kw={"height_ratios": [1.3, 1]})
    for j, (c, tit) in enumerate(MET):
        a1, a2 = axs[0][j], axs[1][j]
        d = cf[cf.metrica == c]; x = [xm(p) for p in d.periodo]
        choque(a1)
        a1.plot(x, d.observado, "-", lw=0.7, color="0.3", label="observado", zorder=3)
        a1.plot(x, d.ajustado, color=AZUL, lw=1.0, label="ajustado pelo modelo", zorder=2)
        a1.plot(x, d.sem_intervencao, "--", color=LAR, lw=1.3, label="sem o choque (contrafactual, até 2 anos)", zorder=2)
        a1.set_title(tit); eixo_anos(a1)
        t = vr[vr.metrica == c]
        xs = [xm(s) for s in t.data]
        a2.bar(xs, t.wald, width=24, color=[VERM if s == "2020-07" else "0.62" for s in t.data], zorder=2)
        a2.axhline(t.limite_5pct.iloc[0], ls="--", color="k", lw=0.9, zorder=3)
        choque(a2); eixo_anos(a2, "2018-01-01", "2024-09-01")
        if j == 0: a2.set_ylabel("força da quebra (Wald)")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h + [plt.Rectangle((0, 0), 1, 1, color="0.62"), plt.Rectangle((0, 0), 1, 1, color=VERM),
                    plt.Line2D([], [], ls="--", color="k", lw=0.9)],
               l + ["varredura: quebra em cada mês", "data testada (jul/2020)", "limite de 5% (sup-Wald)"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=1.0); salva(fig, nome)

# ------------------------------------------------------------------ F4–F7 mensais (relações, sementes, vínculos, fronteira)
CINZA = "0.45"
lemes = lambda nome: pd.read_csv(os.path.join(DL, nome), sep=";")

# F4 (Figura 5): as relações através do choque
R1 = lemes("R_retencao_mensal.csv"); R1["x"] = [xm(p) for p in R1.periodo]
R2 = lemes("R_choque_mensal.csv")
Y = lemes("A_arestas_painel_anual.csv")
Y = Y[(Y.painel == "todos") & (Y.ano_base == 2017) & (Y.tipo == "todas")].sort_values("ano")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
a = axs[0]
a.axvspan(pd.Timestamp("2019-03-01"), pd.Timestamp("2019-06-30"), color=VERM, alpha=0.12, lw=0, zorder=0)
choque(a)
a.plot(R1.x, R1.retencao, "-", lw=0.9, color=AZUL, label="em número")
a.plot(R1.x, R1.retencao_valor, "-", lw=0.9, color=LAR, label="em valor")
a.set_title("Pares do mês que se repetem\num ano depois (no mês-base)"); eixo_anos(a, "2016-11-01", "2025-02-01")
a.set_ylim(0.3, 1.0); a.legend(loc="lower right")
a = axs[1]
a.axvspan(2019.5, 2020.5, color=VERM, alpha=0.12, lw=0)
a.plot(Y.ano, Y.retencao, "o-", ms=3, color=AZUL, label="em número")
a.plot(Y.ano, Y.retencao_valor, "o--", ms=3, color=LAR, label="em valor")
a.plot(Y.ano, Y.retencao_acaso, ":", color=CINZA, lw=1.4, label="modelo aleatório")
a.set_ylim(0, 1.05); a.set_title("Pares de 2017 que voltam\na negociar em cada ano"); a.set_xticks([2018, 2020, 2022, 2024])
a.legend(loc="lower left")
a = axs[2]
cores = [VERM if y == 2020 else "0.55" for y in R2.ano]
a.bar(R2.ano - 0.2, R2.frac_ausentes, 0.4, color=cores, alpha=0.5, label="pares de jan–fev ausentes em abr–mai")
a.bar(R2.ano + 0.2, R2.frac_voltam, 0.4, color=cores, label="dos ausentes: voltam em jun–dez")
a.set_title("Ausentes em abril–maio e\nretorno no mesmo ano"); a.set_xticks(list(R2.ano)[::2]); a.set_ylim(0, 1.1)
a.legend(loc="upper left", fontsize=6.8)
fig.tight_layout(w_pad=1.2); salva(fig, "f4_relacoes")

# F5 (Figura 7): a semente típica, por mês
B = lemes("B_medianas_mensal.csv"); B["x"] = [xm(p) for p in B.periodo]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
for a, (c, tit) in zip(axs, [("parceiros", "Parceiros por semente (escala log)"), ("herfindahl", "Herfindahl do valor"),
                             ("cinco_maiores", "Participação dos 5 maiores")]):
    choque(a)
    for dr, cor, rot in (("montante", AZUL, "fornecedores"), ("jusante", LAR, "clientes")):
        d = B[B.direcao == dr].sort_values("x")
        serie(a, d.x, d[c], cor, rot)
    a.set_title(tit); eixo_anos(a)
axs[0].set_yscale("log"); axs[0].set_yticks([5, 10, 20, 50, 100, 200]); axs[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
axs[0].yaxis.set_minor_formatter(mticker.NullFormatter()); axs[0].set_ylim(6, 160)
axs[1].legend(loc="upper right")
fig.tight_layout(w_pad=0.8); salva(fig, "f5_sementes")

# F6 (Figura 6): vínculos fortes e fracos, presença mensal relativa a out–dez/2017
P = lemes("C_presenca_mensal.csv"); P = P[P.periodo >= "2017-10"].copy(); P["x"] = [xm(p) for p in P.periodo]
fx = [("menos de 1%", "0.5", "menos de 1% do valor da semente"), ("1% a 10%", LAR, "de 1% a 10%"), ("10% ou mais", AZUL, "10% ou mais")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for a, (dr, rot) in zip(axs, (("montante", "fornecedores"), ("jusante", "clientes"))):
    choque(a)
    for f, cor, r in fx:
        d = P[(P.direcao == dr) & (P.faixa == f)].sort_values("x")
        serie(a, d.x, d.relativa, cor, r)
    a.axhline(1, color="k", lw=0.6); a.set_title(f"Relações com {rot}"); eixo_anos(a, "2017-08-01", "2026-02-01")
axs[0].set_ylabel("presença ÷ presença em out–dez/2017"); axs[0].legend(loc="upper right")
fig.tight_layout(w_pad=1.0); salva(fig, "f6_vinculos")

# F7 (Figura 4): a fronteira e a quebra de meados de 2018
c = M[M.versao == "completa"].sort_values("x").set_index("periodo")
s = M[M.versao == "so_SP"].sort_values("x").set_index("periodo")
# v3 (19/11, célula U1): regra nova de UF do nó (emitente > Receita > destinatário), a mesma da rede só SP
F = lemes("F_fronteira_mensal_v3.csv"); F["x"] = [xm(p) for p in F.periodo]
jul18 = pd.Timestamp("2018-07-01")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax = axs[0]; choque(ax)
for d, cor, rot in ((c, AZUL, "rede principal"), (s, LAR, "só SP")):
    serie(ax, d.x, 100 * d.densidade / d.densidade.loc["2017-01":"2018-06"].mean(), cor, rot)
ax.axvline(jul18, color="k", ls=":", lw=1)
ax.set_title("Densidade\n(índice: jan/2017–jun/2018 = 100)"); eixo_anos(ax); ax.legend(loc="lower left")
ax = axs[1]; choque(ax)
serie(ax, F.x, F.nos_fora_circulo, AZUL, "fora de SP")
ax.axvline(jul18, color="k", ls=":", lw=1)
ax.set_title("Estabelecimentos de fora\nde SP no círculo"); eixo_anos(ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", ".")))
ax = axs[2]; choque(ax)
serie(ax, F.x, 100 * F.frac_fora_n, AZUL, "em número")
serie(ax, F.x, 100 * F.frac_fora_v, LAR, "em valor")
ax.set_title("Parceiros das sementes\nfora de SP (%)"); eixo_anos(ax); ax.set_ylim(top=40); ax.legend(loc="upper left", ncol=2)
fig.tight_layout(w_pad=1.0); salva(fig, "f7_fronteira")
