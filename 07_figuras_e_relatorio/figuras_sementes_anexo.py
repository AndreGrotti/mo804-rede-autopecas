# Figuras do anexo do relatório final: cada semente contra a própria trajetória (célula S3).
# Uso: python figuras_sementes_anexo.py <pasta_downloads> <pasta_saida>
# Lê: S3_painel_sementes_mensal.csv (códigos x01…, sem CNPJ). Usa salva()/estilo de figuras_mensais.py e projeta() da S3.
import os, sys, warnings, numpy as np, pandas as pd
AQUI = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(AQUI, "figuras_mensais.py"), encoding="utf-8").read()
exec(_src.split("# ------------------------------------------------------------------ F1")[0])
_s3 = open(os.path.join(AQUI, "..", "05_sementes", "sementes_painel_S3.py"), encoding="utf-8").read()
exec(_s3.split("# ---------- dados")[0].split("# ---------- funções do bloco 5b")[0])            # constantes
exec("def projeta" + _s3.split("def projeta")[1].split("def estat")[0])                           # projeta()
warnings.filterwarnings("ignore", message="Mean of empty slice")
_salva = salva
def salva(fig, nome):
    # como em figuras_mensais, e ainda: sem ids de grupos e sem recortes (as linhas ficam dentro dos eixos)
    _salva(fig, nome); arq = os.path.join(OUT, nome + ".svg"); s = open(arq, encoding="utf-8").read()
    s = re.sub(r'<g id="[^"]*"', "<g", s)
    s = re.sub(r'\s*clip-path="url\(#[^)]*\)"', "", s)
    s = re.sub(r"<clipPath.*?</clipPath>\s*", "", s, flags=re.S)
    s = re.sub(r"<g>\s*</g>\s*", "", s)
    # estilos repetidos viram classes com prefixo próprio da figura (o SVG fica inline no HTML, junto com outros)
    pre = re.sub(r"\W", "", nome)[:3].lower()
    cont = pd.Series(re.findall(r' style="([^"]+)"', s)).value_counts()
    cls = {st: f"{pre}{i}" for i, st in enumerate(cont[cont >= 3].index)}
    s = re.sub(r' style="([^"]+)"', lambda m: f' class="{cls[m.group(1)]}"' if m.group(1) in cls else m.group(0), s)
    css = "".join(f".{c}{{{st}}}" for st, c in cls.items())
    s = re.sub(r"(<svg[^>]*>)", lambda m: m.group(1) + f"<style>{css}</style>", s, count=1)
    open(arq, "w", encoding="utf-8").write(s); print("   enxuto:", round(len(s) / 1024), "KB")

Q = pd.read_csv(os.path.join(DL, "S3_painel_sementes_mensal.csv"), sep=";")
cods = sorted(Q.cod.unique())
X = [xm(m) for m in MESES]
LADO = {"montante": ("fornecedores", AZUL), "jusante": ("clientes", LAR)}

def matriz(dr, v):
    return Q[Q.direcao == dr].pivot(index="cod", columns="t", values=v).reindex(index=cods, columns=range(T)).values.astype(float)

# ---------------- A1: número de parceiros de cada semente, com a trajetória projetada
# média por trimestre (meses com operação) para caber no limite do HTML; a projeção é a mesma da S3, também por trimestre
def trim(M):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return np.nanmean(M.reshape(M.shape[0], T // 3, 3), axis=2)
XT = [xm(MESES[3 * q + 1]) for q in range(T // 3)]
# trajetória desenhada sem os efeitos dos meses do ano: reta (em log) ajustada à projeção da S3, dos extremos do período
def reta(P):
    return np.polyfit(TT, P.T, 1).T                              # S × 2 (inclinação, nível), em log
proj = {dr: reta(projeta(matriz(dr, "log_n"), idx_mes(PRE_FIM))) for dr in LADO}
def corta(b, a, lo, hi):
    """Trecho da reta log y = a + b t dentro de [lo, hi] (em log) e de [0, T-1]; devolve datas e valores."""
    t0, t1 = 0.0, T - 1.0
    if abs(b) > 1e-12:
        ta, tb = sorted([(lo - a) / b, (hi - a) / b]); t0, t1 = max(t0, ta), min(t1, tb)
    elif not lo <= a <= hi: return [], []
    if t0 >= t1: return [], []
    d = lambda t: X[0] + (X[-1] - X[0]) * (t / (T - 1))
    return [d(t0), d(t1)], [np.exp(a + b * t0), np.exp(a + b * t1)]
obs = {dr: trim(np.where(matriz(dr, "n") > 0, matriz(dr, "n"), np.nan)) for dr in LADO}
fig, axs = plt.subplots(7, 5, figsize=(7.2, 9.4), sharex=True)
for k, (ax, c) in enumerate(zip(axs.flat, cods)):
    choque(ax)
    v = np.concatenate([obs[dr][k] for dr in LADO]); v = v[~np.isnan(v)]
    lo, hi = v.min() / 1.8, v.max() * 1.8                       # eixo pela faixa observada; a reta é cortada na borda
    for dr, (rot, cor) in LADO.items():
        ax.plot(XT, obs[dr][k], "-", color=cor, lw=0.8, label=rot, zorder=3)
        xr, yr = corta(*proj[dr][k], np.log(lo), np.log(hi))
        ax.plot(xr, yr, "--", color=cor, lw=0.8, alpha=0.9, zorder=2, label=f"trajetória ({rot})")
    ax.set_yscale("log"); ax.set_ylim(lo, hi); ax.set_title(c, fontsize=7.5, pad=2)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(milhar)
    ax.tick_params(labelsize=6, length=2, pad=1)
    eixo_anos(ax, passo=3)
    for t in ax.get_xticklabels(): t.set_fontsize(6)
h, l = axs.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(h_pad=0.6, w_pad=0.5, rect=(0, 0.025, 1, 1))
salva(fig, "FA1_sementes_parceiros")

# ---------------- A2: desvio de cada semente em relação à própria trajetória, por bloco
# número de parceiros fica em log (é a escala em que a média da tabela é feita), com rótulos em %; pontos além do
# limite do eixo ficam presos na borda (as linhas de mediana e média usam os valores sem corte)
MED = [("log_n", "número de parceiros", lambda d: d, np.log(0.1), np.log(10)),
       ("top5", "cinco maiores (pontos)", lambda d: 100 * d, -60, 60),
       ("hhi", "Herfindahl", lambda d: d, -0.6, 0.6)]
nb = list(BLOCOS)
rotb = ["choque", "1º ano", "2º ano", "jul/22\nem diante"]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
for i, (dr, (rot, cor)) in enumerate(LADO.items()):
    for j, (v, tit, f, ylo, yhi) in enumerate(MED):
        ax = axs[i, j]; Y = matriz(dr, v); D = Y - projeta(Y, idx_mes(PRE_FIM))
        xs, ys = [], []
        for b, nome in enumerate(nb):
            a0, z0 = BLOCOS[nome]
            por = f(np.nanmean(D[:, a0:z0 + 1], axis=1)); por = por[~np.isnan(por)]
            jit = (np.argsort(np.argsort(por)) / max(len(por) - 1, 1) - 0.5) * 0.36
            m = 0.02 * (yhi - ylo)
            xs += list(b + jit); ys += list(np.clip(por, ylo + m, yhi - m))
            ax.plot([b - 0.3, b + 0.3], [np.median(por)] * 2, "-", color="0.15", lw=1.4, zorder=4)
            ax.plot([b - 0.3, b + 0.3], [np.mean(por)] * 2, ":", color="0.15", lw=1.2, zorder=4)
        # cada semente é um segmento mínimo de ponta redonda: um único caminho por painel, em vez de um marcador por ponto
        px = np.ravel(np.column_stack([xs, np.array(xs) + 1e-3, np.full(len(xs), np.nan)]))
        py = np.ravel(np.column_stack([ys, ys, np.full(len(ys), np.nan)]))
        ax.plot(px, py, "-", lw=2.6, solid_capstyle="round", color=cor, alpha=0.7, zorder=3)
        ax.axhline(0, color="0.5", lw=0.6, zorder=1)
        ax.set_xticks(range(len(nb))); ax.set_xticklabels(rotb, fontsize=6.5)
        ax.set_xlim(-0.6, len(nb) - 0.4); ax.set_ylim(ylo, yhi)
        if v == "log_n":
            pc = np.array([-90, -75, -50, 0, 100, 300, 900])
            ax.set_yticks(np.log1p(pc / 100)); ax.set_yticklabels([f"{p:+d}%".replace("+0%", "0") for p in pc])
        ax.set_title(f"{rot.capitalize()}: {tit}", fontsize=8)
        ax.tick_params(labelsize=6.5, length=2)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color="0.15", lw=1.4), Line2D([], [], color="0.15", lw=1.2, ls=":"),
            Line2D([], [], marker="o", ls="none", ms=3, color="0.4")],
           ["mediana das sementes", "média das sementes", "uma semente"], loc="lower center", ncol=3, fontsize=7,
           frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 1))
salva(fig, "FA2_sementes_desvios")
