# ===================== CONFIGURAÇÃO (ajuste aqui) =====================
CAMINHO_CSV = "/home/candre/dados/rede/disciplina/serie/resultados/A_metricas.csv"
VERSAO      = "completa"                 # "completa" ou "so_SP"
DATA        = "2020T2"                   # início da mudança de nível e de tendência
TRANSICAO   = ["2020T1", "2020T2"]       # trimestres do choque: cada um ganha um pulso (controle)
HORIZONTE   = 8                          # contrafactual desenhado só até DATA + 8 trimestres
VARRER      = ("2018T2", "2024T2")       # datas candidatas na varredura de quebras
CORTE_FIM   = None                       # ex.: "2025T2" para excluir os últimos trimestres
RODAR_TESTE = True
REPS_TESTE  = 300
NSIM        = 999
GRAFICOS    = True
SALVAR_CSV  = "/home/candre/dados/rede/disciplina/serie/resultados/intervencao_coef.csv"  # None = não grava
SEMENTE     = 2026
# =====================================================================
# Modelo: y = a + b*t + sazonalidade + pulsos(2020T1, 2020T2) + d*DEGRAU + r*RAMPA + erro AR(1)
# Inferência: MQO; erro-padrão com AR(1) corrigido do viés de amostra pequena; p-valor por simulação.
# Varredura: a mesma quebra (degrau + rampa) testada em cada trimestre de VARRER; mostra ONDE a série
#   quebra e se 2020 é a quebra principal. p "sup" = chance de uma série sem quebra nenhuma produzir,
#   em ALGUMA data, uma quebra tão forte quanto a maior observada.

import os, glob, warnings
import numpy as np, pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEMENTE)
GRADE = np.linspace(-0.9, 0.95, 186)
CONTR = {"degrau (nível)":     {"degrau": 1},
         "rampa (inclinação)": {"rampa": 1},
         "efeito em 1 ano":    {"degrau": 1, "rampa": 4},
         "efeito em 2 anos":   {"degrau": 1, "rampa": 8}}

if not os.path.exists(CAMINHO_CSV):
    cands = glob.glob(os.path.expanduser("~/**/A_metricas*.csv"), recursive=True)
    print("Não achei", repr(CAMINHO_CSV), "— candidatos na sua pasta pessoal:")
    for c in cands: print("  ", c)
    raise FileNotFoundError("Copie o caminho certo para CAMINHO_CSV e rode de novo.")

def periodo(r): return pd.Period(str(r).replace("T", "Q"), freq="Q")
def rotulo(p):  return f"{p.year}T{p.quarter}"

def regressores(idx, quebra=DATA, transicao=TRANSICAO):
    X = pd.DataFrame(index=idx)
    X["const"] = 1.0
    X["tendencia"] = np.arange(len(idx), dtype=float)
    for q in (2, 3, 4):
        X[f"tri{q}"] = (idx.quarter == q).astype(float)
    for d in transicao:
        X[f"pulso_{d}"] = np.asarray(idx == periodo(d), dtype=float)
    if quebra is not None:
        pos = np.array([p.ordinal - periodo(quebra).ordinal for p in idx])
        X["degrau"] = (pos >= 0).astype(float)
        X["rampa"]  = np.where(pos >= 0, pos, 0).astype(float)
    return X.loc[:, (X != 0).any()]

_cache = {}
def modelo(idx, quebra):
    """Matrizes que só dependem das datas: guardadas para reaproveitar nas simulações."""
    chave = (rotulo(idx[0]), rotulo(idx[-1]), quebra)
    if chave not in _cache:
        X = regressores(idx, quebra)
        Xa = X.values.astype(float); n, k = Xa.shape
        A = np.linalg.solve(Xa.T @ Xa, Xa.T); M = np.eye(n) - Xa @ A
        i = np.arange(n); D = np.abs(i[:, None] - i[None, :]); S = 0.5 * (D == 1)
        G = np.empty(len(GRADE)); H = np.empty(len(GRADE))
        for j, ph in enumerate(GRADE):
            MRM = M @ (ph ** D / (1 - ph ** 2)) @ M
            H[j] = np.trace(MRM); G[j] = np.trace(S @ MRM) / H[j]
        _cache[chave] = dict(X=X, Xa=Xa, A=A, n=n, k=k, G=np.maximum.accumulate(G), H=H)
    return _cache[chave]

def phi_s2(P, R):
    """AR(1) corrigido: escolhe o phi cuja autocorrelação ESPERADA dos resíduos é a observada."""
    r1 = (R[1:] * R[:-1]).sum(0) / (R * R).sum(0)
    ph = np.interp(r1, P["G"], GRADE)
    return ph, (R * R).sum(0) / np.interp(ph, GRADE, P["H"])

def q_cruz(a1, a2, ph):
    """a1' R(ph) a2 para erro AR(1) com variância de inovação 1."""
    n = len(a1); c = np.correlate(a1, a2, mode="full")
    h = np.abs(np.arange(-(n - 1), n)); ph = np.atleast_1d(ph)
    return (ph[None, :] ** h[:, None] * c[:, None]).sum(0) / (1 - ph ** 2)

def ruido_ar1(n, ph, s2, nsim):
    s = np.sqrt(s2); E = np.empty((n, nsim))
    E[0] = rng.normal(0, s / np.sqrt(1 - ph ** 2), nsim)
    for t in range(1, n): E[t] = ph * E[t - 1] + rng.normal(0, s, nsim)
    return E

def vetor(P, w): return np.array([w.get(c, 0.0) for c in P["X"].columns])

def estima(P, y, contrastes, nsim=NSIM):
    ya = np.asarray(y, float); b = P["A"] @ ya; e = ya - P["Xa"] @ b
    ph, s2 = phi_s2(P, e[:, None]); ph, s2 = ph[0], s2[0]
    E = ruido_ar1(P["n"], ph, s2, nsim); R = E - P["Xa"] @ (P["A"] @ E)
    phs, s2s = phi_s2(P, R)
    out = []
    for nome, w in contrastes.items():
        a = vetor(P, w) @ P["A"]
        est = a @ ya; ep = np.sqrt(s2 * q_cruz(a, a, ph)[0])
        tsim = (a @ E) / np.sqrt(s2s * q_cruz(a, a, phs))
        out.append({"termo": nome, "coef": est, "ep": ep,
                    "p_sim": (1 + np.sum(np.abs(tsim) >= abs(est / ep))) / (nsim + 1)})
    return pd.DataFrame(out), pd.Series(b, index=P["X"].columns), e, ph

def wald_2(a1, a2, Y, ph, s2):
    """Wald conjunto de degrau e rampa = 0, coluna a coluna de Y."""
    b1, b2 = a1 @ Y, a2 @ Y
    v11, v22, v12 = s2 * q_cruz(a1, a1, ph), s2 * q_cruz(a2, a2, ph), s2 * q_cruz(a1, a2, ph)
    return (b1 ** 2 * v22 - 2 * b1 * b2 * v12 + b2 ** 2 * v11) / (v11 * v22 - v12 ** 2)

def varredura(y, nsim=NSIM):
    idx = y.index; ya = np.asarray(y, float)
    datas = [p for p in pd.period_range(periodo(VARRER[0]), periodo(VARRER[1]), freq="Q")
             if idx[0] < p < idx[-1]]
    P0 = modelo(idx, None); e0 = ya - P0["Xa"] @ (P0["A"] @ ya)
    ph0, s20 = phi_s2(P0, e0[:, None])
    E = ruido_ar1(len(ya), ph0[0], s20[0], nsim)      # séries sem quebra nenhuma
    obs, sims = [], []
    for d in datas:
        P = modelo(idx, rotulo(d))
        a1 = vetor(P, {"degrau": 1}) @ P["A"]; a2 = vetor(P, {"rampa": 1}) @ P["A"]
        e = ya - P["Xa"] @ (P["A"] @ ya); ph, s2 = phi_s2(P, e[:, None])
        obs.append(wald_2(a1, a2, ya[:, None], ph, s2)[0])
        R = E - P["Xa"] @ (P["A"] @ E); phs, s2s = phi_s2(P, R)
        sims.append(wald_2(a1, a2, E, phs, s2s))
    obs, sims = np.array(obs), np.array(sims); sup = sims.max(0)
    tab = pd.DataFrame({"data": [rotulo(d) for d in datas], "wald": obs,
                        "p_pontual": [(1 + np.sum(sims[i] >= obs[i])) / (nsim + 1) for i in range(len(datas))]})
    return tab, (1 + np.sum(sup >= obs.max())) / (nsim + 1), float(np.quantile(sup, 0.95))

def p_lb(resid, lag=4):
    lb = acorr_ljungbox(np.asarray(resid), lags=[lag])
    return float(np.asarray(lb["lb_pvalue"])[0]) if hasattr(lb, "columns") else float(lb[1][0])

def arimax(y, X):
    try:
        return SARIMAX(y, exog=X, order=(1, 0, 0), trend="n").fit(disp=False, maxiter=500)
    except Exception as ex:
        print(f"   [aviso] ARIMAX não convergiu: {ex}"); return None

fmt = lambda v: f"{v: .4g}"
def relata(y, nome, nsim=NSIM):
    print(f"\n=== {nome}  (n = {len(y)}; choque {'+'.join(TRANSICAO)}; quebra a partir de {DATA}) ===")
    P = modelo(y.index, DATA)
    q, b, e, ph = estima(P, y, CONTR, nsim)
    print(q.to_string(index=False, float_format=fmt))
    pul = " | ".join(f"{c[6:]}: {b[c]:+.4g}" for c in b.index if c.startswith("pulso_"))
    print(f"   pulsos (controles do choque, não se testam): {pul}")
    print(f"   erro AR(1) phi corrigido = {ph:+.2f} | Ljung-Box(4) p = {p_lb(e):.3f}")
    m = arimax(y, P["X"])
    if m is not None:
        print(f"   conferência ARIMAX: degrau {m.params['degrau']:+.4g} | rampa {m.params['rampa']:+.4g}")
    tab, p_sup, crit = varredura(y, nsim)
    i = int(tab.wald.idxmax())
    w = tab.loc[tab.data == DATA, "wald"]
    pos = int((tab.wald > float(w.iloc[0])).sum()) + 1 if len(w) else None
    print(f"   VARREDURA {tab.data.iloc[0]}–{tab.data.iloc[-1]}: maior quebra em {tab.data[i]} "
          f"(Wald {tab.wald[i]:.1f}; limite 5% = {crit:.1f}); p de haver alguma quebra = {p_sup:.3f}; "
          f"{DATA} é a {pos}ª de {len(tab)}")
    print("   5 datas com quebra mais forte:")
    print(tab.nlargest(5, "wald").to_string(index=False, float_format=fmt))
    X0 = P["X"].copy(); X0[[c for c in X0 if c in ("degrau", "rampa") or c.startswith("pulso_")]] = 0.0
    dist = np.array([p.ordinal - periodo(DATA).ordinal for p in y.index])
    sem = np.where(dist <= HORIZONTE, X0.values @ b.values, np.nan)
    cf = pd.DataFrame({"observado": np.asarray(y, float), "ajustado": P["Xa"] @ b.values,
                       "sem_intervencao": sem}, index=y.index)
    return q, cf, tab, crit, p_sup

def le_metricas(caminho, versao="completa", regra="trimestre"):
    df = pd.read_csv(caminho, sep=";")
    df = df[(df["regra"] == regra) & (df["versao"] == versao)].copy()
    if df.empty: raise ValueError(f"Nenhuma linha com regra={regra!r} e versao={versao!r}.")
    df.index = pd.PeriodIndex([periodo(t) for t in df["trimestre"]], freq="Q")
    df = df.sort_index()
    if CORTE_FIM: df = df[df.index <= periodo(CORTE_FIM)]
    if df.index.duplicated().any(): raise ValueError("Trimestres repetidos depois do filtro.")
    falta = pd.period_range(df.index.min(), df.index.max(), freq="Q").difference(df.index)
    if len(falta): raise ValueError(f"Faltam trimestres: {[rotulo(p) for p in falta]}")
    s = {"log_vertices": np.log(df["vertices"]), "log_arestas": np.log(df["arestas"]),
         "grau_medio": df["grau_medio"], "densidade_x1000": 1000 * df["densidade"],
         "assortatividade": df["assortatividade"], "transitividade": df["transitividade"]}
    if df["transitividade_aleat"].notna().all():
        s["log_transit_sobre_aleat"] = np.log(df["transitividade"] / df["transitividade_aleat"])
    if df["clust_medio_zero_aleat"].notna().all():
        s["log_clust_sobre_aleat"] = np.log(df["clust_medio_zero"] / df["clust_medio_zero_aleat"])
    return pd.DataFrame(s)

def serie_sintetica(seed=0, d=-0.8, r=0.04, p=-1.5, phi=0.4, sd=0.15):
    idx = pd.period_range("2017Q1", "2025Q4", freq="Q")
    g = np.random.default_rng(seed); e = np.zeros(len(idx))
    for t in range(len(idx)):
        e[t] = (phi * e[t - 1] if t else 0.0) + g.normal(0, sd)
    X = regressores(idx, "2020T2")
    y = (10 + 0.02 * X["tendencia"] + 0.3 * X["tri2"] + 0.1 * X["tri3"] - 0.2 * X["tri4"]
         + d * X["degrau"] + r * X["rampa"] + p * X["pulso_2020T2"] + e)
    return pd.Series(y.values, index=idx)

# ----------------------------- TESTE SINTÉTICO -----------------------------
if RODAR_TESTE:
    print("TESTE 1 — efeitos conhecidos: degrau -0.8, rampa +0.04 (1 ano: -0.64; 2 anos: -0.48), "
          "pulso 2020T2 -1.5, AR(1) 0.4. A varredura deve apontar 2020T2.")
    relata(serie_sintetica(seed=1), "sintética")
    print(f"\nTESTE 2 — {REPS_TESTE} séries SEM efeito: % de vezes que acusa efeito a 5% (o certo é ~5%)")
    rej = {"degrau p_sim": 0, "rampa p_sim": 0, "alguma quebra (varredura)": 0}
    cont = {"degrau": {"degrau": 1}, "rampa": {"rampa": 1}}
    for s in range(REPS_TESTE):
        y0 = serie_sintetica(seed=1000 + s, d=0, r=0, p=0)
        q = estima(modelo(y0.index, DATA), y0, cont, nsim=499)[0]
        rej["degrau p_sim"] += q.p_sim.iloc[0] < 0.05; rej["rampa p_sim"] += q.p_sim.iloc[1] < 0.05
        rej["alguma quebra (varredura)"] += varredura(y0, nsim=199)[1] < 0.05
    for kk, v in rej.items(): print(f"   {kk:26s}: {100 * v / REPS_TESTE:5.1f}%")

# ----------------------------- DADOS REAIS -----------------------------
S = le_metricas(CAMINHO_CSV, VERSAO)
print(f"\n\nSÉRIE REAL {rotulo(S.index[0])}–{rotulo(S.index[-1])}, versão {VERSAO}, {len(S)} trimestres")
quadros, cfs, varr, crits = [], [], [], {}
for col in S.columns:
    q, cf, tab, crit, p_sup = relata(S[col], col)
    quadros.append(q.assign(metrica=col)); cfs.append(cf.assign(metrica=col))
    varr.append(tab.assign(metrica=col, limite_5pct=crit, p_alguma_quebra=p_sup)); crits[col] = crit
coef, contraf, varred = pd.concat(quadros), pd.concat(cfs), pd.concat(varr)
print("\nLembrete: várias métricas testadas ao mesmo tempo; um p < 0,05 isolado é esperado por acaso.")

if SALVAR_CSV:
    base = SALVAR_CSV.replace(".csv", "")
    coef.to_csv(SALVAR_CSV, sep=";", index=False)
    contraf.assign(trimestre=[rotulo(p) for p in contraf.index]).to_csv(f"{base}_contrafactual.csv", sep=";", index=False)
    varred.to_csv(f"{base}_varredura.csv", sep=";", index=False)
    print(f"Gravado: {SALVAR_CSV}, {base}_contrafactual.csv e {base}_varredura.csv")

if GRAFICOS:
    import matplotlib.pyplot as plt
    n = len(S.columns)
    fig, axs = plt.subplots(n, 2, figsize=(13, 2.4 * n), gridspec_kw={"width_ratios": [1.4, 1]}, squeeze=False)
    for (ax, ax2), col in zip(axs, S.columns):
        c = contraf[contraf.metrica == col]; x = [p.to_timestamp() for p in c.index]
        ax.plot(x, c["observado"], "o-", ms=3, color="0.3", label="observado")
        ax.plot(x, c["ajustado"], color="tab:blue", label="ajustado")
        ax.plot(x, c["sem_intervencao"], "--", color="tab:orange", label=f"sem intervenção (até +{HORIZONTE} tri)")
        ax.axvspan(periodo(TRANSICAO[0]).to_timestamp(), (periodo(TRANSICAO[-1]) + 1).to_timestamp(),
                   color="tab:red", alpha=0.12)
        ax.set_title(col, fontsize=9); ax.tick_params(labelsize=7)
        t = varred[varred.metrica == col].reset_index(drop=True)
        ax2.bar(range(len(t)), t.wald, color=["tab:red" if d == DATA else "0.6" for d in t.data])
        ax2.axhline(crits[col], ls="--", color="k", lw=0.8)
        ax2.set_xticks(range(0, len(t), 4)); ax2.set_xticklabels(list(t.data.iloc[::4]), fontsize=7)
        ax2.set_title("força da quebra em cada data (tracejado: limite 5%)", fontsize=8)
        ax2.tick_params(labelsize=7)
    axs[0][0].legend(fontsize=7); plt.tight_layout(); plt.show()
