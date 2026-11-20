# Célula M2 — modelo de intervenção MENSAL: definições + calibração em séries sintéticas (não usa dados reais)
import time, warnings
import numpy as np, pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
warnings.filterwarnings("ignore")

CAMINHO_M  = "/home/candre/dados/rede/disciplina/serie/resultados/M_metricas.csv"
PULSOS     = ["2020-03", "2020-04", "2020-05", "2020-06"]   # meses do choque: um pulso cada (controles)
DATA       = "2020-07"                                         # início do degrau e da rampa
HORIZONTE  = 24                                                # contrafactual até DATA + 24 meses
VARRER     = ("2018-04", "2024-06")                            # datas candidatas na varredura
NSIM       = 999
REPS_TESTE = 300
SEMENTE    = 2026
rng   = np.random.default_rng(SEMENTE)
GRADE = np.linspace(-0.9, 0.95, 186)
CONTR = {"degrau (nível)":   {"degrau": 1},
         "rampa (por mês)":  {"rampa": 1},
         "efeito em 1 ano":  {"degrau": 1, "rampa": 12},
         "efeito em 2 anos": {"degrau": 1, "rampa": 24}}

def periodo(r): return pd.Period(str(r), freq="M")
def rotulo(p):  return f"{p.year}-{p.month:02d}"

def regressores(idx, quebra=DATA, pulsos=PULSOS):
    X = pd.DataFrame(index=idx)
    X["const"] = 1.0
    X["tendencia"] = np.arange(len(idx), dtype=float)
    for m in range(2, 13):
        X[f"mes{m}"] = (idx.month == m).astype(float)
    for d in pulsos:
        X[f"pulso_{d}"] = np.asarray(idx == periodo(d), dtype=float)
    if quebra is not None:
        pos = np.array([p.ordinal - periodo(quebra).ordinal for p in idx])
        X["degrau"] = (pos >= 0).astype(float)
        X["rampa"]  = np.where(pos >= 0, pos, 0).astype(float)
    return X.loc[:, (X != 0).any()]

_cache = {}
def modelo(idx, quebra):
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
    r1 = (R[1:] * R[:-1]).sum(0) / (R * R).sum(0)
    ph = np.interp(r1, P["G"], GRADE)
    return ph, (R * R).sum(0) / np.interp(ph, GRADE, P["H"])

def q_cruz(a1, a2, ph):
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
    b1, b2 = a1 @ Y, a2 @ Y
    v11, v22, v12 = s2 * q_cruz(a1, a1, ph), s2 * q_cruz(a2, a2, ph), s2 * q_cruz(a1, a2, ph)
    return (b1 ** 2 * v22 - 2 * b1 * b2 * v12 + b2 ** 2 * v11) / (v11 * v22 - v12 ** 2)

def varredura(y, nsim=NSIM):
    idx = y.index; ya = np.asarray(y, float)
    datas = [p for p in pd.period_range(periodo(VARRER[0]), periodo(VARRER[1]), freq="M") if idx[0] < p < idx[-1]]
    P0 = modelo(idx, None); e0 = ya - P0["Xa"] @ (P0["A"] @ ya)
    ph0, s20 = phi_s2(P0, e0[:, None])
    E = ruido_ar1(len(ya), ph0[0], s20[0], nsim)
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

def p_lb(resid, lag):
    lb = acorr_ljungbox(np.asarray(resid), lags=[lag])
    return float(np.asarray(lb["lb_pvalue"])[0])

fmt = lambda v: f"{v: .4g}"
def relata(y, nome, nsim=NSIM):
    print(f"\n=== {nome}  (n = {len(y)}; pulsos {PULSOS[0]}–{PULSOS[-1]}; quebra a partir de {DATA}) ===")
    P = modelo(y.index, DATA)
    q, b, e, ph = estima(P, y, CONTR, nsim)
    print(q.to_string(index=False, float_format=fmt))
    print("   pulsos: " + " | ".join(f"{c[6:]}: {b[c]:+.4g}" for c in b.index if c.startswith("pulso_")))
    print(f"   erro AR(1) phi corrigido = {ph:+.2f} | Ljung-Box p: lag 6 = {p_lb(e, 6):.3f}, lag 12 = {p_lb(e, 12):.3f}")
    tab, p_sup, crit = varredura(y, nsim)
    i = int(tab.wald.idxmax())
    print(f"   VARREDURA {tab.data.iloc[0]}–{tab.data.iloc[-1]}: maior quebra em {tab.data[i]} "
          f"(Wald {tab.wald[i]:.1f}; limite 5% = {crit:.1f}); p de haver alguma quebra = {p_sup:.3f}")
    print("   5 datas com quebra mais forte:")
    print(tab.nlargest(5, "wald").to_string(index=False, float_format=fmt))
    X0 = P["X"].copy(); X0[[c for c in X0 if c in ("degrau", "rampa") or c.startswith("pulso_")]] = 0.0
    dist = np.array([p.ordinal - periodo(DATA).ordinal for p in y.index])
    sem = np.where(dist <= HORIZONTE, X0.values @ b.values, np.nan)
    cf = pd.DataFrame({"observado": np.asarray(y, float), "ajustado": P["Xa"] @ b.values, "sem_intervencao": sem}, index=y.index)
    return q, cf, tab, crit, p_sup

def le_mensal(caminho, versao):
    df = pd.read_csv(caminho, sep=";")
    df = df[(df["regra"] == "mes") & (df["versao"] == versao)].copy()
    df.index = pd.PeriodIndex([periodo(t) for t in df["periodo"]], freq="M")
    df = df.sort_index()
    falta = pd.period_range(df.index.min(), df.index.max(), freq="M").difference(df.index)
    if len(falta): raise ValueError(f"Faltam meses: {[rotulo(p) for p in falta]}")
    s = {"log_vertices": np.log(df["vertices"]), "log_arestas": np.log(df["arestas"]),
         "grau_medio": df["grau_medio"], "densidade_x1000": 1000 * df["densidade"],
         "assortatividade": df["assortatividade"], "transitividade": df["transitividade"]}
    if "transitividade_aleat" in df and df["transitividade_aleat"].notna().all():
        s["log_transit_sobre_aleat"] = np.log(df["transitividade"] / df["transitividade_aleat"])
    if "clust_medio_zero_aleat" in df and df["clust_medio_zero_aleat"].notna().all():
        s["log_clust_sobre_aleat"] = np.log(df["clust_medio_zero"] / df["clust_medio_zero_aleat"])
    return pd.DataFrame(s)

def serie_sintetica(seed, d=-0.8, r=0.02, p=-1.5, phi=0.5, phi12=0.0, sd=0.15):
    idx = pd.period_range("2017-01", "2025-12", freq="M")
    g = np.random.default_rng(seed); n = len(idx); e = np.zeros(n)
    for t in range(n):
        e[t] = (phi * e[t - 1] if t else 0.0) + (phi12 * e[t - 12] if t >= 12 else 0.0) + g.normal(0, sd)
    X = regressores(idx, DATA)
    saz = g.normal(0, 0.2, 12)[np.asarray(idx.month) - 1]
    y = 10 + 0.005 * X["tendencia"].values + saz + d * X["degrau"].values + r * X["rampa"].values \
        + p * X["pulso_2020-04"].values + e
    return pd.Series(y, index=idx)

# ----------------------------- CALIBRAÇÃO -----------------------------
t0 = time.time()
print("TESTE 1 — efeitos conhecidos: degrau -0,8; rampa +0,02/mês (1 ano: -0,56; 2 anos: -0,32); pulso abr/2020 -1,5; "
      "AR(1) 0,5. A varredura deve apontar 2020-07 (ou um mês do choque, que é o mesmo modelo).")
relata(serie_sintetica(seed=1), "sintética")

cont = {"degrau": {"degrau": 1}, "rampa": {"rampa": 1}, "2 anos": {"degrau": 1, "rampa": 24}}
for titulo, phi12 in (("TESTE 2 — erro AR(1) 0,5, sem efeito", 0.0),
                      ("TESTE 3 — erro AR(1) 0,5 + sazonal de 12 meses 0,3 (o modelo não prevê), sem efeito", 0.3)):
    rej = {"degrau": 0, "rampa": 0, "2 anos": 0, "alguma quebra (varredura)": 0}
    for s in range(REPS_TESTE):
        y0 = serie_sintetica(seed=1000 + s, d=0, r=0, p=0, phi12=phi12)
        q = estima(modelo(y0.index, DATA), y0, cont, nsim=499)[0]
        for kk, pv in zip(cont, q.p_sim): rej[kk] += pv < 0.05
        rej["alguma quebra (varredura)"] += varredura(y0, nsim=199)[1] < 0.05
    print(f"\n{titulo}: % de {REPS_TESTE} séries que acusam efeito a 5% (o certo é ~5%)")
    for kk, v in rej.items(): print(f"   {kk:26s}: {100 * v / REPS_TESTE:5.1f}%")
print(f"\ncalibração em {(time.time() - t0) / 60:.1f} min")
