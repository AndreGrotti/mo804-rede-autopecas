# Célula I1 — intervenção na rede só SP: regra antiga × regra nova de UF, mensal e trimestral (autônoma)
# Mesmo modelo das células M2/M3 (mensal) e do Anexo D (trimestral): tendência + sazonalidade + pulsos do choque
# + degrau e rampa; erro AR(1) com phi corrigido; p por simulação; varredura sup-Wald das datas de quebra.
import os, time, warnings, numpy as np, pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
warnings.filterwarnings("ignore")
t0 = time.time()
RES = "/home/candre/dados/rede/disciplina/serie/resultados"
NSIM, SEMENTE = 999, 2026
GRADE = np.linspace(-0.9, 0.95, 186)
CFGS = {
    "mensal": dict(f="M", saz=("month", range(2, 13), "mes"), pulsos=["2020-03", "2020-04", "2020-05", "2020-06"],
                   data="2020-07", varrer=("2018-04", "2024-06"), ano=12, foco=("2019-06", "2020-08"), lag=12),
    "trimestral": dict(f="Q", saz=("quarter", range(2, 5), "tri"), pulsos=["2020T1", "2020T2"],
                       data="2020T2", varrer=("2018T2", "2024T2"), ano=4, foco=("2019T1", "2020T4"), lag=4)}
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)

def periodo(r): return pd.Period(str(r).replace("T", "Q"), freq=C["f"])
def rotulo(p):  return f"{p.year}-{p.month:02d}" if C["f"] == "M" else f"{p.year}T{p.quarter}"

def regressores(idx, quebra):
    X = pd.DataFrame(index=idx); X["const"] = 1.0; X["tendencia"] = np.arange(len(idx), dtype=float)
    attr, ks, nome = C["saz"]
    for k in ks: X[f"{nome}{k}"] = (np.asarray(getattr(idx, attr)) == k).astype(float)
    for d in C["pulsos"]: X[f"pulso_{d}"] = np.asarray(idx == periodo(d), dtype=float)
    if quebra is not None:
        pos = np.array([p.ordinal - periodo(quebra).ordinal for p in idx])
        X["degrau"] = (pos >= 0).astype(float); X["rampa"] = np.where(pos >= 0, pos, 0).astype(float)
    return X.loc[:, (X != 0).any()]

_cache = {}
def modelo(idx, quebra):
    chave = (C["f"], rotulo(idx[0]), rotulo(idx[-1]), quebra)
    if chave not in _cache:
        X = regressores(idx, quebra); Xa = X.values.astype(float); n, k = Xa.shape
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
        a = vetor(P, w) @ P["A"]; est = a @ ya; ep = np.sqrt(s2 * q_cruz(a, a, ph)[0])
        tsim = (a @ E) / np.sqrt(s2s * q_cruz(a, a, phs))
        out.append({"termo": nome, "coef": est, "ep": ep, "p_sim": (1 + np.sum(np.abs(tsim) >= abs(est / ep))) / (nsim + 1)})
    return pd.DataFrame(out).set_index("termo"), pd.Series(b, index=P["X"].columns), e, ph

def wald_2(a1, a2, Y, ph, s2):
    b1, b2 = a1 @ Y, a2 @ Y
    v11, v22, v12 = s2 * q_cruz(a1, a1, ph), s2 * q_cruz(a2, a2, ph), s2 * q_cruz(a1, a2, ph)
    return (b1 ** 2 * v22 - 2 * b1 * b2 * v12 + b2 ** 2 * v11) / (v11 * v22 - v12 ** 2)

def varredura(y, nsim=NSIM):
    idx = y.index; ya = np.asarray(y, float)
    datas = [p for p in pd.period_range(periodo(C["varrer"][0]), periodo(C["varrer"][1]), freq=C["f"]) if idx[0] < p < idx[-1]]
    P0 = modelo(idx, None); e0 = ya - P0["Xa"] @ (P0["A"] @ ya); ph0, s20 = phi_s2(P0, e0[:, None])
    E = ruido_ar1(len(ya), ph0[0], s20[0], nsim)                      # séries sem quebra nenhuma
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
    return float(np.asarray(acorr_ljungbox(np.asarray(resid), lags=[lag])["lb_pvalue"])[0])

def series(df, col_per):
    df = df.copy(); df.index = pd.PeriodIndex([periodo(t) for t in df[col_per]], freq=C["f"])
    df = df.sort_index(); df = df[~df.index.duplicated(keep="last")]
    falta = pd.period_range(df.index.min(), df.index.max(), freq=C["f"]).difference(df.index)
    if len(falta): raise ValueError(f"faltam períodos: {[rotulo(p) for p in falta]}")
    return pd.DataFrame({"log_vertices": np.log(df.vertices), "log_arestas": np.log(df.arestas), "grau_medio": df.grau_medio,
                         "densidade_x1000": 1000 * df.densidade, "assortatividade": df.assortatividade,
                         "transitividade": df.transitividade})

def le(arq):
    d = pd.read_csv(os.path.join(RES, arq), sep=";")
    return d if d.shape[1] > 1 else pd.read_csv(os.path.join(RES, arq))

# ---------- dados: as duas regras, nas duas frequências (só as 6 métricas que existem nas duas) ----------
M = le("M_metricas_v2.csv"); M = M[M.regra.eq("mes")] if "regra" in M else M
A = le("A_metricas.csv"); A = A[A.regra.eq("trimestre")] if "regra" in A else A
An = le("A_metricas_soSP_regra_nova.csv")
if "regra_uf" in An: An = An[An.regra_uf.eq("nova")]
fontes = {"mensal": {"antiga": (M[M.versao.eq("so_SP")], "periodo"), "nova": (M[M.versao.eq("so_SP_nova")], "periodo")},
          "trimestral": {"antiga": (A[A.versao.eq("so_SP")], "trimestre"), "nova": (An, "trimestre")}}

resumo, foco, todos = [], [], {}
for freq, C in CFGS.items():
    CONTR = {"degrau": {"degrau": 1}, "rampa": {"rampa": 1},
             "1 ano": {"degrau": 1, "rampa": C["ano"]}, "2 anos": {"degrau": 1, "rampa": 2 * C["ano"]}}
    for regra, (df, col_per) in fontes[freq].items():
        S = series(df, col_per)
        print(f"{freq} | regra {regra}: {rotulo(S.index[0])}–{rotulo(S.index[-1])}, {len(S)} períodos ({time.time() - t0:.0f}s)")
        coefs, cfs, varrs = [], [], []
        for col in S.columns:
            rng = np.random.default_rng(SEMENTE)              # mesmas simulações para as duas regras
            P = modelo(S.index, C["data"])
            q, b, e, ph = estima(P, S[col], CONTR)
            tab, p_sup, crit = varredura(S[col])
            i = int(tab.wald.idxmax()); w0 = tab.loc[tab.data == C["data"], "wald"]
            resumo.append(dict(freq=freq, regra=regra, metrica=col,
                               degrau=q.loc["degrau", "coef"], p_degrau=q.loc["degrau", "p_sim"],
                               rampa=q.loc["rampa", "coef"], p_rampa=q.loc["rampa", "p_sim"],
                               efeito_1a=q.loc["1 ano", "coef"], p_1a=q.loc["1 ano", "p_sim"],
                               efeito_2a=q.loc["2 anos", "coef"], p_2a=q.loc["2 anos", "p_sim"],
                               quebra_max=tab.data[i], wald_max=tab.wald[i], limite_5pct=crit, p_sup=p_sup,
                               posicao_data_choque=int((tab.wald > float(w0.iloc[0])).sum()) + 1 if len(w0) else None,
                               phi=ph, lb=p_lb(e, C["lag"]),
                               **{c: b[c] for c in b.index if c.startswith("pulso_")}))
            coefs.append(q.reset_index().assign(metrica=col))
            varrs.append(tab.assign(metrica=col, limite_5pct=crit, p_alguma_quebra=p_sup))
            X0 = P["X"].copy(); X0[[c for c in X0 if c in ("degrau", "rampa") or c.startswith("pulso_")]] = 0.0
            dist = np.array([p.ordinal - periodo(C["data"]).ordinal for p in S.index])
            cfs.append(pd.DataFrame({"periodo": [rotulo(p) for p in S.index], "observado": S[col].values,
                                     "ajustado": P["Xa"] @ b.values,
                                     "sem_intervencao": np.where(dist <= 2 * C["ano"], X0.values @ b.values, np.nan),
                                     "metrica": col}))
            if col in ("log_vertices", "transitividade", "grau_medio", "assortatividade"):
                j = tab[(tab.data >= C["foco"][0]) & (tab.data <= C["foco"][1])]
                foco.append(j.assign(freq=freq, regra=regra, metrica=col, limite=crit))
        base = os.path.join(RES, f"intervencao_soSP_{regra}_{freq}")
        pd.concat(coefs).to_csv(base + "_coef.csv", sep=";", index=False)
        pd.concat(cfs).to_csv(base + "_contrafactual.csv", sep=";", index=False)
        pd.concat(varrs).to_csv(base + "_varredura.csv", sep=";", index=False)

R = pd.DataFrame(resumo)
R.to_csv(os.path.join(RES, "intervencao_soSP_regras_resumo.csv"), sep=";", index=False)

# ---------- conferência: a regra antiga mensal tem de reproduzir os coeficientes da M3 ----------
arq_m3 = os.path.join(RES, "intervencao_mensal_resumo.csv")
if os.path.exists(arq_m3):
    m3 = pd.read_csv(arq_m3, sep=";"); m3 = m3[m3.versao.eq("so_SP")].set_index("metrica")
    r = R[(R.freq == "mensal") & (R.regra == "antiga")].set_index("metrica")
    comum = r.index.intersection(m3.index)
    print(f"\nConferência com a M3 (só SP mensal, regra antiga): dif. máx. no degrau {(r.loc[comum, 'degrau'] - m3.loc[comum, 'degrau']).abs().max():.2e} "
          f"| no efeito em 2 anos {(r.loc[comum, 'efeito_2a'] - m3.loc[comum, 'efeito_2a']).abs().max():.2e} | quebra máx. igual em "
          f"{(r.loc[comum, 'quebra_max'] == m3.loc[comum, 'quebra_max']).sum()} de {len(comum)} (os p mudam só pelo sorteio)")

# ---------- quadros: antiga | nova lado a lado ----------
for freq in CFGS:
    T = R[R.freq == freq].set_index(["metrica", "regra"])[
        ["degrau", "p_degrau", "efeito_2a", "p_2a", "quebra_max", "wald_max", "limite_5pct", "p_sup", "posicao_data_choque", "phi", "lb"]]
    T = T.unstack("regra").swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)
    T = T[[(r, c) for c in ["degrau", "p_degrau", "efeito_2a", "p_2a", "quebra_max", "wald_max", "limite_5pct", "p_sup",
                            "posicao_data_choque", "phi", "lb"] for r in ("antiga", "nova")]]
    print(f"\n{'=' * 110}\n{freq.upper()} — choque {CFGS[freq]['pulsos'][0]}–{CFGS[freq]['pulsos'][-1]} (pulsos), degrau e rampa a partir de "
          f"{CFGS[freq]['data']}; p por simulação; p_sup = alguma quebra na varredura {CFGS[freq]['varrer'][0]}–{CFGS[freq]['varrer'][1]}")
    print(T.round(4).to_string())
    pul = R[(R.freq == freq)].set_index(["metrica", "regra"])[[c for c in R if c.startswith("pulso_") and R.loc[R.freq == freq, c].notna().any()]]
    print("\nPulsos do choque (desvio no período, controles):"); print(pul.round(4).to_string())
Fo = pd.concat(foco)
for freq in CFGS:
    f = Fo[Fo.freq == freq]
    print(f"\nFOCO {freq}: força da quebra (Wald) em cada data, {CFGS[freq]['foco'][0]}–{CFGS[freq]['foco'][1]}; limite 5% na última linha")
    tb = f.pivot_table(index="data", columns=["metrica", "regra"], values="wald")
    tb.loc["limite 5%"] = f.groupby(["metrica", "regra"]).limite.first()
    print(tb.round(1).to_string())
print(f"\nGravados em resultados: intervencao_soSP_regras_resumo.csv e intervencao_soSP_{{antiga,nova}}_{{mensal,trimestral}}"
      f"_{{coef,contrafactual,varredura}}.csv | {(time.time() - t0) / 60:.1f} min")
