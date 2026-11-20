# Célula S3 — sementes semente a semente: S2 com pesos exatos na simulação e p calibrado no phi estimado (autônoma)
# Duas análises distintas:
#  5a. painel com degrau e rampa: tendência de cada semente estimada com a SÉRIE TODA; inferência Driscoll-Kraay.
#  5b. desvio da trajetória projetada: nível, tendência de cada semente e meses do ano estimados SÓ com jan/2017–fev/2020;
#      inferência por simulação AR(1) do desvio comum, com os MESMOS pesos da estatística observada (meses ausentes incluídos),
#      reestimando projeção, phi e variância a cada vez; p calibrado por dupla simulação no phi estimado.
#      "Efeito em 2 anos" (5a) é o desvio numa data; "2º ano" (5b) é a média de jul/2021–jun/2022.
# Amostra escolhida só pela atividade antes de 2020. Mês sem operação: n = 0; HHI, cinco maiores e n×HHI ficam ausentes.
import os, time, warnings, numpy as np, pandas as pd, statsmodels.api as sm
t0 = time.time()
BASE = "/home/candre/dados/rede/disciplina/serie"
RES, INT = os.path.join(BASE, "resultados"), os.path.join(BASE, "interno")
PULSOS, DATA, PRE_FIM = [202003, 202004, 202005, 202006], 202007, 202002
MIN_PRE = 24                       # meses ativos exigidos em jan/2017–fev/2020 (38 meses), em cada direção
NSIM, NCAL, NSIM_CAL, SEED = 4999, 300, 499, 20261119
TOL_N, TOL_HHI = 0.5, 0.005        # conferência com B_medianas_mensal: acima disso a célula para
T = 108; TT = np.arange(T)
idx_mes = lambda m: (m // 100 - 2017) * 12 + m % 100 - 1                 # jan/2017 = 0
MESES = [a * 100 + m for a in range(2017, 2026) for m in range(1, 13)]
LEIT_A = {"log1n": "log(1 + nº de parceiros), meses sem operação = 0", "ativa": "semente operou no mês (0/1)"}
LEIT_B = {"log_n": "log(nº de parceiros)", "hhi": "Herfindahl", "top5": "cinco maiores", "n_hhi": "n × HHI"}
BLOCOS = {"choque (mar–jun/20)": (idx_mes(202003), idx_mes(202006)), "1º ano (jul/20–jun/21)": (idx_mes(202007), idx_mes(202106)),
          "2º ano (jul/21–jun/22)": (idx_mes(202107), idx_mes(202206)), "3º ano em diante": (idx_mes(202207), idx_mes(202512))}
PREV = {"previsão de 2019 (ajuste 2017–18)": (idx_mes(201901), idx_mes(201912))}
CONTR = {"degrau": {"degrau": 1}, "rampa (por ano)": {"rampa": 12},
         "efeito em 1 ano (jul/21)": {"degrau": 1, "rampa": 12}, "efeito em 2 anos (jul/22)": {"degrau": 1, "rampa": 24}}

# ---------- funções do bloco 5b: projeção com o período anterior e simulação da série média dos desvios ----------
def projeta(Y, fim):
    """Ajusta nível + tendência de cada semente + meses do ano (comuns) nas colunas 0..fim; devolve a projeção S×T."""
    S = Y.shape[0]
    s, t = np.where(~np.isnan(Y[:, :fim + 1]))
    X = np.zeros((len(s), 2 * S + 11))
    X[np.arange(len(s)), s] = 1.0; X[np.arange(len(s)), S + s] = t
    m = t % 12; X[np.arange(len(s))[m > 0], 2 * S + m[m > 0] - 1] = 1.0
    b = np.linalg.lstsq(X, Y[s, t], rcond=None)[0]
    efm = np.r_[0.0, b[2 * S:]]
    return b[:S, None] + b[S:2 * S, None] * TT[None, :] + efm[TT % 12][None, :]

def estat(Y, Yh, blocos):
    """Por bloco: média entre sementes do desvio médio de cada semente; e quantas sementes ficam acima."""
    D = Y - Yh; out = []
    for a, z in blocos.values():
        with warnings.catch_warnings():                                   # semente sem mês ativo no bloco: NaN, esperado
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            por = np.nanmean(D[:, a:z + 1], axis=1)
        out.append((np.nanmean(por), int(np.sum(por > 0)), int(np.sum(~np.isnan(por)))))
    return out

GRADE = np.linspace(-0.9, 0.97, 188)

def operadores(mask, fim, blocos):
    """Pesos exatos da estatística observada. Se um mesmo desvio e_t fosse somado a todas as sementes nos meses
    observados (mask), a conta de estat() — projeção em painel com os meses observados do ajuste, média de cada semente
    nos seus meses observados do bloco, média entre sementes — é linear em e: estat_b = a_b @ e. O mesmo vale para o
    resíduo médio entre sementes no período de ajuste: r = Rm @ e[:L]. Obtidos passando as T séries unitárias."""
    S = mask.shape[0]; L = fim + 1
    s, t = np.where(mask[:, :L])
    X = np.zeros((len(s), 2 * S + 11))
    X[np.arange(len(s)), s] = 1.0; X[np.arange(len(s)), S + s] = t
    m = t % 12; X[np.arange(len(s))[m > 0], 2 * S + m[m > 0] - 1] = 1.0
    B = np.linalg.pinv(X) @ np.eye(T)[t]                                  # coeficientes para cada série unitária
    efm = np.vstack([np.zeros(T), B[2 * S:]])                             # 12 × T
    proj = B[:S, None, :] + B[S:2 * S, None, :] * TT[None, :, None] + efm[TT % 12][None, :, :]
    dev = np.eye(T)[None, :, :] - proj                                    # semente × mês × série unitária
    A = []
    for a0, z0 in blocos.values():
        w = mask[:, a0:z0 + 1]; n = w.sum(1); ok = n > 0
        por = (dev[:, a0:z0 + 1, :] * w[:, :, None]).sum(1)[ok] / n[ok, None]
        A.append(por.mean(0))
    Rm = np.vstack([dev[mask[:, tt], tt, :].mean(0) if mask[:, tt].any() else np.zeros(T) for tt in range(L)])
    return A, Rm[:, :L]                                                   # o resíduo do ajuste só depende do ajuste

def q_cruz(a, ph):                                                        # a' R(phi) a, AR(1) de inovação 1
    c = np.correlate(a, a, mode="full"); h = np.abs(np.arange(-(len(a) - 1), len(a))); ph = np.atleast_1d(ph)
    return (ph[None, :] ** h[:, None] * c[:, None]).sum(0) / (1 - ph ** 2)

def ar1(ph, s2, n, rng):
    E = np.empty((T, n)); E[0] = rng.normal(0, np.sqrt(s2 / (1 - ph ** 2)), n)
    for t in range(1, T): E[t] = ph * E[t - 1] + rng.normal(0, np.sqrt(s2), n)
    return E

def infere(Y, fim, blocos, nsim=NSIM, ncal=NCAL, nsim_cal=NSIM_CAL, rng=None):
    """Nula: o desvio comum segue um AR(1) depois do choque, com phi corrigido do viés de amostra curta e variância
    estimados no resíduo médio do ajuste. A estatística simulada usa os mesmos pesos da observada (operadores), e cada
    simulação reestima phi e variância (estatística estudentizada). p calibrado: o mesmo teste aplicado a ncal séries
    sem efeito geradas com o phi estimado; p_cal = fração delas com p tão pequeno quanto o observado."""
    rng = rng or np.random.default_rng(SEED)
    Yh = projeta(Y, fim); obs = estat(Y, Yh, blocos)
    mask = ~np.isnan(Y); L = fim + 1
    A, Rm = operadores(mask, fim, blocos)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        e = np.nan_to_num(np.nanmean((Y - Yh)[:, :L], axis=0))           # resíduo médio entre sementes, no ajuste
    vazio = [dict(bloco=nome, desvio=m, ic_inf=np.nan, ic_sup=np.nan, phi=np.nan, p=np.nan, p_cal=np.nan, sobem=sobe,
                  sementes=n) for nome, (m, sobe, n) in zip(blocos, obs)]
    if not np.sum(e * e) > 1e-12:                                          # sem variação no ajuste: não há inferência
        return vazio, Yh
    i = np.arange(L); Dm = np.abs(i[:, None] - i[None, :]); Sh = 0.5 * (Dm == 1)
    G, H = np.empty(len(GRADE)), np.empty(len(GRADE))
    for j, p_ in enumerate(GRADE):
        RSR = Rm @ (p_ ** Dm / (1 - p_ ** 2)) @ Rm.T; H[j] = np.trace(RSR); G[j] = np.trace(Sh @ RSR) / H[j]
    G = np.maximum.accumulate(G)
    def phi_s2(R):
        r1 = np.sum(R[1:] * R[:-1], axis=0) / np.sum(R * R, axis=0)
        p_ = np.interp(r1, G, GRADE); return p_, np.sum(R * R, axis=0) / np.interp(p_, GRADE, H)
    def teste(ph, s2, tobs, n):                                            # p e valor crítico, dados phi e variância
        E = ar1(ph, s2, n, rng); phs, s2s = phi_s2(Rm @ E[:L])
        ts = [np.abs(a @ E) / np.sqrt(s2s * q_cruz(a, phs)) for a in A]
        return np.array([(1 + np.sum(t_ >= to)) / (n + 1) for t_, to in zip(ts, tobs)]), ts
    ph, s2 = phi_s2(e[:, None]); ph, s2 = float(ph[0]), float(s2[0])
    ep = np.array([np.sqrt(s2 * q_cruz(a, ph)[0]) for a in A])
    m_obs = np.array([o[0] for o in obs])
    p, ts = teste(ph, s2, np.abs(m_obs) / ep, nsim)
    # calibração: séries sem efeito com o phi estimado, passando pelo procedimento inteiro
    Er = ar1(ph, s2, ncal, rng); phr, s2r = phi_s2(Rm @ Er[:L])
    pr = np.empty((ncal, len(A)))
    for k in range(ncal):
        tr = [abs(a @ Er[:, k]) / np.sqrt(s2r[k] * q_cruz(a, phr[k])[0]) for a in A]
        pr[k] = teste(phr[k], s2r[k], tr, nsim_cal)[0]
    out = []
    for j, (nome, (m, sobe, n)) in enumerate(zip(blocos, obs)):
        alfa = np.quantile(pr[:, j], 0.05)                                 # nível nominal que dá 5% de fato
        crit = np.quantile(ts[j], 1 - max(alfa, 1.0 / (nsim + 1)))
        out.append(dict(bloco=nome, desvio=m, ic_inf=m - crit * ep[j], ic_sup=m + crit * ep[j], phi=ph, p=p[j],
                        p_cal=(1 + np.sum(pr[:, j] <= p[j])) / (ncal + 1), sobem=sobe, sementes=n))
    return out, Yh

# ---------- funções do bloco 5a: painel com degrau e rampa ----------
def desenho(D):
    t = D.t.values; pos = t - idx_mes(DATA)
    X = pd.get_dummies(D.cod, prefix="fe", dtype=float)
    for c in sorted(D.cod.unique()): X[f"tr_{c}"] = np.where(D.cod == c, t, 0.0)
    for m in range(2, 13): X[f"mes{m}"] = (D.mes.values % 100 == m).astype(float)
    for p in PULSOS: X[f"pulso_{p}"] = (D.mes.values == p).astype(float)
    X["degrau"] = (pos >= 0).astype(float); X["rampa"] = np.where(pos >= 0, pos, 0).astype(float)
    return X.loc[:, (X != 0).any()]

def painel(D, v):
    D = D[D[v].notna()].sort_values(["t", "cod"]); X = desenho(D)
    r = sm.OLS(D[v].values, X.values).fit(cov_type="hac-groupsum", cov_kwds={"time": D.t.values, "maxlags": 12})
    out = []
    for nome, w in CONTR.items():
        tt = r.t_test(np.array([[w.get(c, 0.0) for c in X.columns]])); lo, hi = np.asarray(tt.conf_int()).ravel()
        out.append(dict(termo=nome, coef=float(np.squeeze(tt.effect)), ic_inf=lo, ic_sup=hi, p=float(np.squeeze(tt.pvalue))))
    return out

# ---------- dados ----------
# 1. indicadores por semente × direção × mês (antes de agregar)
c1 = pd.read_parquet(os.path.join(BASE, "camada1_mensal.parquet"), columns=["semente", "parceiro", "mes", "v", "direcao"])
c1 = c1[(c1.mes // 100).between(2017, 2025)].copy()
c1["v"] = pd.to_numeric(c1.v, errors="coerce").astype(float)
g = c1[c1.v > 0].groupby(["semente", "direcao", "mes", "parceiro"], as_index=False).v.sum()
ch = ["semente", "direcao", "mes"]
g["sh"] = g.v / g.groupby(ch).v.transform("sum")
g["r"] = g.groupby(ch).v.rank(method="first", ascending=False)
g["sh2"], g["sh5"] = g.sh ** 2, np.where(g.r <= 5, g.sh, 0.0)
P = g.groupby(ch).agg(n=("v", "size"), hhi=("sh2", "sum"), top5=("sh5", "sum")).reset_index()
cods = {s: f"x{i + 1:02d}" for i, s in enumerate(sorted(P.semente.unique()))}     # códigos arbitrários
os.makedirs(INT, exist_ok=True)
pd.Series(cods, name="cod").rename_axis("semente").to_csv(os.path.join(INT, "S_codigos_sementes.csv"), sep=";")
P["cod"] = P.semente.map(cods)

# conferência efetiva com as medianas antigas (mesmos meses, mesmos valores); para se divergir
arq_b = os.path.join(RES, "B_medianas_mensal.csv")
if os.path.exists(arq_b):
    Bm = pd.read_csv(arq_b, sep=";").set_index(["direcao", "mes"])
    med = P.groupby(["direcao", "mes"])[["n", "hhi"]].median()
    faltam = Bm.index.difference(med.index).union(med.index.difference(Bm.index))
    dn = (med.n - Bm.parceiros).abs().max(); dh = (med.hhi - Bm.herfindahl).abs().max()
    print(f"conferência com B_medianas_mensal: meses só num dos lados {len(faltam)} | dif. máx. nº {dn:.3g} | Herfindahl {dh:.3g}")
    if len(faltam) or dn > TOL_N or dh > TOL_HHI:
        raise RuntimeError("o painel não reproduz B_medianas_mensal: mande esta saída antes de seguir")
else:
    print("AVISO: B_medianas_mensal.csv não encontrado; conferência não feita")

# 2. amostra pela atividade ANTES de 2020 e calendário completo
pre = P[P.mes <= PRE_FIM].groupby(["direcao", "cod"]).size()
sel = pre[pre >= MIN_PRE].index
Q = pd.DataFrame([(d, c, m) for d, c in sel for m in MESES], columns=["direcao", "cod", "mes"])
Q = Q.merge(P[["direcao", "cod", "mes", "n", "hhi", "top5"]], how="left", on=["direcao", "cod", "mes"])
Q["n"] = Q.n.fillna(0); Q["ativa"] = (Q.n > 0).astype(float); Q["log1n"] = np.log1p(Q.n)
Q["log_n"] = np.where(Q.n > 0, np.log(Q.n.where(Q.n > 0)), np.nan)
Q["n_hhi"] = np.where(Q.n > 0, Q.n * Q.hhi, np.nan)            # hhi e top5 já são NaN nos meses sem operação
Q["t"] = idx_mes(Q.mes)
Q.to_csv(os.path.join(RES, "S3_painel_sementes_mensal.csv"), sep=";", index=False)
print("\n== Amostra (≥ 24 meses ativos em jan/2017–fev/2020; calendário completo de 108 meses)")
pos = Q[Q.mes >= DATA]
am = pd.DataFrame({"sementes": Q.groupby("direcao").cod.nunique(),
                   "meses sem operação depois do choque": pos.groupby("direcao").ativa.apply(lambda x: int((x == 0).sum())),
                   "sementes com algum mês parado depois": pos[pos.ativa == 0].groupby("direcao").cod.nunique(),
                   "sementes sem operação em 2025": Q[Q.mes >= 202501].groupby(["direcao", "cod"]).ativa.max()
                                                     .eq(0).groupby("direcao").sum()}).fillna(0).astype(int)
print(am.to_string())
print("Fora da amostra (menos de 24 meses ativos antes):",
      {d: int((pre.reindex(pd.MultiIndex.from_product([[d], sorted(cods.values())]), fill_value=0) < MIN_PRE).sum())
       for d in ("montante", "jusante")})

# 3. bloco 5a: painel com a série toda
lin = []
for dr in ("montante", "jusante"):
    D = Q[Q.direcao == dr]
    for v in list(LEIT_A) + list(LEIT_B):
        for o in painel(D, v):
            lin.append(dict(lado="fornecedores" if dr == "montante" else "clientes", variavel=v, **o))
E = pd.DataFrame(lin); E.to_csv(os.path.join(RES, "S3_painel_degrau_rampa.csv"), sep=";", index=False)
print(f"\npainel estimado ({time.time() - t0:.0f}s)")

# 4. bloco 5b: projeção só com o período anterior; simulação com pesos exatos e p calibrado
lin2, fora01 = [], []
for dr in ("montante", "jusante"):
    lado = "fornecedores" if dr == "montante" else "clientes"
    for v in list(LEIT_A) + list(LEIT_B):
        Y = Q[Q.direcao == dr].pivot(index="cod", columns="t", values=v).reindex(columns=range(T)).values.astype(float)
        res, Yh = infere(Y, idx_mes(PRE_FIM), BLOCOS)
        ok18 = np.sum(~np.isnan(Y[:, :idx_mes(201812) + 1]), axis=1) >= 12      # previsão 2019: ≥ 12 meses em 2017–18
        res += infere(Y[ok18], idx_mes(201812), PREV)[0]
        for o in res: lin2.append(dict(lado=lado, variavel=v, **o))
        if v in ("hhi", "top5"):
            for nome, (a0, z0) in BLOCOS.items():
                p_ = Yh[:, a0:z0 + 1]
                fora01.append(dict(lado=lado, variavel=v, bloco=nome, frac_proj_fora_0_1=float(np.mean((p_ < -1e-9) | (p_ > 1 + 1e-9)))))
        print(f"{lado} {v}: simulação feita ({time.time() - t0:.0f}s)")
J = pd.DataFrame(lin2); J.to_csv(os.path.join(RES, "S3_desvios_projecao.csv"), sep=";", index=False)

# 5. saída
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
fmt = lambda a, b, c, p: f"{a:+.4f} [{b:+.4f}; {c:+.4f}] p={p:.3f}"
print("\n== 5a. Painel com degrau e rampa (tendência de cada semente estimada com a série toda; Driscoll-Kraay)")
E["txt"] = [fmt(*x) for x in zip(E.coef, E.ic_inf, E.ic_sup, E.p)]
print(E.pivot_table(index=["lado", "variavel"], columns="termo", values="txt", aggfunc="first")
      [list(CONTR)].reindex([(l, v) for l in ("fornecedores", "clientes") for v in list(LEIT_A) + list(LEIT_B)]).to_string())
print("\n== 5b. Desvio da projeção feita só com jan/2017–fev/2020 (média entre sementes; IC calibrado, p e p calibrado)")
J["txt"] = [f"{m:+.4f} [{lo:+.4f}; {hi:+.4f}] p={p:.3f} pc={pc:.3f} ({s}/{n})"
            for m, lo, hi, p, pc, s, n in zip(J.desvio, J.ic_inf, J.ic_sup, J.p, J.p_cal, J.sobem, J.sementes)]
ordem = list(PREV) + list(BLOCOS)
print(J.pivot_table(index=["lado", "variavel"], columns="bloco", values="txt", aggfunc="first")[ordem]
      .reindex([(l, v) for l in ("fornecedores", "clientes") for v in list(LEIT_A) + list(LEIT_B)]).to_string())
print("\nphi estimado no ajuste (jan/2017–fev/2020):")
print(J[J.bloco == list(BLOCOS)[0]].set_index(["lado", "variavel"]).phi.round(2).unstack(0).to_string())
print("\nProjeções de HHI e cinco maiores fora de [0, 1], por bloco (fração dos semente-meses):")
print(pd.DataFrame(fora01).pivot_table(index=["lado", "variavel"], columns="bloco", values="frac_proj_fora_0_1")[list(BLOCOS)].round(3).to_string())
print("\nLeitura: (k/n) = sementes acima da projeção. log_n e log1n estão em log: exp(desvio) - 1 é a variação da média "
      "geométrica (log_n: só meses com operação). pc = p calibrado no phi estimado; o IC usa o valor crítico calibrado.")
print(f"\nLeitura A (log1n, ativa): todas as sementes da amostra, meses parados incluídos. Leitura B (log_n, hhi, top5, n_hhi): "
      f"só meses com operação.\nGravados em resultados: S3_painel_sementes_mensal, S3_painel_degrau_rampa, S3_desvios_projecao "
      f"(códigos x01…, sem CNPJ) | {(time.time() - t0) / 60:.1f} min")
