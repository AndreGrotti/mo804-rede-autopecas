# Teste formal pedido no parecer de 19/11: o salto de concentração do choque persistiu nos anos seguintes?
# H0: desvio(choque) = desvio(bloco posterior). A diferença entre blocos é linear nos desvios, como cada bloco,
# então usa a mesma inferência da S3 (pesos exatos, AR(1) estudentizado, p calibrado no phi estimado).
# Uso: python sementes_contraste_S3.py <S3_painel_sementes_mensal.csv>   (roda fora do servidor: só códigos x01…)
import os, sys, warnings, numpy as np, pandas as pd
AQUI = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(AQUI, "sementes_painel_S3.py"), encoding="utf-8").read().split("# ---------- funções do bloco 5a")[0])
warnings.filterwarnings("ignore", message="Mean of empty slice")

def contraste(Y, fim, blocos, pares, nsim=NSIM, ncal=NCAL, nsim_cal=NSIM_CAL, rng=None):
    """Como infere(), mas para diferenças entre blocos: d = desvio(bloco i) − desvio(bloco j)."""
    rng = rng or np.random.default_rng(SEED)
    Yh = projeta(Y, fim); obs = estat(Y, Yh, blocos); mask = ~np.isnan(Y); L = fim + 1
    A0, Rm = operadores(mask, fim, blocos)
    A = [A0[i] - A0[j] for i, j in pares]; m_obs = np.array([obs[i][0] - obs[j][0] for i, j in pares])
    e = np.nan_to_num(np.nanmean((Y - Yh)[:, :L], axis=0))
    i_ = np.arange(L); Dm = np.abs(i_[:, None] - i_[None, :]); Sh = 0.5 * (Dm == 1)
    G, H = np.empty(len(GRADE)), np.empty(len(GRADE))
    for k, p_ in enumerate(GRADE):
        RSR = Rm @ (p_ ** Dm / (1 - p_ ** 2)) @ Rm.T; H[k] = np.trace(RSR); G[k] = np.trace(Sh @ RSR) / H[k]
    G = np.maximum.accumulate(G)
    def phi_s2(R):
        r1 = np.sum(R[1:] * R[:-1], axis=0) / np.sum(R * R, axis=0)
        p_ = np.interp(r1, G, GRADE); return p_, np.sum(R * R, axis=0) / np.interp(p_, GRADE, H)
    def teste(ph, s2, tobs, n):
        E = ar1(ph, s2, n, rng); phs, s2s = phi_s2(Rm @ E[:L])
        ts = [np.abs(a @ E) / np.sqrt(s2s * q_cruz(a, phs)) for a in A]
        return np.array([(1 + np.sum(t_ >= to)) / (n + 1) for t_, to in zip(ts, tobs)]), ts
    ph, s2 = phi_s2(e[:, None]); ph, s2 = float(ph[0]), float(s2[0])
    ep = np.array([np.sqrt(s2 * q_cruz(a, ph)[0]) for a in A])
    p, ts = teste(ph, s2, np.abs(m_obs) / ep, nsim)
    Er = ar1(ph, s2, ncal, rng); phr, s2r = phi_s2(Rm @ Er[:L]); pr = np.empty((ncal, len(A)))
    for k in range(ncal):
        tr = [abs(a @ Er[:, k]) / np.sqrt(s2r[k] * q_cruz(a, phr[k])[0]) for a in A]
        pr[k] = teste(phr[k], s2r[k], tr, nsim_cal)[0]
    out = []
    for j, (m, (a, b)) in enumerate(zip(m_obs, pares)):
        alfa = np.quantile(pr[:, j], 0.05); crit = np.quantile(ts[j], 1 - max(alfa, 1.0 / (nsim + 1)))
        out.append(dict(contraste=f"{list(blocos)[a][:6]} − {list(blocos)[b][:6]}", dif=m, ic_inf=m - crit * ep[j],
                        ic_sup=m + crit * ep[j], phi=ph, p=p[j], p_cal=(1 + np.sum(pr[:, j] <= p[j])) / (ncal + 1)))
    return out

Q = pd.read_csv(sys.argv[1], sep=";")
lin = []
for dr in ("montante", "jusante"):
    for v in ("hhi", "top5", "log_n"):
        Y = Q[Q.direcao == dr].pivot(index="cod", columns="t", values=v).reindex(columns=range(T)).values.astype(float)
        for o in contraste(Y, idx_mes(PRE_FIM), BLOCOS, [(0, 1), (0, 2)]):
            lin.append(dict(lado="fornecedores" if dr == "montante" else "clientes", variavel=v, **o))
        print(dr, v, "feito", flush=True)
R = pd.DataFrame(lin)
pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
R.to_csv(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "S3_contraste_choque_blocos.csv"), sep=";", index=False)
