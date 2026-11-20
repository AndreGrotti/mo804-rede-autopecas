# Calibração da S3: painéis sem efeito, choque comum AR(1) + ruído individual AR(1) + meses ausentes
# Uso: python calibra_S3.py sementes_painel_S3.py 250 1999 300 499
import sys, time, numpy as np
exec(open(sys.argv[1], encoding="utf-8").read().split("# ---------- dados")[0])
REPS, NS, NC, NSC = (int(x) for x in sys.argv[2:6])
def gera(g, phi_c, S=35):
    c = np.zeros(T); e = np.zeros((S, T))
    for t in range(T):
        c[t] = (phi_c * c[t - 1] if t else 0) + g.normal(0, 1.0)
        e[:, t] = (0.3 * e[:, t - 1] if t else 0) + g.normal(0, 0.7, S)
    Y = g.normal(0, 2, S)[:, None] + g.normal(0, .02, S)[:, None] * TT + g.normal(0, .3, 12)[TT % 12] + c + e
    mask = g.random((S, T)) > 0.05; mask[:3, idx_mes(202101):] = False          # ausências e 3 sementes que param
    return np.where(mask, Y, np.nan)
for phi_c in (0.0, 0.6, 0.9):
    g = np.random.default_rng(7); rp = np.zeros(len(BLOCOS)); rc = np.zeros(len(BLOCOS)); t0 = time.time()
    for r in range(REPS):
        res, _ = infere(gera(g, phi_c), idx_mes(PRE_FIM), BLOCOS, nsim=NS, ncal=NC, nsim_cal=NSC, rng=np.random.default_rng(500 + r))
        rp += [o["p"] < .05 for o in res]; rc += [o["p_cal"] < .05 for o in res]
    print(f"phi do choque comum = {phi_c}: % com p < 0,05 em {REPS} painéis sem efeito ({time.time() - t0:.0f}s)")
    for j, b in enumerate(BLOCOS):
        print(f"   {b:24s} p {100 * rp[j] / REPS:5.1f}% | p calibrado {100 * rc[j] / REPS:5.1f}%")
