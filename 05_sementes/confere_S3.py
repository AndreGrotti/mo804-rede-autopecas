# Conferência dos pesos exatos da S3 (exemplo do parecer, máscara irregular, calendário completo).
# Uso: python confere_S3.py sementes_painel_S3.py
import sys, numpy as np
exec(open(sys.argv[1], encoding="utf-8").read().split("# ---------- dados")[0])
g = np.random.default_rng(3); fim = idx_mes(PRE_FIM); L = fim + 1
# 1. exemplo do parecer: semente A opera 6 meses do 1º ano, B os 12; desvio comum de 5 pontos só no 1º mês do bloco
bl = {"1º ano": BLOCOS["1º ano (jul/20–jun/21)"]}; a0, z0 = bl["1º ano"]
mask = np.ones((2, T), bool); mask[0, a0 + 6:z0 + 1] = False
A, Rm = operadores(mask, fim, bl)
e = np.zeros(T); e[a0] = 5.0
Y = np.where(mask, e[None, :], np.nan)
print("exemplo do parecer: pesos exatos", round(A[0] @ e, 4), "| estat() direto", round(estat(Y, projeta(Y, fim), bl)[0][0], 4),
      "| esperado 0,625 | peso igual por mês daria", round(5 / 12, 4))
# 2. máscara irregular e desvio comum aleatório: pesos exatos reproduzem estat() e o resíduo médio do ajuste
S = 35; mask = g.random((S, T)) > 0.08; mask[:5, 60:] = False; mask[:, :L] |= g.random((S, L)) > 0.3
A, Rm = operadores(mask, fim, BLOCOS)
dif, difr = 0, 0
for _ in range(20):
    e = np.cumsum(g.normal(size=T)) + g.normal(size=T)
    Y = np.where(mask, e[None, :] + g.normal(0, 0, (S, T)), np.nan)
    Yh = projeta(Y, fim)
    dif = max(dif, max(abs(o[0] - a @ e) for o, a in zip(estat(Y, Yh, BLOCOS), A)))
    difr = max(difr, np.abs(np.nanmean((Y - Yh)[:, :L], 0) - Rm @ e[:L]).max())
print(f"máscara irregular: dif. máx. estat × pesos {dif:.2e} | resíduo do ajuste {difr:.2e}")
# 3. calendário completo: pesos exatos = os da S2 (média dos meses)
mask = np.ones((S, T), bool); A, Rm = operadores(mask, fim, BLOCOS)
XF = np.column_stack([np.ones(T), TT] + [(TT % 12 == m).astype(float) for m in range(1, 12)])
Ap = np.linalg.pinv(XF[:L]); Apad = np.zeros((XF.shape[1], T)); Apad[:, :L] = Ap
d = max(np.abs(a - ((c := np.where((TT >= a0_) & (TT <= z0_), 1.0 / (z0_ - a0_ + 1), 0.0)) - Apad.T @ (XF.T @ c))).max()
        for a, (a0_, z0_) in zip(A, BLOCOS.values()))
print(f"calendário completo: dif. máx. entre pesos exatos e os da S2 {d:.2e}")
