# Mediana, Hodges-Lehmann e média sem as 3 extremas dos desvios por semente (seção 3.4).
# Uso: python medianas_S3.py sementes_painel_S3.py S3_painel_sementes_mensal.csv
import sys, warnings, numpy as np, pandas as pd
exec(open(sys.argv[1], encoding="utf-8").read().split("# ---------- dados")[0])
Q = pd.read_csv(sys.argv[2], sep=";")
warnings.filterwarnings("ignore", message="Mean of empty slice")
lin = []
for dr in ("montante", "jusante"):
    for v in ("log_n", "hhi", "top5", "n_hhi"):
        Y = Q[Q.direcao == dr].pivot(index="cod", columns="t", values=v).reindex(columns=range(T)).values.astype(float)
        Yh = projeta(Y, idx_mes(PRE_FIM)); D = Y - Yh
        for nome, (a, z) in BLOCOS.items():
            por = np.nanmean(D[:, a:z + 1], axis=1); por = por[~np.isnan(por)]
            # Hodges-Lehmann: mediana das médias de Walsh
            w = (por[:, None] + por[None, :]) / 2; hl = np.median(w[np.triu_indices(len(por))])
            q = np.percentile(por, [25, 75])
            # quanto da média vem das 3 sementes mais extremas
            o = np.argsort(-np.abs(por)); sem3 = np.mean(np.delete(por, o[:3]))
            lin.append(dict(lado=dr, var=v, bloco=nome[:8], media=por.mean(), mediana=np.median(por), hl=hl,
                            q25=q[0], q75=q[1], media_sem_3_extremas=sem3, acima=int((por > 0).sum()), n=len(por)))
pd.set_option("display.width", 200)
print(pd.DataFrame(lin).round(4).to_string(index=False))
# n×HHI: decompor no choque  n×HHI = n / (1/HHI)
