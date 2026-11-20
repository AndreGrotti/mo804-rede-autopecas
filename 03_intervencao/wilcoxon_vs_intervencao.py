# Anexo F: o que um teste de Wilcoxon diria nos dados mensais e com que frequência ele acusa efeito sem efeito nenhum.
# Uso: python wilcoxon_vs_intervencao.py <pasta_com_M_metricas_v2.csv_e_intervencao_mensal_resumo.csv>
import os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
DL = sys.argv[1]
M = pd.read_csv(os.path.join(DL, "M_metricas_v2.csv"), sep=";")
M = M[M.versao == "completa"].copy()
M["ano"] = M.periodo.str[:4].astype(int); M["mes"] = M.periodo.str[5:7].astype(int)
M["densidade_x1000"] = M.densidade * 1000
R = pd.read_csv(os.path.join(DL, "intervencao_mensal_resumo.csv"), sep=";")
R = R[R.versao == "completa"].set_index("metrica")
met = [("vertices", "log_vertices", "Estabelecimentos"), ("arestas", "log_arestas", "Ligações"), ("grau_medio", "grau_medio", "Grau médio"),
       ("densidade_x1000", "densidade_x1000", "Densidade"), ("assortatividade", "assortatividade", "Assortatividade"),
       ("transitividade", "transitividade", "Transitividade")]
lin = []
for col, chave, nome in met:
    a = M[M.ano == 2019].set_index("mes")[col]; b = M[M.ano == 2021].set_index("mes")[col]
    p_par = wilcoxon(b - a).pvalue
    pre = M[M.ano.between(2017, 2019)][col]; pos = M[M.ano.between(2021, 2025)][col]
    p_rs = mannwhitneyu(pre, pos).pvalue
    lin.append(dict(metrica=nome, sentido_2021_vs_2019=np.sign(np.median(b - a)), p_pareado_2019_2021=p_par,
                    sentido_pos_vs_pre=np.sign(pos.median() - pre.median()), p_soma_postos=p_rs,
                    p_efeito_2_anos=R.loc[chave, "p_2a"], quebra_max=R.loc[chave, "quebra_max"], p_sup=R.loc[chave, "p_sup"]))
T = pd.DataFrame(lin); pd.set_option("display.width", 220); print(T.round(4).to_string(index=False))

# simulação: séries mensais SEM efeito (tendência + meses do ano + erro AR(1)); % em que cada teste acusa efeito a 5%
rng = np.random.default_rng(20261120); t = np.arange(108); mes = t % 12; saz = rng.normal(0, 0.5, 12)
def serie(phi, b):
    e = np.zeros(108); e[0] = rng.normal(0, 1 / np.sqrt(1 - phi ** 2))
    for i in range(1, 108): e[i] = phi * e[i - 1] + rng.normal()
    return b * t + saz[mes] + e
ano = 2017 + t // 12
print("\n% de séries sem efeito em que o teste acusa efeito (p < 0,05), 2000 séries por cenário")
for phi, b, rot in [(0.0, 0.0, "sem tendência, meses independentes"), (0.6, 0.0, "sem tendência, phi = 0,6"),
                    (0.0, 0.03, "tendência suave, meses independentes"), (0.6, 0.03, "tendência suave, phi = 0,6")]:
    rp = rr = 0
    for _ in range(2000):
        y = serie(phi, b)
        rp += wilcoxon(y[ano == 2021] - y[ano == 2019]).pvalue < 0.05
        rr += mannwhitneyu(y[(ano >= 2017) & (ano <= 2019)], y[(ano >= 2021)]).pvalue < 0.05
    print(f"  {rot:40s} pareado 2019×2021 {100 * rp / 2000:5.1f}% | soma de postos 2017–19×2021–25 {100 * rr / 2000:5.1f}%")
