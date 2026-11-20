# Anexo D: correções de Bonferroni e de Holm por família de testes.
# Uso: python multiplas.py <pasta_com_os_CSVs_de_resultados>
import sys, pandas as pd, numpy as np
DL = sys.argv[1]
def corr(nome, d):
    m = len(d); it = sorted(d.items(), key=lambda x: x[1]); bonf = [k for k, p in it if p <= 0.05 / m]; holm = []
    for i, (k, p) in enumerate(it):
        if p <= 0.05 / (m - i): holm.append(k)
        else: break
    print(f"{nome} (m={m}, limite {0.05/m:.4f})\n   Bonferroni: {bonf}\n   Holm: {holm}")
R = pd.read_csv(f"{DL}/intervencao_mensal_resumo.csv", sep=";")
c = R[R.versao == "completa"].set_index("metrica")
corr("principal degrau", c.p_degrau.to_dict()); corr("principal 2 anos", c.p_2a.to_dict()); corr("principal varredura", c.p_sup.to_dict())
S = pd.read_csv(f"{DL}/intervencao_soSP_regras_resumo.csv", sep=";")
s = S[(S.freq == "mensal") & (S.regra == "nova")].set_index("metrica")
corr("so SP degrau", s.p_degrau.to_dict()); corr("so SP 2 anos", s.p_2a.to_dict()); corr("so SP varredura", s.p_sup.to_dict())
J = pd.read_csv(f"{DL}/S3_desvios_projecao.csv", sep=";")
J = J[J.variavel.isin(["log_n", "hhi", "top5"])]
ch = J[J.bloco.str.startswith("choque")]
corr("sementes choque (p_cal)", {f"{a}-{b}": p for a, b, p in zip(ch.lado, ch.variavel, ch.p_cal)})
corr("sementes choque (p)", {f"{a}-{b}": p for a, b, p in zip(ch.lado, ch.variavel, ch.p)})
pos = J[~J.bloco.str.startswith(("choque", "previs"))]
corr("sementes pós-choque (p_cal)", {f"{a}-{b}-{c}": p for a, b, c, p in zip(pos.lado, pos.variavel, pos.bloco, pos.p_cal)})
C = pd.read_csv(f"{DL}/S3_contraste_choque_blocos.csv", sep=";"); C = C[C.variavel != "log_n"]
corr("contrastes (p_cal)", {f"{a}-{b}-{c}": p for a, b, c, p in zip(C.lado, C.variavel, C.contraste, C.p_cal)})
