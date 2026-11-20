# Célula M4b — fração em valor fora de SP: mensal (~30%) × trimestral (~14%). Requer BASE e fora_orig do kernel.
import os, pandas as pd
c1 = pd.read_parquet(os.path.join(BASE, "camada1_mensal.parquet"))
c1 = c1[(c1.mes // 100 == 2017) & (c1.v > 0)].copy()
c1["uf"] = c1.uf_parceiro.fillna("<vazio>").replace("", "<vazio>")
tot = c1.groupby("direcao").v.sum()
print("UFs com mais valor em 2017, % do valor de cada direção:")
u = c1.groupby(["direcao", "uf"]).v.sum().div(tot, level=0).mul(100).round(1)
print(u.sort_values(ascending=False).groupby(level=0).head(6).to_string())
regras = {"uf != SP (como na M4)": c1.uf_parceiro.ne("SP"),
          "uf preenchida e != SP": c1.uf.ne("<vazio>") & c1.uf.ne("SP"),
          "parceiro em fora_orig": c1.parceiro.isin(fora_orig)}
for nome, m in regras.items():
    print(f"{nome:26s}", c1[m].groupby("direcao").v.sum().div(tot).round(3).to_dict())
print("parceiros com mais de uma UF nas linhas:", int((c1.groupby("parceiro").uf_parceiro.nunique() > 1).sum()))
