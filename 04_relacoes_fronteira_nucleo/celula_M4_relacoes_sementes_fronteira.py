# Célula M4 — versões mensais: relações (3.3), vínculos fortes e fracos (Fig. 6), sementes (3.4) e fronteira (Fig. 4)
# Requer do kernel (células M1 e M1e): BASE, ANOS, MESES, nos_mes, painel, sementes, fora_orig
import os, time, numpy as np, pandas as pd
for v in ("BASE", "ANOS", "MESES", "nos_mes", "painel", "sementes", "fora_orig"):
    if v not in globals():
        raise NameError(f"'{v}' não está no kernel: rode as células M1 e M1e antes desta")
RES = os.path.join(BASE, "resultados")
t0 = time.time()
rot = lambda a, m: f"{a}-{m:02d}"
pd.set_option("display.width", 220)

# ---------- 1. pares do painel (3.522) em cada mês, com valor ----------
LP = pd.Index(sorted(painel)); NP = len(LP)
pares = {}
for a in ANOS:
    e = pd.read_parquet(os.path.join(BASE, f"arestas_mensais_{a}.parquet"), columns=["origem", "destino", "mes", "v"])
    e["m"] = (e.mes % 100 if e.mes.max() > 12 else e.mes).astype(int)
    i, j = LP.get_indexer(e.origem), LP.get_indexer(e.destino)
    ok = (i >= 0) & (j >= 0) & (i != j)
    d = pd.DataFrame({"m": e.m.values[ok], "par": np.minimum(i[ok], j[ok]).astype(np.int64) * NP + np.maximum(i[ok], j[ok]),
                      "v": e.v.values[ok]})
    for m, g in d.groupby("m"):
        pares[(a, m)] = g.groupby("par").v.sum()
    print(f"{a}: pares do painel lidos ({time.time() - t0:.0f}s)")

# retenção: pares do mês que voltam a negociar no mesmo mês do ano seguinte
lin = []
for k, (a, m) in enumerate(MESES[:-12]):
    p0, p1 = pares.get((a, m)), pares.get(MESES[k + 12])
    if p0 is None or p1 is None: continue
    r = p0.index.isin(p1.index)
    lin.append(dict(periodo=rot(a, m), pares=len(p0), retencao=r.mean(), retencao_valor=p0[r].sum() / p0.sum()))
R1 = pd.DataFrame(lin)
R1.to_csv(os.path.join(RES, "R_retencao_mensal.csv"), sep=";", index=False)

# choque: base jan–fev, ausentes em abr–mai, voltam em jun–dez (cada ano)
uniao = lambda a, ms: set().union(*[set(pares[(a, m)].index) for m in ms if (a, m) in pares])
lin = []
for a in ANOS:
    base, meio, fim = uniao(a, (1, 2)), uniao(a, (4, 5)), uniao(a, range(6, 13))
    aus = base - meio
    lin.append(dict(ano=a, pares_base=len(base), frac_ausentes=len(aus) / len(base),
                    frac_voltam=len(aus & fim) / len(aus) if aus else np.nan))
R2 = pd.DataFrame(lin)
R2.to_csv(os.path.join(RES, "R_choque_mensal.csv"), sep=";", index=False)

# ---------- 2. sementes por mês (camada 1) ----------
c1 = pd.read_parquet(os.path.join(BASE, "camada1_mensal.parquet"))
c1 = c1[(c1.mes // 100).between(ANOS[0], ANOS[-1]) & (c1.v > 0)].copy()
print("direções na camada 1:", list(c1.direcao.unique()))
c1["vf"] = np.where(c1.uf_parceiro.ne("SP"), c1.v, 0.0)
p = c1.groupby(["semente", "direcao", "mes", "parceiro"], as_index=False)[["v", "vf"]].sum()
chave = ["semente", "direcao", "mes"]
p["s"] = p.v / p.groupby(chave).v.transform("sum")
p["s2"] = p.s ** 2
p["rk"] = p.groupby(chave).s.rank(ascending=False, method="first")
S = p.groupby(chave).agg(parceiros=("parceiro", "size"), herfindahl=("s2", "sum"), valor=("v", "sum"), valor_fora=("vf", "sum"))
S["cinco_maiores"] = p[p.rk <= 5].groupby(chave).s.sum()
S["frac_fora_v"] = S.valor_fora / S.valor
S = S.reset_index()
B = S.groupby(["direcao", "mes"]).agg(sementes_ativas=("semente", "size"), parceiros=("parceiros", "median"),
                                      herfindahl=("herfindahl", "median"), cinco_maiores=("cinco_maiores", "median"),
                                      frac_fora_v=("frac_fora_v", "median")).reset_index()
B["periodo"] = B.mes.map(lambda x: rot(x // 100, x % 100))
B.to_csv(os.path.join(RES, "B_medianas_mensal.csv"), sep=";", index=False)

# ---------- 3. vínculos fortes e fracos: força em jan–set/2017, referência out–dez/2017 ----------
r = c1[c1.mes.between(201701, 201709)].groupby(["semente", "direcao", "parceiro"]).v.sum()
forca = r / r.groupby(level=[0, 1]).transform("sum")
rel = pd.cut(forca, [0, 0.01, 0.10, 1.0001], right=False, labels=["menos de 1%", "1% a 10%", "10% ou mais"]).astype(str)
rel = rel.rename("faixa").reset_index()
ult = c1.groupby(["semente", "direcao"]).mes.max().rename("ult").reset_index()
rel = rel.merge(ult, on=["semente", "direcao"])
pres = c1[["semente", "direcao", "parceiro", "mes"]].drop_duplicates().merge(rel[["semente", "direcao", "parceiro", "faixa"]],
                                                                              on=["semente", "direcao", "parceiro"])
cnt = pres.groupby(["direcao", "faixa", "mes"]).size()
lin = []
for (d, f), gr in rel.groupby(["direcao", "faixa"]):
    for (a, m) in MESES:
        mm = a * 100 + m
        den = int((gr.ult >= mm).sum())
        lin.append(dict(direcao=d, faixa=f, periodo=rot(a, m), relacoes=den, presenca=cnt.get((d, f, mm), 0) / den if den else np.nan))
P = pd.DataFrame(lin)
ref = P[P.periodo.isin(["2017-10", "2017-11", "2017-12"])].groupby(["direcao", "faixa"]).presenca.mean().rename("ref").reset_index()
P = P.merge(ref, on=["direcao", "faixa"]); P["relativa"] = P.presenca / P.ref
P.to_csv(os.path.join(RES, "C_presenca_mensal.csv"), sep=";", index=False)

# ---------- 4. fronteira por mês ----------
q = c1.groupby(["mes", "parceiro"], as_index=False)[["v", "vf"]].sum()
q["fora_n"] = q.parceiro.isin(fora_orig)
F = q.groupby("mes").agg(parceiros=("parceiro", "size"), frac_fora_n=("fora_n", "mean"), v=("v", "sum"), vf=("vf", "sum")).reset_index()
F["frac_fora_v"] = F.vf / F.v
F["periodo"] = F.mes.map(lambda x: rot(x // 100, x % 100))
F["nos_fora_circulo"] = F.mes.map(lambda x: len(nos_mes.get((x // 100, x % 100), set()) & fora_orig))
F[["periodo", "parceiros", "frac_fora_n", "frac_fora_v", "nos_fora_circulo"]].to_csv(os.path.join(RES, "F_fronteira_mensal.csv"), sep=";", index=False)

# ---------- 5. resumo ----------
print(f"\nGravados em {RES}: R_retencao_mensal, R_choque_mensal, B_medianas_mensal, C_presenca_mensal, F_fronteira_mensal "
      f"({(time.time() - t0) / 60:.1f} min)")
print("\n== Retenção um ano depois (pares do painel), meses em torno do choque")
print(R1[R1.periodo.between("2019-01", "2021-06")].round(3).to_string(index=False))
print(f"média de toda a série: {R1.retencao.mean():.3f} (número) | {R1.retencao_valor.mean():.3f} (valor)")
print("\n== Choque: base jan–fev, ausentes em abr–mai, voltam em jun–dez")
print(R2.round(3).to_string(index=False))
print("\n== Sementes (mediana), meses em torno do choque")
print(B[B.periodo.between("2019-12", "2020-09")].pivot_table(index="periodo", columns="direcao",
      values=["parceiros", "herfindahl", "cinco_maiores"]).round(3).to_string())
print("\n== Presença relativa por faixa de valor, meses selecionados")
print(P[P.periodo.isin(["2019-12", "2020-02", "2020-03", "2020-04", "2020-05", "2020-07", "2020-12", "2025-12"])]
      .pivot_table(index=["direcao", "faixa"], columns="periodo", values="relativa").round(2).to_string())
print(rel.groupby(["direcao", "faixa"]).size().to_string())
print("\n== Fronteira: estabelecimentos de fora de SP no círculo, 2018")
print(F[F.periodo.between("2018-01", "2019-03")][["periodo", "nos_fora_circulo", "frac_fora_n", "frac_fora_v"]].round(3).to_string(index=False))
