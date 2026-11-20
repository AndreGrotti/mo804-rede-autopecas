# Célula U1 (revista) — regra nova de UF: classificação, fronteira, índices de valor e rede só SP (mensal e trimestral)
# Regra: sementes = SP; senão UF predominante (em valor) como EMITENTE; senão UF da Receita; senão UF predominante como destinatário
import os, ast, time, numpy as np, pandas as pd, igraph as ig
t0 = time.time()
BASE = "/home/candre/dados/rede/disciplina/serie"
RES, INT = os.path.join(BASE, "resultados"), os.path.join(BASE, "interno")
ANOS = list(range(2017, 2026))
rot = lambda a, m: f"{a}-{m:02d}"
z14 = lambda c: str(c).strip().zfill(14)
num = lambda s: pd.to_numeric(s, errors="coerce").astype(float).fillna(0.0)
limpa = lambda s: s.astype(object).where(s.notna(), "").astype(str).str.strip().str.upper().to_numpy(dtype=object)
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
def le(arq):
    d = pd.read_csv(os.path.join(RES, arq), sep=";")
    return d if d.shape[1] > 1 else pd.read_csv(os.path.join(RES, arq))

# ---------- 1. nós, sementes, regra antiga e evidências de UF ----------
c1 = pd.read_parquet(os.path.join(BASE, "camada1_mensal.parquet"),
                     columns=["semente", "parceiro", "mes", "uf_parceiro", "v", "direcao"])
c1 = c1[(c1.mes // 100).between(ANOS[0], ANOS[-1])].copy()
c1["v"], c1["uf"] = num(c1.v), limpa(c1.uf_parceiro)
sementes = set(c1.semente.unique())
for a in ANOS:
    n = pd.read_parquet(os.path.join(BASE, f"nos_{a}.parquet"))
    sementes |= set(n.loc[n.semente.astype(bool), "cnpj"])
nos_mes = {(int(k) // 100, int(k) % 100): set(g.parceiro) | set(g.semente) for k, g in c1.groupby("mes")}
todos_nos = set().union(*nos_mes.values())
sp_set = set(sementes) | set(c1.loc[c1.uf == "SP", "parceiro"])
w = lambda v: v.clip(lower=0) + 1e-6                      # peso em valor; o épsilon desempata por contagem
mont, jus = c1[c1.direcao.eq("montante")], c1[c1.direcao.eq("jusante")]
ue = [mont.assign(w=w(mont.v)).groupby(["parceiro", "uf"]).w.sum()]   # montante: o parceiro vende (é o emitente)
ud = [jus.assign(w=w(jus.v)).groupby(["parceiro", "uf"]).w.sum()]     # jusante: o parceiro compra (é o destinatário)
for a in ANOS:
    e = pd.read_parquet(os.path.join(BASE, f"arestas_mensais_{a}.parquet"),
                        columns=["origem", "destino", "uf_origem", "uf_destino", "v"])
    e["uo"], e["ud"], e["w"] = limpa(e.uf_origem), limpa(e.uf_destino), w(num(e.v))
    sp_set |= set(e.loc[e.uo == "SP", "origem"]) | set(e.loc[e.ud == "SP", "destino"])
    ue.append(e.groupby(["origem", "uo"]).w.sum().rename_axis(["parceiro", "uf"]))
    ud.append(e.groupby(["destino", "ud"]).w.sum().rename_axis(["parceiro", "uf"]))
    del e
    print(f"{a}: arestas lidas ({time.time() - t0:.0f}s)")
fora_orig = todos_nos - sp_set
print(f"nós da série: {len(todos_nos)} | de fora pela regra antiga: {len(fora_orig)} (na F1: 12.440 de 42.813)")

def predominante(partes):
    U = pd.concat(partes).groupby(level=[0, 1]).sum().reset_index()
    U = U[U.uf.astype(str).str.len() == 2]
    U["sh"] = U.w / U.groupby("parceiro").w.transform("sum")
    n_uf = U.groupby("parceiro").uf.nunique()
    U = U.sort_values("w").drop_duplicates("parceiro", keep="last").set_index("parceiro")
    return U.uf, U.sh, n_uf
uf_e, sh_e, n_e = predominante(ue)
uf_d, sh_d, _ = predominante(ud)
del ue, ud

# ---------- 2. Receita: UF atual de todos os nós (uma consulta, em lotes) ----------
ARQ = "/home/candre/dados/Projeto_Rede/extrai_c2_mes .py"
fonte = open(ARQ, encoding="utf-8").read(); arvore = ast.parse(fonte)
mantidos = [x for x in arvore.body if isinstance(x, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (isinstance(x, (ast.Assign, ast.AnnAssign)) and not any(isinstance(y, ast.Call) for y in ast.walk(x)))]
ns = {}; exec(compile(ast.Module(body=mantidos, type_ignores=[]), ARQ, "exec"), ns)
fconex = [x.name for x in mantidos if isinstance(x, ast.FunctionDef) and "pyodbc.connect" in ast.get_source_segment(fonte, x)]
if not fconex:
    raise RuntimeError("não achei no script a função que chama pyodbc.connect")
conn = ns[fconex[0]]()
lista = sorted(todos_nos)
chaves = sorted({z14(c) for c in lista}); partes = []
for i in range(0, len(chaves), 1000):
    inl = ",".join("'" + c + "'" for c in chaves[i:i + 1000])
    partes.append(pd.read_sql(f"""SELECT num_cnpj, sigla_uf, ind_excluido, data_incl_ods
                                  FROM ods.tb_dsod_bcadastros_cnpj_estabelecimento WHERE num_cnpj IN ({inl})""", conn))
conn.close()
rfb = pd.concat(partes, ignore_index=True)
rfb["c"] = rfb.num_cnpj.astype(str).str.strip().str.zfill(14); rfb["uf"] = limpa(rfb.sigla_uf)
rfb = rfb[rfb.uf.astype(str).str.len() == 2]
rfb = rfb.sort_values(["ind_excluido", "data_incl_ods"], ascending=[False, True]).drop_duplicates("c", keep="last").set_index("c")
print(f"Receita (consulta de {time.strftime('%d/%m/%Y')}): {len(rfb)} de {len(chaves)} nós encontrados | "
      f"carga de {rfb.data_incl_ods.min()} a {rfb.data_incl_ods.max()} ({time.time() - t0:.0f}s)")

# ---------- 3. classificação nova e comparação com a antiga ----------
C = pd.DataFrame(index=pd.Index(lista, name="cnpj"))
C["chave"] = [z14(c) for c in lista]
C["semente"] = C.index.isin(sementes)
C["uf_emit"], C["sh_emit"], C["n_uf_emit"] = uf_e, sh_e, n_e
C["uf_rfb"] = C.chave.map(rfb.uf)
C["uf_dest"], C["sh_dest"] = uf_d, sh_d
C["uf_nova"] = C.uf_emit.fillna(C.uf_rfb).fillna(C.uf_dest).fillna("").astype(object)
C.loc[C.semente, "uf_nova"] = "SP"
C["fonte"] = np.select([C.semente.to_numpy(), C.uf_emit.notna().to_numpy(), C.uf_rfb.notna().to_numpy(), C.uf_dest.notna().to_numpy()],
                       ["semente", "emitente", "receita", "destinatario"], "sem UF")
fora_novo = set(C.index[(~C.semente & C.uf_nova.ne("SP")).to_numpy()])   # sem UF conta como fora, como na regra antiga
C["antiga"] = np.where(C.index.isin(list(fora_orig)), "fora", "SP")
C["nova"] = np.where(C.index.isin(list(fora_novo)), "fora", "SP")
print("\n== 3a. Antiga (linhas) × nova (colunas), em nº de estabelecimentos")
print(pd.crosstab(C.antiga, C.nova, margins=True).to_string())
print("\n== 3b. Fonte da UF nova, por célula da tabela acima")
print(pd.crosstab([C.antiga, C.nova], C.fonte).to_string())
c1v = c1[c1.v > 0].copy()
c1v["antiga"], c1v["nova"] = c1v.parceiro.map(C.antiga), c1v.parceiro.map(C.nova)
x = c1v.pivot_table(index="antiga", columns="nova", values="v", aggfunc="sum", fill_value=0)
print("\n== 3c. Valor das sementes (fração do total), antiga × nova"); print((x / x.values.sum()).round(4).to_string())
b = C[C.uf_emit.notna() & C.uf_rfb.notna() & ~C.semente]
print(f"\n== 3d. Emitente × Receita (onde há as duas): {len(b)} nós | mesma UF {(b.uf_emit == b.uf_rfb).mean():.4f} | "
      f"divergem em SP × fora: {((b.uf_emit == 'SP') != (b.uf_rfb == 'SP')).sum()}")
print(f"emitem por mais de uma UF: {(C.n_uf_emit > 1).sum()} | UF principal de emissão < 90% do valor: {(C.sh_emit < 0.9).sum()} "
      f"| sem UF nenhuma: {(C.fonte == 'sem UF').sum()}")
mist = pd.read_csv(os.path.join(INT, "uf_mista_parceiros.csv"), sep=";", dtype=str).iloc[:, 0].str.strip().str.zfill(14)
print(f"dos {len(mist)} parceiros de UF mista (P1): achados {mist.isin(set(C.chave)).sum()} | "
      f"a regra nova põe fora de SP: {mist.isin(set(C.chave[C.nova == 'fora'])).sum()}")
C.to_csv(os.path.join(INT, "uf_regra_nova.csv"), sep=";")

# ---------- 4. fronteira e índices de valor (regra nova; a antiga ao lado) ----------
c1v["fn"], c1v["fa"] = c1v.parceiro.isin(fora_novo), c1v.parceiro.isin(fora_orig)
c1v["vf"], c1v["vfa"] = np.where(c1v.fn, c1v.v, 0.0), np.where(c1v.fa, c1v.v, 0.0)
c1v["vs"] = c1v.v - c1v.vf
q = c1v.groupby(["mes", "parceiro"], as_index=False)[["fn", "fa"]].max()
Fv = c1v.groupby("mes")[["v", "vf", "vfa", "vs"]].sum()
F = pd.DataFrame(index=Fv.index)
F["frac_fora_n"], F["frac_fora_n_antiga"] = q.groupby("mes").fn.mean(), q.groupby("mes").fa.mean()
F["frac_fora_v"], F["frac_fora_v_antiga"] = Fv.vf / Fv.v, Fv.vfa / Fv.v
base = Fv.loc[[202001, 202002]].mean()
F["idx_v_fora"], F["idx_v_sp"] = 100 * Fv.vf / base.vf, 100 * Fv.vs / base.vs
F["nos_fora_circulo"] = [len(nos_mes.get((m // 100, m % 100), set()) & fora_novo) for m in F.index]
F["nos_fora_circulo_antiga"] = [len(nos_mes.get((m // 100, m % 100), set()) & fora_orig) for m in F.index]
F.insert(0, "periodo", [rot(m // 100, m % 100) for m in F.index])
F.to_csv(os.path.join(RES, "F_fronteira_mensal_v3.csv"), sep=";", index=False)
F2 = le("F_fronteira_mensal_v2.csv").set_index("periodo")
print(f"\nconferência com a F1: dif. máx. frac_fora_n antiga {(F.set_index('periodo').frac_fora_n_antiga - F2.frac_fora_n).abs().max():.2e} | "
      f"frac_fora_v antiga {(F.set_index('periodo').frac_fora_v_antiga - F2.frac_fora_v).abs().max():.2e}")
print("\n== 4a. Fronteira mensal em torno do choque (índices: média de jan–fev/2020 = 100)")
print(F[F.periodo.between("2019-10", "2020-12")].round(3).to_string(index=False))
c1v["tri"] = (c1v.mes // 100).astype(str) + "T" + ((c1v.mes % 100 - 1) // 3 + 1).astype(str)
Fq = c1v.groupby("tri")[["v", "vf", "vs"]].sum()
Fq = pd.DataFrame({"frac_fora_v": Fq.vf / Fq.v, "idx_v_fora": 100 * Fq.vf / Fq.vf["2020T1"], "idx_v_sp": 100 * Fq.vs / Fq.vs["2020T1"]})
print("\n== 4b. Trimestral (índices: 2020T1 = 100)"); print(Fq.loc["2019T3":"2021T2"].round(3).to_string())
Fa = F.assign(ano=F.index // 100).groupby("ano")[["frac_fora_v", "frac_fora_v_antiga", "frac_fora_n", "frac_fora_n_antiga"]].mean()
print("\n== 4c. Médias anuais das frações mensais"); print(Fa.round(3).to_string())

# ---------- 5. rede só SP: as duas regras no mesmo grafo ----------
_cache = {}
def arestas_ano(a):
    if a not in _cache:
        _cache.clear()
        e = pd.read_parquet(os.path.join(BASE, f"arestas_mensais_{a}.parquet"), columns=["origem", "destino", "mes"])
        e["m"] = (e.mes % 100 if e.mes.max() > 12 else e.mes).astype(int)
        _cache[a] = e[e.origem != e.destino]
    return _cache[a]
def grafo(nos, meses):
    e = arestas_ano(meses[0][0]); e = e[e.m.isin([m for _, m in meses])]
    V = pd.Index(sorted(nos)); ia, ib = V.get_indexer(e.origem), V.get_indexer(e.destino)
    ok = (ia >= 0) & (ib >= 0)
    par = np.unique(np.sort(np.c_[ia[ok], ib[ok]], axis=1), axis=0)
    return V, ig.Graph(n=len(V), edges=par.tolist())
def medidas(g):
    n, m = g.vcount(), g.ecount()
    return dict(vertices=n, arestas=m, grau_medio=2 * m / n, densidade=2 * m / (n * (n - 1)),
                transitividade=g.transitivity_undirected(),
                clust_medio_zero=float(np.mean(g.transitivity_local_undirected(mode="zero"))),
                assortatividade=g.assortativity_degree(directed=False))
def redes_sp(meses, sem_iso):
    V, g = grafo(set().union(*[nos_mes.get(x, set()) for x in meses]), meses)
    out = {}
    for regra, fs in (("antiga", fora_orig), ("nova", fora_novo)):
        h = g.induced_subgraph([i for i, c in enumerate(V) if c not in fs])
        if sem_iso: h.delete_vertices(h.vs.select(_degree=0))
        out[regra] = medidas(h)
    return out
COLS = ["vertices", "arestas", "grau_medio", "densidade", "transitividade", "assortatividade"]
Mall = le("M_metricas.csv")
Mold = Mall[Mall.versao.eq("so_SP")]
if "regra" in Mold: Mold = Mold[Mold.regra.eq("mes")]
Mold = Mold.drop_duplicates("periodo", keep="last").set_index("periodo")
A = le("A_metricas.csv"); Aold = A[A.versao.eq("so_SP")]
if "regra" in Aold: Aold = Aold[Aold.regra.eq("trimestre")]
Aold = Aold.drop_duplicates("trimestre", keep="last").set_index("trimestre")
conv = {}
for freq, alvos in (("mensal", [(2017, [1]), (2020, [4]), (2025, [12])]),
                    ("trimestral", [(2017, [1, 2, 3]), (2020, [4, 5, 6]), (2025, [10, 11, 12])])):
    difs = {}
    for sem_iso in (False, True):
        d = []
        for a, ms in alvos:
            r = redes_sp([(a, m) for m in ms], sem_iso)["antiga"]
            ref = Mold.loc[rot(a, ms[0])] if freq == "mensal" else Aold.loc[f"{a}T{(ms[0] - 1) // 3 + 1}"]
            d += [abs(100 * (r[c] / ref[c] - 1)) for c in COLS]
        difs[sem_iso] = max(d)
    conv[freq] = min(difs, key=difs.get) if min(difs.values()) <= 0.5 else False
    print(f"\n== 5. Regra antiga recalculada × série gravada ({freq}): dif. máx. % com isolados {difs[False]:.3f} | "
          f"sem isolados {difs[True]:.3f} | usando: {'sem' if conv[freq] else 'com'} isolados")
    if min(difs.values()) > 0.5:
        print(f"   AVISO: nenhuma convenção reproduz a série {freq} gravada; a comparação abaixo vale entre as duas regras recalculadas")
lin_m, lin_t = [], []
for a in ANOS:
    for m in range(1, 13):
        if (a, m) in nos_mes:
            for regra, r in redes_sp([(a, m)], conv["mensal"]).items():
                lin_m.append(dict(periodo=rot(a, m), regra_uf=regra, **r))
    for qq in range(1, 5):
        for regra, r in redes_sp([(a, m) for m in range(3 * qq - 2, 3 * qq + 1)], conv["trimestral"]).items():
            lin_t.append(dict(trimestre=f"{a}T{qq}", regra_uf=regra, **r))
    print(f"{a}: só SP com as duas regras ({time.time() - t0:.0f}s)")
Mx, Ax = pd.DataFrame(lin_m), pd.DataFrame(lin_t)
Mx.to_csv(os.path.join(RES, "M_metricas_soSP_regra_nova.csv"), sep=";", index=False)
Ax.to_csv(os.path.join(RES, "A_metricas_soSP_regra_nova.csv"), sep=";", index=False)
Mn = Mx[Mx.regra_uf == "nova"].drop(columns="regra_uf")
pd.concat([Mall[Mall.versao != "so_SP_nova"], Mn.assign(regra="mes", versao="so_SP_nova")],
          ignore_index=True).to_csv(os.path.join(RES, "M_metricas_v2.csv"), sep=";", index=False)

# ---------- 6. antigo × corrigido ----------
def lado(D, chave):
    P = D.pivot(index=chave, columns="regra_uf", values=COLS)
    P.columns = [f"{c}_{r}" for c, r in P.columns]
    return P
cmp, cq = lado(Mx, "periodo"), lado(Ax, "trimestre")
ger = Mx[Mx.regra_uf == "antiga"].set_index("periodo")[COLS]
print(f"\nRegra antiga recalculada × M_metricas gravado, 108 meses: dif. máx. % "
      f"{(100 * (ger / Mold.loc[ger.index, COLS] - 1)).abs().max().max():.3f}")
print("\n== 6a. Só SP mensal: diferença % (nova ÷ antiga − 1), média e extremos nos 108 meses")
print(pd.DataFrame({c: 100 * (cmp[c + "_nova"] / cmp[c + "_antiga"] - 1) for c in COLS}).describe().loc[["mean", "min", "max"]].round(2).to_string())
mostra = ["vertices", "grau_medio", "transitividade", "assortatividade"]
cols_lado = [f"{c}_{r}" for c in mostra for r in ("antiga", "nova")]
print("\n== 6b. Só SP mensal em torno do choque (antiga | nova)")
print(cmp.loc["2019-09":"2020-12", cols_lado].round(4).to_string())
print("\n== 6c. Só SP trimestral (antiga | nova)")
print(cq.loc["2018T1":"2021T4", cols_lado].round(4).to_string())
print(f"\nGravados em resultados: F_fronteira_mensal_v3, M_metricas_soSP_regra_nova, A_metricas_soSP_regra_nova, M_metricas_v2 | "
      f"em interno (com CNPJ): uf_regra_nova | {(time.time() - t0) / 60:.1f} min")
