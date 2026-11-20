# Célula M1 — métricas mensais (principal, só SP e painel), com conferência contra a série trimestral
# Retomável: meses já gravados em resultados/M_metricas.csv são pulados.
import os, time, random
import numpy as np, pandas as pd, igraph as ig

BASE  = "/home/candre/dados/rede/disciplina/serie"
RES   = os.path.join(BASE, "resultados")
SAIDA = os.path.join(RES, "M_metricas.csv")
ANOS  = list(range(2017, 2026))
MESES = [(a, m) for a in ANOS for m in range(1, 13)]
NSORT = 3                      # redes sorteadas por mês (principal e painel)
SEMENTE_RNG = 20261117

# ---------- 1. nós de cada mês, sementes, UF e painel ----------
c1 = pd.read_parquet(os.path.join(BASE, "camada1_mensal.parquet"), columns=["semente", "parceiro", "mes", "uf_parceiro"])
c1 = c1[(c1.mes // 100).between(ANOS[0], ANOS[-1])]
sementes = set(c1.semente.unique())
for a in ANOS:
    n = pd.read_parquet(os.path.join(BASE, f"nos_{a}.parquet"))
    sementes |= set(n.loc[n.semente, "cnpj"])
uf = c1.drop_duplicates("parceiro").set_index("parceiro").uf_parceiro
fora = set(uf[uf != "SP"].index) - sementes
nos_mes = {(int(k) // 100, int(k) % 100): set(g.parceiro) | set(g.semente) for k, g in c1.groupby("mes")}
anos_de = {}
for (a, m), s in nos_mes.items():
    for c in s:
        anos_de.setdefault(c, set()).add(a)
painel = {c for c, s in anos_de.items() if len(s) == len(ANOS)}
print(f"sementes {len(sementes)} | meses com nós {len(nos_mes)} de {len(MESES)} | painel {len(painel)} (referência 3.522)")

# ---------- 2. funções ----------
_cache = {}
def arestas_ano(a):
    if a not in _cache:
        _cache.clear()
        e = pd.read_parquet(os.path.join(BASE, f"arestas_mensais_{a}.parquet"), columns=["origem", "destino", "mes"])
        e["m"] = (e.mes % 100 if e.mes.max() > 12 else e.mes).astype(int)
        _cache[a] = e[e.origem != e.destino]
    return _cache[a]

def grafo(nos, meses):
    """Subgrafo induzido pelas vendas dos meses dados (mesmo ano) entre os nós dados; sem direção e sem peso."""
    a = meses[0][0]
    e = arestas_ano(a)
    e = e[e.m.isin([m for _, m in meses])]
    V = pd.Index(sorted(nos))
    ia, ib = V.get_indexer(e.origem), V.get_indexer(e.destino)
    ok = (ia >= 0) & (ib >= 0)
    par = np.unique(np.sort(np.c_[ia[ok], ib[ok]], axis=1), axis=0)
    return V, ig.Graph(n=len(V), edges=par.tolist())

def medidas(g, aleat):
    n, m = g.vcount(), g.ecount()
    r = dict(vertices=n, arestas=m, grau_medio=2 * m / n, densidade=2 * m / (n * (n - 1)),
             transitividade=g.transitivity_undirected(),
             clust_medio_zero=float(np.mean(g.transitivity_local_undirected(mode="zero"))),
             assortatividade=g.assortativity_degree(directed=False))
    if aleat:
        t, c = [], []
        for _ in range(NSORT):
            h = g.copy(); h.rewire(10 * m)
            t.append(h.transitivity_undirected())
            c.append(np.mean(h.transitivity_local_undirected(mode="zero")))
        r.update(transitividade_aleat=float(np.mean(t)), clust_medio_zero_aleat=float(np.mean(c)))
    return r

def sem_isolados(h):
    h = h.copy(); h.delete_vertices(h.vs.select(_degree=0)); return h

def versoes(meses, semente_rng, sp_sem_isolados, aleat=True):
    random.seed(semente_rng)
    nos = set().union(*[nos_mes.get(x, set()) for x in meses])
    V, g = grafo(nos, meses)
    out = {}
    out["completa"] = medidas(g, aleat)
    out["completa"]["sementes_no_grafo"] = int(V.isin(list(sementes)).sum())
    sp = g.induced_subgraph([i for i, c in enumerate(V) if c not in fora])
    out["so_SP"] = medidas(sem_isolados(sp) if sp_sem_isolados else sp, False)
    Vp, gp = grafo(painel, meses)
    iso = int(sum(1 for d in gp.degree() if d == 0))
    out["painel_todos"] = medidas(sem_isolados(gp), aleat)
    out["painel_todos"]["nos_sem_aresta"] = iso
    return out

# ---------- 3. conferência: a mesma regra em trimestres deve reproduzir A_metricas e A_painel ----------
A = pd.read_csv(os.path.join(RES, "A_metricas.csv"), sep=";")
A = A[A.regra == "trimestre"].set_index(["versao", "trimestre"])
AP = pd.read_csv(os.path.join(RES, "A_painel.csv"), sep=";")
AP = AP[AP.painel == "todos"].set_index("trimestre")
COLS = ["vertices", "arestas", "grau_medio", "densidade", "transitividade", "assortatividade"]
linhas, escolha = [], {}
for t in ["2017T1", "2020T2", "2025T4"]:
    a, q = int(t[:4]), int(t[-1])
    meses = [(a, m) for m in range(3 * q - 2, 3 * q + 1)]
    for sp_iso in (False, True):
        r = versoes(meses, SEMENTE_RNG, sp_iso, aleat=False)
        if not sp_iso:
            for c in COLS:
                linhas.append(("completa", t, c, r["completa"][c], A.loc[("completa", t), c]))
                linhas.append(("painel_todos", t, c, r["painel_todos"][c], AP.loc[t, c]))
        for c in COLS:
            linhas.append((f"so_SP {'sem' if sp_iso else 'com'} isolados", t, c, r["so_SP"][c], A.loc[("so_SP", t), c]))
conf = pd.DataFrame(linhas, columns=["versao", "trimestre", "medida", "nosso", "serie"])
conf["dif_%"] = 100 * (conf.nosso / conf.serie - 1)
print("\n== Conferência com a série trimestral (diferença %, máxima absoluta por versão)")
print(conf.groupby("versao")["dif_%"].apply(lambda s: s.abs().max()).round(3).to_string())
dif_sp = conf[conf.versao.str.startswith("so_SP")].groupby("versao")["dif_%"].apply(lambda s: s.abs().max())
SP_SEM_ISOLADOS = dif_sp.idxmin().startswith("so_SP sem")
print(f"só SP: uso a convenção '{dif_sp.idxmin()}'")
if conf.groupby("versao")["dif_%"].apply(lambda s: s.abs().max()).min() > 0.5:
    print("ATENÇÃO: nenhuma versão reproduz a série trimestral dentro de 0,5%; mande esta saída antes de seguir")

# ---------- 4. laço mensal (retomável) ----------
feitos = set()
tab = []
if os.path.exists(SAIDA):
    ant = pd.read_csv(SAIDA, sep=";")
    tab = [ant]
    feitos = set(ant.periodo)
    print(f"retomando: {len(feitos)} meses já gravados")
t0 = time.time()
for i, (a, m) in enumerate(MESES):
    p = f"{a}-{m:02d}"
    if p in feitos or (a, m) not in nos_mes:
        continue
    t1 = time.time()
    r = versoes([(a, m)], SEMENTE_RNG + i, SP_SEM_ISOLADOS)
    tab.append(pd.DataFrame([dict(regra="mes", versao=v, periodo=p, **d) for v, d in r.items()]))
    pd.concat(tab, ignore_index=True).to_csv(SAIDA, sep=";", index=False)
    print(f"{p}: {r['completa']['vertices']} nós, {r['completa']['arestas']} arestas ({time.time() - t1:.0f}s)")
M = pd.concat(tab, ignore_index=True)
print(f"\nGravado {SAIDA}: {len(M)} linhas, {M.periodo.nunique()} meses ({(time.time() - t0) / 60:.1f} min)")
print(M.groupby("versao").size().to_string())
