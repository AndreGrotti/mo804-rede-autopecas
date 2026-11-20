# Célula 2 — o núcleo persistente é também o centro estrutural?
# Densidade por blocos, camadas de k-core e gradiente de persistência, contra a rede sorteada com os mesmos graus.
# Saídas só agregadas (sem CNPJ) em resultados/N_*.csv. Retomável: trimestres já gravados são pulados.
import os, time, random
import numpy as np, pandas as pd, igraph as ig

BASE = "/home/candre/dados/rede/disciplina/serie"
RES = os.path.join(BASE, "resultados")
ANOS = list(range(2017, 2026))
TRIS = [f"{a}T{q}" for a in ANOS for q in range(1, 5)]
NSORT = 2                     # redes sorteadas por trimestre
SEMENTE_RNG = 20261116
EXTREMOS = {2017, 2018, 2019, 2024, 2025}
ARQ = {k: os.path.join(RES, f"N_{k}.csv") for k in ("blocos", "kcore", "gradiente")}

# ---------- 1. nós de cada trimestre: sementes + quem negociou com elas ----------
c1 = pd.read_parquet(os.path.join(BASE, "camada1_mensal.parquet"),
                     columns=["semente", "parceiro", "mes", "uf_parceiro"])
if c1.mes.max() <= 12:
    raise ValueError("camada1.mes não traz o ano; ajustar a conversão do mês")
c1["tri"] = (c1.mes // 100).astype(str) + "T" + ((c1.mes % 100 - 1) // 3 + 1).astype(str)
c1 = c1[c1.tri.isin(TRIS)]

sementes = set(c1.semente.unique())
for a in ANOS:
    n = pd.read_parquet(os.path.join(BASE, f"nos_{a}.parquet"))
    sementes |= set(n.loc[n.semente, "cnpj"])
LS = list(sementes)

uf = c1.drop_duplicates("parceiro").set_index("parceiro").uf_parceiro
fora = {c: (u != "SP") for c, u in uf.items()}
nos_tri = {t: set(g.parceiro) | set(g.semente) for t, g in c1.groupby("tri")}
print(f"sementes: {len(sementes)} | trimestres com nós: {len(nos_tri)}")

# ---------- 2. persistência e núcleo ----------
pres = pd.Series([c for t in TRIS for c in nos_tri.get(t, ())]).value_counts()   # nº de trimestres presentes
anos_de = {}
for t in TRIS:
    for c in nos_tri.get(t, ()):
        anos_de.setdefault(c, set()).add(int(t[:4]))
todos = {c for c, s in anos_de.items() if len(s) == len(ANOS)}
extr = {c for c, s in anos_de.items() if EXTREMOS <= s}
print(f"presentes em todos os anos: {len(todos)} ({len(todos - sementes)} sem sementes) | referência 3.522")
print(f"presentes nos cinco anos dos extremos: {len(extr)} ({len(extr - sementes)} sem sementes) | referência 4.020")
nucleo = todos - sementes
LN = list(nucleo)

ROT = ["1", "2–4", "5–12", "13–24", "25–35", "36"]
classe = pd.cut(pres, bins=[0, 1, 4, 12, 24, 35, 36], labels=ROT).astype(str).to_dict()

# ---------- 3. funções ----------
_cache = {}
def arestas_ano(a):
    if a not in _cache:
        _cache.clear()
        e = pd.read_parquet(os.path.join(BASE, f"arestas_mensais_{a}.parquet"), columns=["origem", "destino", "mes"])
        e["m"] = e.mes % 100 if e.mes.max() > 12 else e.mes
        _cache[a] = e
    return _cache[a]

def grafo(t):
    a, q = int(t[:4]), int(t[-1])
    e = arestas_ano(a)
    e = e[e.m.between(3 * q - 2, 3 * q)]
    V = pd.Index(sorted(nos_tri[t]))
    ia, ib = V.get_indexer(e.origem), V.get_indexer(e.destino)
    ok = (ia >= 0) & (ib >= 0) & (ia != ib)
    par = np.unique(np.sort(np.c_[ia[ok], ib[ok]], axis=1), axis=0)
    return V, ig.Graph(n=len(V), edges=par.tolist())

def blocos(g, gr):
    E = np.array(g.get_edgelist())
    a, b = gr[E[:, 0]], gr[E[:, 1]]
    N, P = np.sum(gr == 1), np.sum(gr == 2)
    nn = np.sum((a == 1) & (b == 1)); pp = np.sum((a == 2) & (b == 2))
    np_ = np.sum(((a == 1) & (b == 2)) | ((a == 2) & (b == 1)))
    return dict(dens_NN=nn / (N * (N - 1) / 2), dens_NP=np_ / (N * P), dens_PP=pp / (P * (P - 1) / 2))

def kcore(g, gr, fo):
    ns = np.where(gr != 0)[0]                      # sem as sementes e sem as ligações com elas
    h = g.induced_subgraph(ns.tolist())
    k, d = np.array(h.coreness()), np.array(h.degree())
    gns, fns = gr[ns], fo[ns]
    thr = np.quantile(k, 0.9); top = k >= thr; inn = k == k.max()
    r = dict(kmax=int(k.max()), k_p90=float(thr), frac_nucleo=np.mean(gns == 1),
             frac_nucleo_top10=np.mean(gns[top] == 1), frac_nucleo_kmax=np.mean(gns[inn] == 1), n_kmax=int(inn.sum()))
    for nome, m in (("SP", ~fns), ("fora", fns)):
        r[f"frac_nucleo_{nome}"] = np.mean(gns[m] == 1) if m.any() else np.nan
        r[f"frac_nucleo_top10_{nome}"] = np.mean(gns[m & top] == 1) if (m & top).any() else np.nan
    return r, k, d, top, ns

# ---------- 4. laço por trimestre (retomável) ----------
tabs = {k: [] for k in ARQ}
feitos = set()
if all(os.path.exists(p) for p in ARQ.values()):
    for k, p in ARQ.items():
        tabs[k] = [pd.read_csv(p, sep=";")]
    feitos = set(tabs["blocos"][0].trimestre)
    print(f"retomando: {len(feitos)} trimestres já gravados")

for i, t in enumerate(TRIS):
    if t in feitos or t not in nos_tri:
        continue
    t0 = time.time()
    V, g = grafo(t)
    gr = np.where(V.isin(LS), 0, np.where(V.isin(LN), 1, 2))
    fo = np.array([fora.get(c, False) for c in V])

    b_obs = blocos(g, gr)
    k_obs, k, d, top, ns = kcore(g, gr, fo)
    b_al, k_al = [], []
    for s in range(NSORT):
        random.seed(SEMENTE_RNG + 100 * i + s)
        r = g.copy(); r.rewire(10 * g.ecount())
        b_al.append(blocos(r, gr)); k_al.append(kcore(r, gr, fo)[0])
    b_al, k_al = pd.DataFrame(b_al).mean(), pd.DataFrame(k_al).mean()

    lb = dict(trimestre=t, nos=g.vcount(), arestas=g.ecount(), n_nucleo=int(np.sum(gr == 1)), n_perif=int(np.sum(gr == 2)))
    lb.update(b_obs); lb.update({f"{c}_aleat": v for c, v in b_al.items()})
    lk = dict(trimestre=t); lk.update(k_obs); lk.update({f"{c}_aleat": v for c, v in k_al.items()})
    cl = np.array([classe.get(c, "?") for c in V[ns]])
    ga = (pd.DataFrame(dict(classe=cl, fora=fo[ns], grau=d, k=k, top=top))
            .groupby(["classe", "fora"]).agg(n=("grau", "size"), grau_medio=("grau", "mean"),
                                             k_medio=("k", "mean"), frac_top10=("top", "mean")).reset_index())
    ga.insert(0, "trimestre", t)

    tabs["blocos"].append(pd.DataFrame([lb])); tabs["kcore"].append(pd.DataFrame([lk])); tabs["gradiente"].append(ga)
    for kk, p in ARQ.items():
        pd.concat(tabs[kk], ignore_index=True).to_csv(p, sep=";", index=False)
    print(f"{t}: {g.vcount()} nós, {g.ecount()} arestas, núcleo {lb['n_nucleo']} ({time.time() - t0:.0f}s)")

B = pd.concat(tabs["blocos"], ignore_index=True).set_index("trimestre").loc[[t for t in TRIS if t in nos_tri]]
K = pd.concat(tabs["kcore"], ignore_index=True).set_index("trimestre").loc[B.index]
G = pd.concat(tabs["gradiente"], ignore_index=True)

# ---------- 5. conferência contra A_metricas ----------
A = pd.read_csv(os.path.join(RES, "A_metricas.csv"), sep=";")
A = A[(A.regra == "trimestre") & (A.versao == "completa")].set_index("trimestre")
conf = pd.DataFrame({"nos": B.nos, "nos_A": A.vertices, "arestas": B.arestas, "arestas_A": A.arestas}).dropna()
conf["dif_nos_%"] = 100 * (conf.nos / conf.nos_A - 1)
conf["dif_arestas_%"] = 100 * (conf.arestas / conf.arestas_A - 1)
print("\n== Conferência com A_metricas (diferença %, máx. absoluta):",
      f"nós {conf['dif_nos_%'].abs().max():.2f} | arestas {conf['dif_arestas_%'].abs().max():.2f}")
print(conf.loc[["2017T1", "2020T2", "2025T4"]].round(2).to_string())

# ---------- 6. resumos por período ----------
PER = {"2017–19": TRIS[:12], "2019T4–20T1": TRIS[11:13], "2020T2": TRIS[13:14],
       "2020T3–21T4": TRIS[14:20], "2024–25": TRIS[28:]}
def media(df, cols):
    return pd.DataFrame({p: df.loc[[t for t in ts if t in df.index], cols].mean() for p, ts in PER.items()}).T

b = media(B, ["dens_NN", "dens_NP", "dens_PP", "dens_NN_aleat", "dens_NP_aleat", "dens_PP_aleat"])
for c in ("NN", "NP", "PP"):
    b[f"{c}_obs/aleat"] = b[f"dens_{c}"] / b[f"dens_{c}_aleat"]
b[[c for c in b if c.startswith("dens")]] *= 1000
print("\n== Densidade por blocos (por mil): núcleo–núcleo, núcleo–periferia, periferia–periferia (sem sementes)")
print(b.round(3).to_string())

kk = media(K, ["frac_nucleo", "frac_nucleo_top10", "frac_nucleo_top10_aleat", "frac_nucleo_kmax", "frac_nucleo_kmax_aleat",
               "frac_nucleo_SP", "frac_nucleo_top10_SP", "frac_nucleo_fora", "frac_nucleo_top10_fora", "kmax", "kmax_aleat"])
kk["enriq_top10"] = kk.frac_nucleo_top10 / kk.frac_nucleo
kk["enriq_top10_aleat"] = kk.frac_nucleo_top10_aleat / kk.frac_nucleo
print("\n== k-core (sem sementes): fração do núcleo entre todos, nos 10% mais internos e na camada máxima")
print(kk.round(3).to_string())

G["s_grau"], G["s_k"], G["s_top"] = G.grau_medio * G.n, G.k_medio * G.n, G.frac_top10 * G.n
def grad(df, por):
    s = df.groupby(por)[["n", "s_grau", "s_k", "s_top"]].sum()
    return pd.DataFrame({"nós-trimestre": s.n, "grau_medio": s.s_grau / s.n,
                         "k_medio": s.s_k / s.n, "frac_top10": s.s_top / s.n})
print("\n== Gradiente de persistência (nº de trimestres presentes, de 36), todos os trimestres juntos")
print(grad(G, "classe").reindex(ROT).round(3).to_string())
print("\n== O mesmo, separado entre SP e fora de SP")
print(grad(G, ["fora", "classe"]).round(3).to_string())
