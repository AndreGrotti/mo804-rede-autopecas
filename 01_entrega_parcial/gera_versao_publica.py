# ---------- Geração da versão pública do dataset (roda UMA vez, no ambiente interno) ----------
import os, hmac, hashlib, secrets, numpy as np, pandas as pd, igraph as ig
base = '/home/candre/dados/rede/'; pasta = base + 'disciplina/'; ANO = 2018
pub = pasta + 'publico/'; os.makedirs(pub, exist_ok=True)

# 1. chave secreta do hash: gerada uma vez, guardada fora da pasta pública, nunca compartilhada
arq_chave = os.path.expanduser('~/chave_hash_rede.txt')
if not os.path.exists(arq_chave):
    open(arq_chave, 'w').write(secrets.token_hex(32)); os.chmod(arq_chave, 0o600)
chave = bytes.fromhex(open(arq_chave).read().strip())
def h(cnpj):                                            # HMAC-SHA256 com chave secreta, 16 hex = 64 bits
    return hmac.new(chave, str(cnpj).encode(), hashlib.sha256).hexdigest()[:16]

# 2. nós: id = hash, atributos categóricos, tipo de empresa (classe CNAE agregada), VA normalizado
nos = pd.read_parquet(pasta + 'mapa_ids_INTERNO.parquet')
nos['hash'] = nos['cnpj'].map(h)
assert nos['hash'].is_unique
l1f = pd.read_parquet(base + 's23_camada1_final.parquet', columns=['vizinho', 'cnae_final'])
cn = (l1f.dropna(subset=['cnae_final']).drop_duplicates('vizinho')
         .set_index('vizinho')['cnae_final'].astype(str).str.replace(r'\D', '', regex=True).str[:4])
cn4 = nos['cnpj'].map(cn); div = pd.to_numeric(cn4.str[:2], errors='coerce')
tipo = np.full(len(nos), 'comércio e serviços gerais', dtype=object)
tipo[((div >= 5) & (div <= 33)).values] = 'indústria'
tipo[cn4.eq('4530').values] = 'comércio de autopeças'
tipo[cn4.isna().values] = 'sem CNAE em nenhuma fonte'
tipo[nos['semente'].values] = 'semente'
nos['tipo'] = tipo

gia = pd.read_parquet(base + 's31_gia_vc.parquet')
col_no = next(c for c in gia.columns if c.lower() in ('cnpj', 'no', 'num_cnpj', 'num_cnpj_char'))
col_mes = next(c for c in gia.columns if 'mes' in c.lower() or 'ref' in c.lower())
col_vc = next(c for c in gia.columns if '_vc' in c.lower() or c.lower().endswith('vc') or 'adic' in c.lower())
g18 = gia[gia[col_mes].astype(str).str[:4].eq(str(ANO))]
va = g18.groupby([col_no, col_mes])[col_vc].last().groupby(level=0).sum().abs()
nos['va'] = nos['cnpj'].map(va)
nos['tem_va'] = nos['va'].notna()
nos['va_norm'] = (nos['va'] / nos['va'].sum()).fillna(0).round(10)   # fração do VA total da rede; a soma não é divulgada

# 3. arestas: pontas em hash, valor normalizado como fração do fluxo total da rede
ar = pd.read_parquet(pasta + f'arestas_{ANO}_induzido.parquet')
ar = ar[ar['origem'].isin(nos['cnpj']) & ar['destino'].isin(nos['cnpj'])]
ar['v'] = ar['v'].clip(lower=0)
ar_pub = pd.DataFrame({'origem': ar['origem'].map(h), 'destino': ar['destino'].map(h),
                       'meses': ar['meses'].astype(int), 'v_norm': (ar['v'] / ar['v'].sum()).round(12)})
ar_pub = ar_pub.groupby(['origem', 'destino'], as_index=False).agg(meses=('meses', 'max'), v_norm=('v_norm', 'sum'))

# 4. grafo público: simples, não direcionado; aresta = soma dos dois sentidos
und = ar_pub.assign(u=np.minimum(ar_pub['origem'], ar_pub['destino']), z=np.maximum(ar_pub['origem'], ar_pub['destino']))
und = und.groupby(['u', 'z'], as_index=False).agg(meses=('meses', 'max'), v_norm=('v_norm', 'sum'))
und = und[und['u'] != und['z']]
nos_pub = nos[['hash', 'semente', 'uf', 'direcao', 'tipo', 'tem_va', 'va_norm']].rename(columns={'hash': 'id'})
nos_pub['camada'] = np.where(nos_pub['semente'], 0, 1)
nos_pub = nos_pub.drop(columns='semente').sort_values('id').reset_index(drop=True)
pos = {k: i for i, k in enumerate(nos_pub['id'])}
gp = ig.Graph(n=len(nos_pub), edges=list(zip(und['u'].map(pos), und['z'].map(pos))), directed=False)
for c in ('id', 'camada', 'uf', 'direcao', 'tipo', 'va_norm'):
    gp.vs['name' if c == 'id' else c] = nos_pub[c].tolist()
gp.vs['tem_va'] = nos_pub['tem_va'].astype(int).tolist()
gp.es['meses'] = und['meses'].tolist(); gp.es['v_norm'] = und['v_norm'].tolist()
assert gp.is_simple()                                                  # sem laços nem arestas múltiplas

# 5. gravação da versão pública
gp.write_graphml(pub + f'rede_autopecas_sp_{ANO}.graphml')
nos_pub.to_csv(pub + f'nos_{ANO}.csv', index=False)
und.rename(columns={'u': 'origem', 'z': 'destino'}).to_csv(pub + f'arestas_{ANO}.csv', index=False)
print(f'versão pública gravada em {pub}: {gp.vcount():,} nós, {gp.ecount():,} arestas')
print('atributos de nó:', gp.vs.attributes(), '| de aresta:', gp.es.attributes())

# 6. verificação: as métricas do notebook são idênticas às do grafo original
g0 = ig.Graph.Read_GraphML(pasta + f'rede_autopecas_sp_{ANO}.graphml').simplify()
def metricas(g):
    d = np.array(g.degree()); cl = np.array(g.transitivity_local_undirected(mode='zero'))
    return pd.Series({'n': g.vcount(), 'm': g.ecount(), 'grau_medio': 2 * g.ecount() / g.vcount(),
        'C_media_excl': cl[d >= 2].mean(), 'C_media_incl': cl.mean(), 'transitividade': g.transitivity_undirected(),
        'dist_media': g.average_path_length(unconn=True), 'diametro': g.connected_components().giant().diameter(),
        'densidade': g.density(), 'componentes': len(g.connected_components()),
        'grau_max': d.max(), 'grau_seq_hash': hash(tuple(sorted(d.tolist())))})
cmp = pd.concat([metricas(g0).rename('original'), metricas(gp).rename('público')], axis=1)
print(cmp.to_string()); assert np.allclose(cmp['original'].astype(float), cmp['público'].astype(float))
print('\nOK: todas as métricas coincidem; a versão pública pode ser compartilhada')
