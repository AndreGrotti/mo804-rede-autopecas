"""
Rede de grupos de CNAE em torno das sementes: 2017 contra 2025.

Cada estabelecimento das redes anuais (sementes + parceiros do ano, com todas as vendas do ano entre
eles) recebe o seu CNAE principal; as vendas são somadas por par de grupos de CNAE (3 dígitos, como o
29.4), do grupo do vendedor para o do comprador.

Duas etapas, nesta ordem:

  1. cadastro  Consulta o CNAE de cada estabelecimento e faz as conferências. Para quem é inscrito em SP,
               o CADESP, com o CNAE vigente em 31/12 de cada ano; para os demais, o cadastro da Receita
               (uma foto, sem histórico). Grava cnae_nos_INTERNO.parquet (tem CNPJ: fica no servidor).
               Leva alguns minutos.
  2. rede      Agrega as vendas por par de grupos, separa o esqueleto (as maiores arestas até somar 80%
               do valor de cada ano) e desenha as figuras. Não usa o banco.

  tabelas      Só se a etapa cadastro não encontrar alguma tabela: lista as tabelas do banco com nomes
               parecidos, para corrigir a configuração abaixo.

No Jupyter:  %run serie_cnae.py cadastro
             %run serie_cnae.py rede

Precisa dos arquivos da extração (camada1_mensal, arestas_mensais_<ano>, sementes_rotulos_INTERNO).
Os valores saem só como participação no total de cada ano.
"""
import importlib.util
import os
import random
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configuração
AQUI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('serie_extracao', os.path.join(AQUI, 'serie_extracao.py'))
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)            # reaproveita a conexão e as pastas da extração

OUT = ex.OUT
RES = OUT + 'resultados/'
ANOS = (2017, 2025)
COBERTURA = 0.80        # o esqueleto: as maiores arestas de cada ano até somar esta fração do valor
N_TOP = 15              # linhas da tabela dos maiores fluxos
GRUPO_FOCO = '29.4'     # fabricação de peças e acessórios para veículos automotores
ANO_FOTO_RFB = 2025     # ano da foto do cadastro da Receita (confira)
SEED = 2026
LOTE = 500              # CNPJs por consulta

# Nomes das tabelas (schema ods). Se a etapa cadastro não as encontrar, rode a etapa tabelas.
T_CAD_ESTAB = 'ods.tb_dsod_cadesp_estabelecimento'
T_CAD_CNAE = 'ods.tb_dsod_cadesp_estabelecimento_cnae'
T_CAD_DIM = 'ods.tb_dsod_cadesp_cnae'
T_RFB = 'ods.tb_dsod_bcadastros_cnpj_estabelecimento'

# Categorias das cores, pela divisão (2 primeiros dígitos). Ordem fixa: a cor segue a categoria.
CATEGORIAS = [
    ('Indústria automotiva (div. 29)', '#eb6834'),
    ('Outras indústrias (div. 10–33)', '#2a78d6'),
    ('Comércio de veículos e peças (div. 45)', '#1baf7a'),
    ('Comércio atacadista (div. 46)', '#eda100'),
    ('Comércio varejista (div. 47)', '#e87ba4'),
    ('Transporte e armazenagem (div. 49–53)', '#008300'),
    ('Outras atividades', '#6250d6'),
    ('Sem CNAE', '#898781'),
]
COR = dict(CATEGORIAS)
SEM = 's/ CNAE'


def categoria(grupo):
    if grupo == SEM:
        return 'Sem CNAE'
    d = int(grupo[:2])
    if d == 29:
        return CATEGORIAS[0][0]
    if 10 <= d <= 33:
        return CATEGORIAS[1][0]
    if d == 45:
        return CATEGORIAS[2][0]
    if d == 46:
        return CATEGORIAS[3][0]
    if d == 47:
        return CATEGORIAS[4][0]
    if 49 <= d <= 53:
        return CATEGORIAS[5][0]
    return CATEGORIAS[6][0]


def limpa_cnae(c):
    """Só dígitos, 7 posições (um zero à esquerda pode ter se perdido); vazio vira NaN."""
    s = pd.Series(c, dtype=object).fillna('').astype(str).str.replace(r'\D', '', regex=True)
    s = s.where(s.str.len().between(5, 7)).str.zfill(7)
    return s


def grupo_de(cnae):
    s = pd.Series(cnae, dtype=object)
    return s.where(s.isna(), s.str[:2] + '.' + s.str[2])


# ---------------------------------------------------------------- universo das redes anuais
def universo(ano, s, c1):
    """Nós: sementes + parceiros com operação no ano. Arestas: vendas do ano entre dois nós, somadas por
    par dirigido (vendedor -> comprador)."""
    noano = c1['mes'].between(ano * 100 + 1, ano * 100 + 12)
    nos = set(s['cnpj']) | set(c1.loc[noano, 'parceiro'])
    a = pd.read_parquet(OUT + f'arestas_mensais_{ano}.parquet', columns=['origem', 'destino', 'mes', 'v'])
    a = a[a['mes'].between(ano * 100 + 1, ano * 100 + 12) & a['origem'].isin(nos) & a['destino'].isin(nos)]
    E = a.groupby(['origem', 'destino'], as_index=False)['v'].sum()
    return nos, E


def entradas():
    s = pd.read_parquet(OUT + 'sementes_rotulos_INTERNO.parquet')
    c1 = pd.read_parquet(OUT + 'camada1_mensal.parquet')
    U = {ano: universo(ano, s, c1) for ano in ANOS}
    return s, c1, U


# ---------------------------------------------------------------- consultas ao banco
def em_lotes(cnpjs, sql_de_lista, rotulo):
    cnpjs = sorted(cnpjs)
    partes, t0 = [], time.time()
    nl = -(-len(cnpjs) // LOTE)
    for i in range(nl):
        lista = ','.join(f"'{c}'" for c in cnpjs[i * LOTE:(i + 1) * LOTE])
        partes.append(ex.le_sql(sql_de_lista(lista)))
        if (i + 1) % 10 == 0 or i + 1 == nl:
            print(f'      {rotulo}: lote {i + 1} de {nl}, {time.time() - t0:.0f} s', flush=True)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def confere_tabelas():
    """Lê uma linha de cada tabela e confere as colunas usadas; para com uma mensagem clara se faltar."""
    precisa = {T_CAD_ESTAB: ['num_cnpj_char', 'id_estabelecimento', 'cod_cnae'],
               T_CAD_CNAE: ['id_estabelecimento', 'cod_cnae', 'ind_cnae_primario', 'data_inicio_validade',
                            'data_fim_validade'],
               T_CAD_DIM: ['cod_cnae', 'descr_cnae'],
               T_RFB: ['num_cnpj', 'cod_cnae_fiscal', 'ind_excluido']}
    for t, cols in precisa.items():
        try:
            d = ex.le_sql(f'SELECT TOP 1 * FROM {t}', tentativas=1)
        except Exception as e:
            sys.exit(f'Tabela {t} não encontrada ({str(e)[:120]}). Rode a etapa "tabelas" e corrija o nome '
                     f'na configuração do script.')
        falta = [c for c in cols if c.lower() not in {x.lower() for x in d.columns}]
        if falta:
            sys.exit(f'Na tabela {t} faltam as colunas {falta}. Colunas encontradas: {list(d.columns)}')
    print('   tabelas e colunas conferidas', flush=True)


def consulta_cadesp(cnpjs):
    """Histórico do CNAE principal (com vigência) e o CNAE atual de cada estabelecimento inscrito em SP."""
    hist = em_lotes(cnpjs, lambda L: f"""
        SELECT e.num_cnpj_char AS cnpj, c.cod_cnae AS cnae, c.data_inicio_validade AS ini,
               c.data_fim_validade AS fim
        FROM {T_CAD_ESTAB} e JOIN {T_CAD_CNAE} c ON c.id_estabelecimento = e.id_estabelecimento
        WHERE c.ind_cnae_primario = 1 AND e.num_cnpj_char IN ({L})""", 'CADESP, histórico do CNAE')
    atual = em_lotes(cnpjs, lambda L: f"""
        SELECT num_cnpj_char AS cnpj, cod_cnae AS cnae
        FROM {T_CAD_ESTAB} WHERE num_cnpj_char IN ({L})""", 'CADESP, estabelecimentos')
    return hist, atual


def consulta_rfb(cnpjs):
    return em_lotes(cnpjs, lambda L: f"""
        SELECT num_cnpj AS cnpj, cod_cnae_fiscal AS cnae
        FROM {T_RFB} WHERE ind_excluido = 0 AND num_cnpj IN ({L})""", 'Receita')


def consulta_descricoes():
    return ex.le_sql(f'SELECT cod_cnae AS cnae, descr_cnae AS descr FROM {T_CAD_DIM}')


def etapa_tabelas():
    for p in ('cadesp_estabelecimento', 'cadesp_cnae', 'bcadastros_cnpj_estab'):
        d = ex.le_sql(f"""SELECT u.user_name AS dono, t.table_name AS tabela
                          FROM SYS.SYSTAB t JOIN SYS.SYSUSER u ON t.creator = u.user_id
                          WHERE t.table_name LIKE '%{p}%' ORDER BY 1, 2""")
        print(f'\n   tabelas com "{p}" no nome:')
        print(d.to_string(index=False) if len(d) else '      nenhuma')


# ---------------------------------------------------------------- CNAE vigente numa data
def vigente(hist, data):
    """CNAE principal vigente na data; se nenhum registro cobre a data (estabelecimento baixado antes),
    o último registro anterior; se só há registros posteriores, o primeiro deles."""
    data = pd.Timestamp(data)
    h = hist.dropna(subset=['cnae']).copy()
    ant = h[h['ini'] <= data]
    cobre = ant[ant['fim'].isna() | (ant['fim'] >= data)]
    r1 = cobre.sort_values('ini').groupby('cnpj').tail(1).assign(tipo='vigente')
    r2 = ant[~ant['cnpj'].isin(r1['cnpj'])].sort_values('ini').groupby('cnpj').tail(1).assign(tipo='anterior')
    r3 = (h[~h['cnpj'].isin(r1['cnpj']) & ~h['cnpj'].isin(r2['cnpj'])].sort_values('ini')
          .groupby('cnpj').head(1).assign(tipo='posterior'))
    return pd.concat([r1, r2, r3])[['cnpj', 'cnae', 'tipo']].set_index('cnpj')


def peso_dos_nos(E):
    """Valor que passa por cada estabelecimento (vendas + compras dentro da rede)."""
    return E.groupby('origem')['v'].sum().add(E.groupby('destino')['v'].sum(), fill_value=0)


# ---------------------------------------------------------------- etapa 1: cadastro
def etapa_cadastro():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    s, c1, U = entradas()
    todos = set().union(*[U[a][0] for a in ANOS])
    print(f'   estabelecimentos nas redes de {" e ".join(map(str, ANOS))}: '
          + ', '.join(f'{a}: {len(U[a][0]):,}' for a in ANOS) + f'; distintos: {len(todos):,}', flush=True)
    confere_tabelas()

    hist, atual = consulta_cadesp(todos)
    for d in (hist, atual):
        d['cnpj'] = d['cnpj'].astype(str).str.strip().str.zfill(14)
        d['cnae'] = limpa_cnae(d['cnae']).values
    hist['ini'] = pd.to_datetime(hist['ini'], errors='coerce')
    hist['fim'] = pd.to_datetime(hist['fim'], errors='coerce')
    rfb = consulta_rfb(todos)
    rfb['cnpj'] = rfb['cnpj'].astype(str).str.strip().str.zfill(14)
    rfb['cnae'] = limpa_cnae(rfb['cnae']).values
    rfb = rfb.dropna(subset=['cnae'])
    # O índice da Receita não é único: CNPJ com mais de um CNAE fica com o mais frequente.
    nd = rfb.groupby('cnpj')['cnae'].nunique()
    rfb1 = rfb.groupby('cnpj')['cnae'].agg(lambda x: x.value_counts().index[0])
    print(f'   CADESP: {hist["cnpj"].nunique():,} estabelecimentos com histórico de CNAE, '
          f'{atual["cnpj"].nunique():,} inscritos; Receita: {len(rfb1):,} '
          f'({int((nd > 1).sum()):,} com mais de um CNAE, resolvidos pelo mais frequente)', flush=True)
    atual1 = atual.dropna(subset=['cnae']).drop_duplicates('cnpj').set_index('cnpj')['cnae']
    inscritos = set(atual['cnpj'])

    linhas = []
    for ano in ANOS:
        nos = sorted(U[ano][0])
        v = vigente(hist, f'{ano}-12-31')
        d = pd.DataFrame({'cnpj': nos})
        d['ano'] = ano
        d['cnae_cadesp'] = d['cnpj'].map(v['cnae'])
        d['tipo_cadesp'] = d['cnpj'].map(v['tipo'])
        sem_hist = d['cnae_cadesp'].isna() & d['cnpj'].isin(inscritos)
        d.loc[sem_hist, 'cnae_cadesp'] = d.loc[sem_hist, 'cnpj'].map(atual1)
        d.loc[sem_hist & d['cnae_cadesp'].notna(), 'tipo_cadesp'] = 'atual (sem histórico)'
        d['cnae_rfb'] = d['cnpj'].map(rfb1)
        d['cnae'] = d['cnae_cadesp'].fillna(d['cnae_rfb'])
        d['fonte'] = np.where(d['cnae_cadesp'].notna(), 'CADESP',
                              np.where(d['cnae_rfb'].notna(), 'Receita', 'sem CNAE'))
        d['grupo'] = grupo_de(d['cnae']).fillna(SEM).values
        linhas.append(d)
    N = pd.concat(linhas, ignore_index=True)
    N.to_parquet(OUT + 'cnae_nos_INTERNO.parquet', index=False)

    desc = consulta_descricoes()
    desc['cnae'] = limpa_cnae(desc['cnae']).values
    desc = desc.dropna(subset=['cnae']).drop_duplicates('cnae', keep='last')
    desc.to_parquet(OUT + 'cnae_descricoes.parquet', index=False)

    conferencias(N, hist, U)
    print(f'\ncadastro concluído em {(time.time() - t0) / 60:.1f} min; cnae_nos_INTERNO.parquet em {OUT}')


def conferencias(N, hist, U):
    conf = []
    print('\nCONFERÊNCIAS')
    # (1) cobertura: estabelecimentos e valor por fonte do CNAE
    print('   (1) de onde veio o CNAE de cada estabelecimento (em número e no valor que passa por eles):')
    for ano in ANOS:
        d = N[N['ano'] == ano].set_index('cnpj')
        w = peso_dos_nos(U[ano][1]).reindex(d.index).fillna(0)
        for f in ('CADESP', 'Receita', 'sem CNAE'):
            m = d['fonte'] == f
            conf.append({'conferencia': 'fonte do CNAE', 'ano': ano, 'grupo': f, 'estabelecimentos': int(m.sum()),
                         'frac_estabelecimentos': m.mean(), 'frac_valor': w[m].sum() / w.sum()})
            print(f'      {ano} {f:9s}: {m.sum():6,} ({m.mean():6.1%}) | valor {w[m].sum() / w.sum():6.1%}')
        for t, m in d[d['fonte'] == 'CADESP'].groupby('tipo_cadesp').size().items():
            conf.append({'conferencia': 'registro do CADESP usado', 'ano': ano, 'grupo': t, 'estabelecimentos': int(m)})
        print('         CADESP, registro usado: ' + ', '.join(
            f'{t} {m:,}' for t, m in d[d['fonte'] == 'CADESP'].groupby('tipo_cadesp').size().items()))

    # (2) quantos estabelecimentos de SP mudam de grupo entre os dois anos (estimativa do erro da foto da
    # Receita para os de fora de SP no ano mais distante da foto)
    a0, a1 = ANOS
    v0, v1 = vigente(hist, f'{a0}-12-31'), vigente(hist, f'{a1}-12-31')
    base = N[(N['ano'] == a0) & (N['fonte'] == 'CADESP')]['cnpj']
    ok = base[base.isin(v0.index) & base.isin(v1.index)]
    c0, c1_ = v0.loc[ok, 'cnae'], v1.loc[ok, 'cnae']
    mg = grupo_de(c0).values != grupo_de(c1_).values
    ms = c0.values != c1_.values
    w = peso_dos_nos(U[a0][1]).reindex(ok).fillna(0).values
    print(f'   (2) estabelecimentos de SP da rede de {a0} com CNAE nas duas datas: {len(ok):,}. Mudaram de grupo '
          f'entre {a0} e {a1}: {mg.mean():.1%} ({(w * mg).sum() / w.sum():.1%} do valor de {a0}); '
          f'de subclasse: {ms.mean():.1%}')
    conf.append({'conferencia': f'SP: mudança de CNAE {a0}-{a1}', 'ano': a0, 'grupo': 'grupo',
                 'estabelecimentos': len(ok), 'frac_estabelecimentos': mg.mean(), 'frac_valor': (w * mg).sum() / w.sum()})
    conf.append({'conferencia': f'SP: mudança de CNAE {a0}-{a1}', 'ano': a0, 'grupo': 'subclasse',
                 'estabelecimentos': len(ok), 'frac_estabelecimentos': ms.mean(), 'frac_valor': (w * ms).sum() / w.sum()})

    # (3) as duas fontes concordam no ano da foto da Receita?
    vf = vigente(hist, f'{ANO_FOTO_RFB}-12-31')
    d = N.drop_duplicates('cnpj').set_index('cnpj')
    amb = d.index[d.index.isin(vf.index) & d['cnae_rfb'].notna()]
    cc, cr = vf.loc[amb, 'cnae'].values, d.loc[amb, 'cnae_rfb'].values
    ig, isub = grupo_de(cc).values == grupo_de(cr).values, cc == cr
    print(f'   (3) estabelecimentos com CNAE nas duas fontes ({ANO_FOTO_RFB}): {len(amb):,}. Mesmo grupo: '
          f'{ig.mean():.1%}; mesma subclasse: {isub.mean():.1%}')
    conf.append({'conferencia': f'CADESP x Receita em {ANO_FOTO_RFB}', 'ano': ANO_FOTO_RFB, 'grupo': 'grupo',
                 'estabelecimentos': len(amb), 'frac_estabelecimentos': ig.mean()})
    conf.append({'conferencia': f'CADESP x Receita em {ANO_FOTO_RFB}', 'ano': ANO_FOTO_RFB, 'grupo': 'subclasse',
                 'estabelecimentos': len(amb), 'frac_estabelecimentos': isub.mean()})
    pd.DataFrame(conf).to_csv(RES + 'CNAE_conferencias.csv', sep=';', index=False)


# ---------------------------------------------------------------- etapa 2: rede de grupos
def fluxos(ano, N, U):
    """Vendas somadas por par de grupos (vendedor -> comprador), com a participação no total do ano."""
    g = N[N['ano'] == ano].set_index('cnpj')['grupo']
    E = U[ano][1].copy()
    E['gv'], E['gc'] = E['origem'].map(g).values, E['destino'].map(g).values
    F = (E.groupby(['gv', 'gc']).agg(valor=('v', 'sum'), pares=('v', 'size'), vendedores=('origem', 'nunique'),
                                     compradores=('destino', 'nunique')).reset_index())
    F['participacao'] = F['valor'] / F['valor'].sum()
    F = F.sort_values('participacao', ascending=False).reset_index(drop=True)
    F['acumulado'] = F['participacao'].cumsum()
    # esqueleto: as maiores arestas até alcançar a cobertura (a que cruza o limite entra)
    F['esqueleto'] = (F['acumulado'] - F['participacao']) < COBERTURA
    F['ano'] = ano
    return F.drop(columns='valor')


def grupos(ano, F, N):
    vend = F.groupby('gv')['participacao'].sum()
    comp = F.groupby('gc')['participacao'].sum()
    inter = F[F['gv'] == F['gc']].set_index('gv')['participacao']
    n = N[N['ano'] == ano].groupby('grupo').size()
    G = pd.DataFrame({'vendas': vend, 'compras': comp, 'interno': inter, 'estabelecimentos': n}).fillna(0)
    G['passa'] = (G['vendas'] + G['compras']) / 2       # fração do valor que passa pelo grupo
    G.index.name = 'grupo'
    G['ano'] = ano
    G['categoria'] = [categoria(x) for x in G.index]
    return G.reset_index()


def exemplos(N, U, desc):
    """Para nomear cada grupo: a descrição da subclasse que mais movimenta valor dentro dele."""
    linhas = []
    for ano in ANOS:
        d = N[N['ano'] == ano].set_index('cnpj')
        w = peso_dos_nos(U[ano][1]).reindex(d.index).fillna(0)
        linhas.append(pd.DataFrame({'grupo': d['grupo'], 'cnae': d['cnae'], 'w': w.values}))
    L = pd.concat(linhas).dropna(subset=['cnae']).groupby(['grupo', 'cnae'])['w'].sum().reset_index()
    L = L.sort_values('w').groupby('grupo').tail(1)
    dd = desc.set_index('cnae')['descr']
    return dict(zip(L['grupo'], L['cnae'].map(dd).fillna('').str.strip().str.capitalize()))


def etapa_rede():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    s, c1, U = entradas()
    N = pd.read_parquet(OUT + 'cnae_nos_INTERNO.parquet')
    desc = pd.read_parquet(OUT + 'cnae_descricoes.parquet')
    ex_g = exemplos(N, U, desc)

    F = {a: fluxos(a, N, U) for a in ANOS}
    G = {a: grupos(a, F[a], N) for a in ANOS}
    for a in ANOS:
        f = F[a]
        e = f[f['esqueleto']]
        foco = (f['gv'] == GRUPO_FOCO) | (f['gc'] == GRUPO_FOCO)
        print(f'   {a}: {f["gv"].nunique():,} grupos vendedores, {len(f):,} pares de grupos; esqueleto de '
              f'{len(e):,} pares ({e["participacao"].sum():.1%} do valor), {int((e["gv"] == e["gc"]).sum())} '
              f'deles dentro do próprio grupo; arestas que tocam o {GRUPO_FOCO}: {f.loc[foco, "participacao"].sum():.1%} '
              f'do valor; sem CNAE em alguma ponta: {f.loc[(f["gv"] == SEM) | (f["gc"] == SEM), "participacao"].sum():.1%}',
              flush=True)

    T = pd.concat([F[a] for a in ANOS], ignore_index=True)
    T['categoria_vendedor'] = [categoria(x) for x in T['gv']]
    T['categoria_comprador'] = [categoria(x) for x in T['gc']]
    T.rename(columns={'gv': 'grupo_vendedor', 'gc': 'grupo_comprador'}).to_csv(
        RES + 'CNAE_fluxos.csv', sep=';', index=False)
    GG = pd.concat([G[a] for a in ANOS], ignore_index=True)
    GG['exemplo_de_atividade'] = GG['grupo'].map(ex_g).fillna('')
    GG.to_csv(RES + 'CNAE_grupos.csv', sep=';', index=False)

    # Tabela dos maiores fluxos: ordenados pela média das participações nos dois anos.
    a0, a1 = ANOS
    P = (F[a0].set_index(['gv', 'gc'])['participacao'].rename(f'part_{a0}').to_frame()
         .join(F[a1].set_index(['gv', 'gc'])['participacao'].rename(f'part_{a1}'), how='outer').fillna(0))
    P['variacao_pp'] = 100 * (P[f'part_{a1}'] - P[f'part_{a0}'])
    for a in ANOS:
        P[f'posicao_{a}'] = P[f'part_{a}'].rank(ascending=False, method='min').astype(int)
    P['media'] = P[[f'part_{a0}', f'part_{a1}']].mean(axis=1)
    top = P.sort_values('media', ascending=False).head(N_TOP).drop(columns='media').reset_index()
    top.insert(2, 'vendedor_exemplo', top['gv'].map(ex_g).fillna(''))
    top.insert(3, 'comprador_exemplo', top['gc'].map(ex_g).fillna(''))
    top = top.rename(columns={'gv': 'grupo_vendedor', 'gc': 'grupo_comprador'})
    top.to_csv(RES + 'CNAE_top15.csv', sep=';', index=False)
    print(f'\nOS {N_TOP} MAIORES FLUXOS (participação no valor de cada ano):')
    vis = top.copy()
    for a in ANOS:
        vis[f'part_{a}'] = (100 * vis[f'part_{a}']).round(1)
    vis['variacao_pp'] = vis['variacao_pp'].round(1)
    vis['vendedor_exemplo'] = vis['vendedor_exemplo'].str[:38]
    vis['comprador_exemplo'] = vis['comprador_exemplo'].str[:38]
    print(vis.to_string(index=False))

    figuras(F, G)
    print(f'\nrede de CNAEs concluída em {(time.time() - t0) / 60:.1f} min; CNAE_*.csv e fig_CNAE_*.png em {RES}')


# ---------------------------------------------------------------- figuras
WMAX = 9.0        # espessura da maior aresta (pontos)
AMAX = 2600.0     # área do maior nó (pontos²)
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#898781', '#fcfcfb'


def layout(F):
    """Posições calculadas uma vez, sobre a união dos esqueletos dos dois anos."""
    import igraph as ig
    random.seed(SEED)
    K = pd.concat([F[a][F[a]['esqueleto']] for a in ANOS])
    nos = sorted(set(K['gv']) | set(K['gc']))
    idx = {g: i for i, g in enumerate(nos)}
    E = K[K['gv'] != K['gc']].groupby(['gv', 'gc'])['participacao'].max().reset_index()
    E['u'] = [min(idx[a], idx[b]) for a, b in zip(E['gv'], E['gc'])]
    E['w'] = [max(idx[a], idx[b]) for a, b in zip(E['gv'], E['gc'])]
    E = E.groupby(['u', 'w'])['participacao'].sum().reset_index()
    g = ig.Graph(n=len(nos), edges=list(zip(E['u'], E['w'])))
    pesos = np.sqrt(E['participacao'].values / E['participacao'].max()) if len(E) else []
    xy = np.array(g.layout_fruchterman_reingold(weights=list(pesos), niter=3000).coords) if len(E) else \
        np.random.default_rng(SEED).random((len(nos), 2))
    xy = (xy - xy.min(0)) / np.where(np.ptp(xy, 0) > 0, np.ptp(xy, 0), 1)
    return {g_: xy[i] for g_, i in idx.items()}


def _raio_pts(area):
    return np.sqrt(area / np.pi)


def _nos(ax, pos, Gi, amax, alpha=1.0, anel=True, rotulos=True):
    """Desenha os nós; devolve o raio em pontos de cada um, para as setas pararem na borda."""
    import matplotlib.patheffects as pe
    raio = {}
    for g_, (x, y) in pos.items():
        r = Gi.loc[g_] if g_ in Gi.index else None
        if r is None or r['passa'] <= 0:
            ax.scatter([x], [y], s=30, facecolor='none', edgecolor=MUTED, linewidth=0.8, zorder=3)
            raio[g_] = _raio_pts(30)
            continue
        area = max(30.0, AMAX * r['passa'] / amax)
        lw = 0.5 + (6.0 * np.sqrt(r['interno'] / amax) if anel else 0)
        ax.scatter([x], [y], s=area, color=COR[categoria(g_)], alpha=alpha, edgecolor=INK2 if anel else SURF,
                   linewidth=lw, zorder=3)
        raio[g_] = _raio_pts(area) + lw / 2
    if rotulos:
        lim = sorted([Gi.loc[g_, 'passa'] for g_ in pos if g_ in Gi.index], reverse=True)
        corte = lim[min(24, len(lim) - 1)] if lim else 0
        for g_, (x, y) in pos.items():
            if g_ in Gi.index and (Gi.loc[g_, 'passa'] >= corte or g_ == GRUPO_FOCO):
                ax.text(x, y, g_, fontsize=7.5, ha='center', va='center', color=INK, zorder=6,
                        fontweight='bold' if g_ == GRUPO_FOCO else 'normal',
                        path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
    return raio


def _seta(ax, p0, p1, w, cor, alpha, r0, r1, z):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=6 + 1.5 * w, linewidth=w,
                                 color=cor, alpha=alpha, connectionstyle='arc3,rad=0.18',
                                 shrinkA=r0, shrinkB=r1 + 1, zorder=z))


def _legenda(ax, presentes, smax, rotulo_largura):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', linestyle='', markersize=8, markerfacecolor=c, markeredgecolor='none', label=n)
         for n, c in CATEGORIAS if n in presentes]
    for p in [x for x in (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20) if x <= smax][-3:]:
        h.append(Line2D([], [], color=INK2, linewidth=WMAX * np.sqrt(p / smax), label=rotulo_largura(p)))
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7.5,
              frameon=False, handlelength=3)


def painel_ano(ax, ano, F, G, pos, smax, amax):
    Gi = G[ano].set_index('grupo')
    raio = _nos(ax, pos, Gi, amax)
    e = F[ano][F[ano]['esqueleto'] & (F[ano]['gv'] != F[ano]['gc'])]
    for _, r in e.sort_values('participacao').iterrows():
        foco = GRUPO_FOCO in (r['gv'], r['gc'])
        w = WMAX * np.sqrt(r['participacao'] / smax)
        _seta(ax, pos[r['gv']], pos[r['gc']], w, COR[categoria(r['gv'])], 0.9 if foco else 0.45,
              raio[r['gv']], raio[r['gc']], 5 if foco else 2)
    n_int = int((F[ano]['esqueleto'] & (F[ano]['gv'] == F[ano]['gc'])).sum())
    ax.set_title(f'{ano}: {len(e)} arestas entre grupos e {n_int} vendas dentro do grupo (anel), '
                 f'{F[ano].loc[F[ano]["esqueleto"], "participacao"].sum():.0%} do valor', fontsize=10, color=INK)
    _legenda(ax, set(Gi['categoria']), smax, lambda p: f'{100 * p:g}% do valor do ano'.replace('.', ','))


def painel_mudanca(ax, F, G, pos, amax):
    a0, a1 = ANOS
    P = (F[a0].set_index(['gv', 'gc'])['participacao'].rename('p0').to_frame()
         .join(F[a1].set_index(['gv', 'gc'])['participacao'].rename('p1'), how='outer').fillna(0))
    esq = set(F[a0].loc[F[a0]['esqueleto'], ['gv', 'gc']].itertuples(index=False, name=None)) | \
        set(F[a1].loc[F[a1]['esqueleto'], ['gv', 'gc']].itertuples(index=False, name=None))
    P = P[P.index.isin(list(esq))].reset_index()
    P = P[P['gv'] != P['gc']]
    P['d'] = P['p1'] - P['p0']
    dmax = P['d'].abs().max() if len(P) else 1
    Gm = pd.concat([G[a0], G[a1]]).groupby('grupo')[['passa', 'interno']].max()
    raio = _nos(ax, pos, Gm, amax, alpha=0.35, anel=False)
    for _, r in P.sort_values('d', key=np.abs).iterrows():
        if abs(r['d']) < 0.001:
            cor, w, al = MUTED, 0.6, 0.5
        else:
            cor, w, al = ('#2a78d6' if r['d'] > 0 else '#e34948'), WMAX * np.sqrt(abs(r['d']) / dmax), 0.8
        _seta(ax, pos[r['gv']], pos[r['gc']], w, cor, al, raio[r['gv']], raio[r['gc']], 4)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color='#2a78d6', linewidth=3, label=f'participação subiu de {a0} para {a1}'),
         Line2D([], [], color='#e34948', linewidth=3, label='participação caiu'),
         Line2D([], [], color=MUTED, linewidth=0.8, label='mudou menos de 0,1 ponto')]
    for p in [x for x in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05) if x <= dmax][-3:]:
        if True:
            h.append(Line2D([], [], color=INK2, linewidth=WMAX * np.sqrt(p / dmax),
                            label=f'{100 * p:.1f} ponto percentual'.replace('.', ',')))
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7.5,
              frameon=False, handlelength=3)
    ax.set_title(f'Mudança de {a0} para {a1} nas arestas dos dois esqueletos (escala própria)', fontsize=10,
                 color=INK)


def figuras(F, G):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pos = layout(F)
    smax = max(F[a].loc[F[a]['esqueleto'] & (F[a]['gv'] != F[a]['gc']), 'participacao'].max() for a in ANOS)
    amax = max(G[a]['passa'].max() for a in ANOS)
    nota = (f'Nós: grupos de CNAE (3 dígitos); tamanho = fração do valor que passa pelo grupo; anel = vendas '
            f'dentro do próprio grupo. Arestas: do grupo vendedor para o comprador, na cor do vendedor; '
            f'espessura em raiz quadrada da participação no valor do ano; destacadas as que tocam o {GRUPO_FOCO}. '
            f'Esqueleto: as maiores arestas até {COBERTURA:.0%} do valor de cada ano.')

    def prepara(ax):
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.06, 1.06)
        ax.set_aspect('equal')
        ax.axis('off')

    for a in ANOS:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=SURF)
        prepara(ax)
        painel_ano(ax, a, F, G, pos, smax, amax)
        fig.text(0.02, 0.01, nota, fontsize=7, color=INK2, wrap=True)
        fig.savefig(RES + f'fig_CNAE_{a}.png', dpi=160, facecolor=SURF, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=SURF)
    prepara(ax)
    painel_mudanca(ax, F, G, pos, amax)
    fig.savefig(RES + 'fig_CNAE_mudanca.png', dpi=160, facecolor=SURF, bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(27, 9.6), facecolor=SURF)
    for ax in axs:
        prepara(ax)
    painel_ano(axs[0], ANOS[0], F, G, pos, smax, amax)
    painel_ano(axs[1], ANOS[1], F, G, pos, smax, amax)
    painel_mudanca(axs[2], F, G, pos, amax)
    fig.text(0.01, 0.005, nota, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(RES + 'fig_CNAE_comparacao.png', dpi=130, facecolor=SURF, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)


if __name__ == '__main__':
    etapa = sys.argv[1] if len(sys.argv) > 1 else ''
    if etapa == 'cadastro':
        etapa_cadastro()
    elif etapa == 'rede':
        etapa_rede()
    elif etapa == 'tabelas':
        etapa_tabelas()
    else:
        print(__doc__)
