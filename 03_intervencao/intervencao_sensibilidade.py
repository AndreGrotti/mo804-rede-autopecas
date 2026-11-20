# Sensibilidade à janela da pandemia — rode DEPOIS da célula principal (usa S, estima, CONTR etc.)
# Cada cenário = (DATA em que começa o novo nível/tendência, trimestres do choque com pulso próprio)
CENARIOS = {
    "A: choque so 2020T2":         ("2020T2", ["2020T2"]),
    "B: choque 2020T1-T2 (atual)": ("2020T2", ["2020T1", "2020T2"]),
    "C: choque 2019T4-2020T2":     ("2019T4", ["2019T4", "2020T1", "2020T2"]),
    "D: novo nivel desde 2019T4":  ("2019T4", ["2020T1", "2020T2"]),
}

def modelo(idx, quebra):
    """Versão que lê TRANSICAO na hora da chamada (a da célula principal fixava a janela)."""
    chave = (rotulo(idx[0]), rotulo(idx[-1]), quebra, tuple(TRANSICAO))
    if chave not in _cache:
        X = regressores(idx, quebra, TRANSICAO)
        Xa = X.values.astype(float); n, k = Xa.shape
        A = np.linalg.solve(Xa.T @ Xa, Xa.T); M = np.eye(n) - Xa @ A
        i = np.arange(n); D = np.abs(i[:, None] - i[None, :]); Sm = 0.5 * (D == 1)
        G = np.empty(len(GRADE)); H = np.empty(len(GRADE))
        for j, ph in enumerate(GRADE):
            MRM = M @ (ph ** D / (1 - ph ** 2)) @ M
            H[j] = np.trace(MRM); G[j] = np.trace(Sm @ MRM) / H[j]
        _cache[chave] = dict(X=X, Xa=Xa, A=A, n=n, k=k, G=np.maximum.accumulate(G), H=H)
    return _cache[chave]

_cache.clear()
DATA0, TRANS0 = DATA, TRANSICAO
linhas = []
for nome, (DATA, TRANSICAO) in CENARIOS.items():
    for col in S.columns:
        q = estima(modelo(S.index, DATA), S[col], CONTR)[0].set_index("termo")
        linhas.append({"cenario": nome, "metrica": col,
                       "degrau": q.loc["degrau (nível)", "coef"], "p_degrau": q.loc["degrau (nível)", "p_sim"],
                       "ef2": q.loc["efeito em 2 anos", "coef"], "p_ef2": q.loc["efeito em 2 anos", "p_sim"]})
DATA, TRANSICAO = DATA0, TRANS0
_cache.clear()

sens = pd.DataFrame(linhas)
sens["degrau (p)"] = [f"{c:+.3g} ({p:.3f})" for c, p in zip(sens.degrau, sens.p_degrau)]
sens["2 anos (p)"] = [f"{c:+.3g} ({p:.3f})" for c, p in zip(sens.ef2, sens.p_ef2)]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 10)
for v in ("degrau (p)", "2 anos (p)"):
    print(f"\n{v.upper()} — coeficiente (p_sim), por cenário")
    print(sens.pivot(index="metrica", columns="cenario", values=v).reindex(S.columns).to_string())
