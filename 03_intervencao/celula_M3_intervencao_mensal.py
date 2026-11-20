# Célula M3 — análise de intervenção MENSAL nos dados reais (principal, só SP, painel); usa as definições da M2
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES = "/home/candre/dados/rede/disciplina/serie/resultados"
VERSOES = ["completa", "so_SP", "painel_todos"]
resumo, foco = [], []
for v in VERSOES:
    S = le_mensal(CAMINHO_M, v)
    print("\n" + "#" * 100 + f"\nVERSÃO {v}: {rotulo(S.index[0])}–{rotulo(S.index[-1])}, {len(S)} meses")
    quadros, cfs, varr, crits = [], [], [], {}
    for col in S.columns:
        q, cf, tab, crit, p_sup = relata(S[col], col)
        quadros.append(q.assign(metrica=col)); cfs.append(cf.assign(metrica=col))
        varr.append(tab.assign(metrica=col, limite_5pct=crit, p_alguma_quebra=p_sup)); crits[col] = crit
        P = modelo(S.index, DATA); ya = S[col].values; e = ya - P["Xa"] @ (P["A"] @ ya)
        i = int(tab.wald.idxmax()); qq = q.set_index("termo")
        resumo.append(dict(versao=v, metrica=col,
                           degrau=qq.loc["degrau (nível)", "coef"], p_degrau=qq.loc["degrau (nível)", "p_sim"],
                           efeito_2a=qq.loc["efeito em 2 anos", "coef"], p_2a=qq.loc["efeito em 2 anos", "p_sim"],
                           quebra_max=tab.data[i], wald_max=tab.wald[i], limite=crit, p_sup=p_sup,
                           lb12=p_lb(e, 12)))
        if col in ("transitividade", "grau_medio", "assortatividade"):
            j = tab[(tab.data >= "2019-06") & (tab.data <= "2020-08")]
            foco.append(j.assign(versao=v, metrica=col, limite=crit))
    coef, contraf, varred = pd.concat(quadros), pd.concat(cfs), pd.concat(varr)
    base = os.path.join(RES, f"intervencao_mensal_{v}")
    coef.to_csv(base + "_coef.csv", sep=";", index=False)
    contraf.assign(periodo=[rotulo(p) for p in contraf.index]).to_csv(base + "_contrafactual.csv", sep=";", index=False)
    varred.to_csv(base + "_varredura.csv", sep=";", index=False)

    n = len(S.columns)
    fig, axs = plt.subplots(n, 2, figsize=(13, 2.4 * n), gridspec_kw={"width_ratios": [1.4, 1]}, squeeze=False)
    for (ax, ax2), col in zip(axs, S.columns):
        c = contraf[contraf.metrica == col]; x = [p.to_timestamp() for p in c.index]
        ax.plot(x, c["observado"], "o-", ms=2, lw=0.6, color="0.3", label="observado")
        ax.plot(x, c["ajustado"], color="tab:blue", lw=1, label="ajustado")
        ax.plot(x, c["sem_intervencao"], "--", color="tab:orange", label=f"sem intervenção (até +{HORIZONTE} meses)")
        ax.axvspan(periodo(PULSOS[0]).to_timestamp(), (periodo(PULSOS[-1]) + 1).to_timestamp(), color="tab:red", alpha=0.12)
        ax.set_title(f"{v}: {col}", fontsize=9); ax.tick_params(labelsize=7)
        t = varred[varred.metrica == col].reset_index(drop=True)
        ax2.bar(range(len(t)), t.wald, color=["tab:red" if d == DATA else "0.6" for d in t.data])
        ax2.axhline(crits[col], ls="--", color="k", lw=0.8)
        ax2.set_xticks(range(0, len(t), 12)); ax2.set_xticklabels(list(t.data.iloc[::12]), fontsize=7)
        ax2.set_title("força da quebra em cada mês (tracejado: limite 5%)", fontsize=8); ax2.tick_params(labelsize=7)
    axs[0][0].legend(fontsize=7); plt.tight_layout()
    fig.savefig(base + ".png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nGravado: {base}_coef.csv, _contrafactual.csv, _varredura.csv e .png")

R = pd.DataFrame(resumo)
R.to_csv(os.path.join(RES, "intervencao_mensal_resumo.csv"), sep=";", index=False)
pd.set_option("display.width", 220)
print("\n" + "=" * 100 + "\nRESUMO (p por simulação; p_sup = alguma quebra na varredura; lb12 = Ljung-Box defasagem 12)")
print(R.round(4).to_string(index=False))
F = pd.concat(foco)
print("\nFOCO: força da quebra mês a mês, jun/2019–ago/2020 (Wald; compare com o limite)")
print(F.pivot_table(index="data", columns=["versao", "metrica"], values="wald").round(1).to_string())
print(F.groupby(["versao", "metrica"]).limite.first().round(1).to_string())
