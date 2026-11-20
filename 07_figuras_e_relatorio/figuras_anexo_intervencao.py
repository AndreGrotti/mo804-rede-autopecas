# Figuras ESQUEMÁTICAS do anexo sobre a análise de intervenção (dados sintéticos, só para ilustrar a ideia).
# Uso: python figuras_anexo_intervencao.py <pasta_saida>
# SVG escrito à mão (poucos KB), no estilo das figuras do relatório: azul = observado, tracejado cinza = contrafactual.
import os, sys, numpy as np
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
AZUL, VERM, CINZA, ESC = "#1f77b4", "#d62728", "#6b6b6b", "#333"
FONTE = "font-family:'Segoe UI',Arial,sans-serif"
T0 = 38                                    # mar/2020 = mês 38 a partir de jan/2017
rng = np.random.default_rng(2026)

def svg(nome, w, h, corpo, pre):
    css = (f".{pre}t{{{FONTE};font-size:11px;fill:{ESC}}}.{pre}s{{{FONTE};font-size:10px;fill:{CINZA}}}"
           f".{pre}a{{fill:none;stroke:{ESC};stroke-width:.8}}")
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" role="img">'
         f'<defs><marker id="{pre}m" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
         f'<path d="M0 0L10 5L0 10z" fill="{ESC}"/></marker></defs><style>{css}</style>{corpo}</svg>')
    open(os.path.join(OUT, nome + ".svg"), "w", encoding="utf-8").write(s)
    print(nome, round(len(s) / 1024, 1), "KB")

class Eixo:
    """Painel com escala linear: x em meses (ou outra unidade), y em unidades da métrica."""
    def __init__(s, pre, x0, y0, w, h, xlim, ylim): s.pre, s.x0, s.y0, s.w, s.h, s.xl, s.yl = pre, x0, y0, w, h, xlim, ylim
    def X(s, v): return s.x0 + (v - s.xl[0]) / (s.xl[1] - s.xl[0]) * s.w
    def Y(s, v): return s.y0 + s.h - (v - s.yl[0]) / (s.yl[1] - s.yl[0]) * s.h
    def linha(s, xs, ys, cor, lw=1.2, tr="", op=1):
        p = " ".join(f"{s.X(a):.0f},{s.Y(b):.0f}" for a, b in zip(xs, ys) if not np.isnan(b))
        d = f' stroke-dasharray="{tr}"' if tr else ""
        return f'<polyline points="{p}" fill="none" stroke="{cor}" stroke-width="{lw}"{d} opacity="{op}"/>'
    def faixa(s, a, b, cor=VERM, op=0.13):
        return f'<rect x="{s.X(a):.0f}" y="{s.y0}" width="{s.X(b) - s.X(a):.0f}" height="{s.h}" fill="{cor}" opacity="{op}"/>'
    def moldura(s): return f'<rect x="{s.x0}" y="{s.y0}" width="{s.w}" height="{s.h}" class="{s.pre}a"/>'
    def texto(s, x, y, t, cls="t", anc="start", px=False):
        xx, yy = (x, y) if px else (s.X(x), s.Y(y))
        return f'<text x="{xx:.0f}" y="{yy:.0f}" class="{s.pre}{cls}" text-anchor="{anc}">{t}</text>'
    def seta(s, x1, y1, x2, y2, cor=ESC):
        return (f'<line x1="{s.X(x1):.0f}" y1="{s.Y(y1):.0f}" x2="{s.X(x2):.0f}" y2="{s.Y(y2):.0f}" stroke="{cor}" '
                f'stroke-width="1" marker-end="url(#{s.pre}m)" marker-start="url(#{s.pre}m)"/>')
    def anos(s, ini=2017, fim=2025, passo=2):
        out = []
        for a in range(ini, fim + 1, passo):
            x = s.X((a - 2017) * 12 + 6)
            out.append(f'<text x="{x:.0f}" y="{s.y0 + s.h + 14}" class="{s.pre}s" text-anchor="middle">{a}</text>')
        return "".join(out)

# ---------------- A1: contrafactual × antes e depois
t = np.arange(108); saz = 4 * np.sin(2 * np.pi * (t - 2) / 12)
u = np.zeros(108)
for i in range(1, 108): u[i] = 0.5 * u[i - 1] + rng.normal(0, 1.4)
cf = 100 + 0.25 * t + saz
ob = cf + u + np.where(t >= T0 + 4, -8.0, 0.0)
ob[T0:T0 + 4] += np.array([-22, -33, -18, -9])
E = Eixo("a", 40, 12, 660, 190, (0, 107), (62, 136))
pre_m, pos_m = ob[:T0].mean(), ob[T0 + 4:].mean()
c = [E.faixa(T0, T0 + 4), E.moldura(),
     E.linha([0, T0 - 1], [pre_m] * 2, CINZA, 1.4, "1,3"), E.linha([T0 + 4, 107], [pos_m] * 2, CINZA, 1.4, "1,3"),
     E.texto(1, pre_m + 2.5, "média antes", "s"), E.texto(106, pos_m - 4.5, "média depois", "s", "end"),
     E.linha(t[T0 - 1:], cf[T0 - 1:], CINZA, 1.4, "5,3"), E.linha(t, ob, AZUL, 1.3),
     E.seta(72, cf[72] - 0.6, 72, ob[72] + 0.6),
     E.texto(72, ob[60:85].min() - 5, "efeito = observado − contrafactual", "t", "middle"),
     E.texto(T0 + 5, 67, "choque", "s"),
     E.texto(T0 + 6, cf[T0 + 6:T0 + 40].max() + 4, "contrafactual: tendência e meses do ano de antes, prolongados", "s"),
     E.texto(4, 132, "a média depois fica acima da média antes, mas a série ficou abaixo de onde estaria", "t"),
     E.anos()]
svg("FAi1_contrafactual", 720, 222, "".join(c), "a")

# ---------------- A2: pulso, degrau, rampa e degrau + rampa (sem ruído)
painel = [("Pulsos", "só nos meses do choque", lambda k: np.where((k >= 0) & (k < 4), [-12, -18, -10, -5][0], 0.0)),
          ("Degrau", "muda o nível e fica", lambda k: np.where(k >= 4, -8.0, 0.0)),
          ("Rampa", "muda o ritmo, que se acumula", lambda k: np.where(k >= 4, -0.35 * (k - 3), 0.0)),
          ("Degrau + rampa", "cai e se recupera", lambda k: np.where(k >= 4, -12 + 0.3 * (k - 3), 0.0))]
tt = np.arange(-24, 49); base = 50 + 0.15 * tt
c = []
for j, (tit, sub, f) in enumerate(painel):
    E = Eixo("b", 8 + j * 178, 40, 165, 120, (-24, 48), (28, 66))
    k = tt.copy(); ef = f(k).astype(float)
    if j == 0: ef = np.select([k == 0, k == 1, k == 2, k == 3], [-12, -18, -10, -5], 0.0)
    c += [E.faixa(0, 4), E.moldura(), E.linha(tt[23:], base[23:], CINZA, 1.3, "5,3"), E.linha(tt, base + ef, AZUL, 1.6),
          E.texto(E.x0 + 6, 18, tit, "t", px=True), E.texto(E.x0 + 6, 32, sub, "s", px=True)]
    if j == 3:
        x2 = 4 + 24; y_cf, y_ob = 50 + 0.15 * x2, 50 + 0.15 * x2 - 12 + 0.3 * (x2 - 3)
        c += [E.seta(x2, y_cf - 0.4, x2, y_ob + 0.4), E.texto(x2 - 2, (y_cf + y_ob) / 2 - 1.2, "efeito em 2 anos", "s", "end")]
c.append(f'<text x="360" y="178" class="bs" text-anchor="middle">tracejado: contrafactual; faixa vermelha: meses do choque</text>')
svg("FAi2_formas", 720, 184, "".join(c), "b")

# ---------------- A3: ruído persistente e p-valor por simulação
c = []
E = Eixo("c", 40, 34, 300, 150, (0, 60), (-7, 7))
c += [E.moldura(), E.linha([0, 60], [0, 0], CINZA, 1, "5,3"),
      E.texto(E.x0, 20, "Séries só de ruído, com a persistência estimada", "t", px=True)]
for r in range(4):
    v = np.zeros(61)
    for i in range(1, 61): v[i] = 0.75 * v[i - 1] + rng.normal(0, 1.0)
    c.append(E.linha(np.arange(61), v, ["#9aa0a6", "#b0b4b9", "#8a9097", "#c1c4c8"][r], 1.1))
c.append(E.texto(E.x0 + 150, E.y0 + 166, "sem choque, o desvio ainda se afasta e demora a voltar", "s", "middle", px=True))
z = np.abs(rng.normal(0, 1, 999)); obs = 2.6; pval = (1 + np.sum(z >= obs)) / 1000
H, ed = np.histogram(z, bins=np.arange(0, 4.01, 0.2))
F = Eixo("c", 400, 34, 300, 150, (0, 4), (0, H.max() * 1.1))
c += [F.moldura(), F.texto(F.x0, 20, "Efeito encontrado em 999 séries sem efeito", "t", px=True)]
for h_, a in zip(H, ed[:-1]):
    cor = VERM if a >= obs - 1e-9 else "#9aa0a6"
    c.append(f'<rect x="{F.X(a) + 0.5:.0f}" y="{F.Y(h_):.0f}" width="{F.X(0.2) - F.X(0) - 1:.0f}" height="{F.Y(0) - F.Y(h_):.0f}" fill="{cor}"/>')
c += [F.linha([obs, obs], [0, H.max() * 1.05], VERM, 1.6),
      F.texto(obs + 0.07, H.max() * 0.82, "observado", "t"),
      F.texto(obs + 0.07, H.max() * 0.68, f"p = {pval:.3f}".replace(".", ","), "t"),
      F.texto(obs + 0.07, H.max() * 0.56, "(fração em vermelho)", "s"),
      F.texto(F.x0 + 150, F.y0 + 166, "tamanho do efeito (desvios-padrão)", "s", "middle", px=True)]
svg("FAi3_simulacao", 720, 206, "".join(c), "c")

# ---------------- A4: varredura de datas (sup-Wald)
d = np.arange(15, 90)                       # abr/2018 (15) a jun/2024 (89): 75 datas
w = rng.chisquare(2, len(d)) * 0.8 + 18 * np.exp(-((d - 34) / 5.0) ** 2) + 4 * np.exp(-((d - 34) / 14.0) ** 2)
mesmo = (d >= T0) & (d <= T0 + 4)
w[mesmo] = w[np.argmax(d >= T0 + 5)]
E = Eixo("d", 40, 30, 660, 150, (13, 92), (0, 30))
c = [E.moldura(), E.texto(E.x0, 18, "Força da quebra (estatística de Wald) em cada data candidata", "t", px=True)]
lw = E.X(1) - E.X(0) - 1.2
for a, v, m in zip(d, w, mesmo):
    c.append(f'<rect x="{E.X(a) - lw / 2:.1f}" y="{E.Y(v):.0f}" width="{lw:.1f}" height="{E.Y(0) - E.Y(v):.0f}" fill="{"#c1c4c8" if m else AZUL}"/>')
c += [E.linha([13, 92], [15, 15], VERM, 1.4, "6,3"), E.texto(91, 16.2, "limite de 5% considerando todas as datas", "t", "end"),
      E.linha([13, 92], [6, 6], CINZA, 1.1, "2,3"), E.texto(91, 7.2, "limite de 5% para uma data só", "s", "end"),
      E.texto(34, w[d == 34][0] + 1.2, "pico", "t", "middle"),
      E.texto(T0 + 2, w[mesmo][0] + 1.2, "choque", "s", "middle")]
for a in range(2018, 2025):
    c.append(f'<text x="{E.X((a - 2017) * 12 + 6):.0f}" y="{E.y0 + E.h + 14}" class="ds" text-anchor="middle">{a}</text>')
svg("FAi4_varredura", 720, 198, "".join(c), "d")
