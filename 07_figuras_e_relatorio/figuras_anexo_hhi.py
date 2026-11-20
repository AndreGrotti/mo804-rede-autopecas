# Figura ESQUEMÁTICA do anexo sobre o índice de Herfindahl: quatro carteiras de parceiros (dados artificiais).
# Uso: python figuras_anexo_hhi.py <pasta_saida>
import os, sys, numpy as np
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
FONTE = "font-family:'Segoe UI',Arial,sans-serif"
TONS = ["#1f4e79", "#2f6fa8", "#5b9bd5", "#9dc3e6"]

dez = np.array([52] + [48 / 9] * 9)
cart = [("Quatro parceiros iguais", np.array([25, 25, 25, 25.])),
        ("Quatro parceiros desiguais", np.array([40, 30, 20, 10.])),
        ("Dez parceiros, um dominante", dez),
        ("Os mesmos, sem os seis menores", dez[:4])]
fmt = lambda v, k=2: f"{v:.{k}f}".replace(".", ",")
X0, W, H, G, Y0 = 220, 300, 22, 40, 26
c = [f'<text x="{X0}" y="14" class="ht">participação de cada parceiro no valor</text>',
     f'<text x="{X0 + W + 14}" y="14" class="ht">HHI</text>',
     f'<text x="{X0 + W + 62}" y="14" class="ht">equivalente</text>',
     f'<text x="{X0 + W + 148}" y="14" class="ht">n × HHI</text>']
for i, (rot, s) in enumerate(cart):
    s = s / s.sum(); hhi = float(np.sum(s ** 2)); y = Y0 + i * G
    c.append(f'<text x="0" y="{y + 15}" class="hs">{rot}</text>')
    x = X0
    for j, v in enumerate(s):
        w = v * W
        c.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 1, 0.5):.1f}" height="{H}" fill="{TONS[min(j, 3)]}"/>')
        if w > 26: c.append(f'<text x="{x + w / 2:.1f}" y="{y + 15}" class="hb" text-anchor="middle">{round(100 * v)}%</text>')
        x += w
    c += [f'<text x="{X0 + W + 14}" y="{y + 15}" class="hs">{fmt(hhi)}</text>',
          f'<text x="{X0 + W + 62}" y="{y + 15}" class="hs">{fmt(1 / hhi, 1)} de {len(s)}</text>',
          f'<text x="{X0 + W + 148}" y="{y + 15}" class="hs">{fmt(len(s) * hhi, 1)}</text>']
css = (f".ht{{{FONTE};font-size:11px;fill:#333;font-weight:600}}.hs{{{FONTE};font-size:11px;fill:#333}}"
       f".hb{{{FONTE};font-size:10px;fill:#fff}}")
s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 {Y0 + 4 * G - 10}" width="100%" role="img">'
     f'<defs><marker id="hm" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
     f'<path d="M0 0L10 5L0 10z" fill="#333"/></marker></defs><style>{css}</style>{"".join(c)}</svg>')
open(os.path.join(OUT, "FHHI_carteiras.svg"), "w", encoding="utf-8").write(s)
print("FHHI_carteiras", round(len(s) / 1024, 1), "KB")
for rot, v in cart:
    v = v / v.sum(); h = np.sum(v ** 2); print(f"  {rot}: HHI {h:.3f}, equivalente {1 / h:.2f}, nxHHI {len(v) * h:.2f}")
