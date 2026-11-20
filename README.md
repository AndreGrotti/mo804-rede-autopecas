# A rede de autopeças e o choque de 2020 — código do projeto final

Disciplina MO804/MC908 — Redes Complexas (Prof. Ruben Interian), Instituto de Computação, Unicamp.
Aluno: André Luis Grotti Clemente. Novembro de 2026.

Este repositório reúne o código usado no relatório final. O trabalho acompanha a rede comercial em torno de
35 estabelecimentos paulistas de autopeças (as *sementes*) de 2017 a 2025, montada a partir de notas fiscais
eletrônicas, e estuda como ela atravessou o choque da pandemia de 2020.

## Dados

Os dados são notas fiscais eletrônicas protegidas por sigilo fiscal (art. 198 do Código Tributário Nacional) e
**não estão neste repositório**. Os scripts rodam no ambiente interno da Secretaria da Fazenda de São Paulo, onde
estão os arquivos de rede. Os arquivos de saída usados nas figuras e nos testes (CSVs em `resultados/`) trazem só
medidas agregadas, sem CNPJ e sem valores em reais; as sementes aparecem como S01–S35 ou x01–x35. A versão pública
da rede usa CNPJ transformado por HMAC-SHA256 com chave secreta e valores como frações do total
(`01_entrega_parcial/gera_versao_publica.py`); mesmo assim, por causa do risco de reidentificação pela estrutura,
os arquivos de rede ficam em acesso restrito.

Os caminhos nos scripts (`/home/candre/dados/rede/...`) apontam para o servidor interno e precisam ser ajustados
para rodar em outro ambiente.

## Organização

| Pasta | Conteúdo | Seção do relatório |
|---|---|---|
| `01_entrega_parcial/` | Notebook da Entrega Parcial (rede de 2018) e geração da versão pública da rede | 1.3 |
| `02_rede_e_metricas/` | Métricas mensais nas três versões da rede (principal, só SP e painel), regra de UF dos nós e rede de atividades (CNAE) | 2.2, 2.3, 3.1, 3.5 |
| `03_intervencao/` | Análise de intervenção (contrafactual, pulsos, degrau, rampa, erro AR(1), p por simulação, varredura sup-Wald), trimestral e mensal, e a rede só SP com as duas regras de UF | 2.4, 3.2, Anexos A e F |
| `04_relacoes_fronteira_nucleo/` | Retenção e continuidade das relações, vínculos fortes e fracos, fronteira de SP e núcleo persistente | 3.3, 3.5, 3.6 |
| `05_sementes/` | Sementes uma a uma: painel semente × mês, desvio da trajetória projetada, inferência por simulação com choques comuns, calibração e conferências | 3.4, Anexo C |
| `06_comparacoes_multiplas/` | Correções de Bonferroni e de Holm | Anexo D |
| `07_figuras_e_relatorio/` | Figuras em SVG e montagem do relatório em HTML | todas as figuras |

Os arquivos `celula_*.py` e `nucleo_*.py` são células de notebook Jupyter executadas em sequência no servidor;
cada uma diz no cabeçalho o que lê, o que grava e de qual célula anterior depende.

## Ordem de execução

1. **Extração e montagem da série** (servidor): `serie_extracao.py` extrai as notas uma vez por mês; cada período
   (mês ou trimestre) é montado com a mesma regra, nós = sementes e todos os que negociaram com elas no período,
   arestas = todas as vendas entre esses nós (subgrafo induzido). `serie_metricas.py` calcula as métricas
   trimestrais e as redes aleatórias de referência.
2. **Métricas mensais**: `02_rede_e_metricas/celula_M1_metricas_mensais.py`; regra de UF dos nós:
   `celula_U1_regra_de_UF.py` (UF como emitente; sem emissão observada, a da Receita; por fim, a das notas recebidas).
3. **Intervenção**: `03_intervencao/celula_M2_intervencao_mensal_definicoes.py` (definições e calibração em séries
   sintéticas) e `celula_M3_intervencao_mensal.py` (dados reais); `intervencao_soSP_regras.py` para a rede só SP;
   `intervencao_pandemia.py` é a versão trimestral; `wilcoxon_vs_intervencao.py` compara com o teste de Wilcoxon (Anexo F).
4. **Relações, fronteira e núcleo**: `04_relacoes_fronteira_nucleo/`.
5. **Sementes**: `05_sementes/sementes_painel_S3.py`, depois `sementes_contraste_S3.py`;
   `calibra_S3.py` e `confere_S3.py` reproduzem a calibração e a conferência dos pesos.
6. **Comparações múltiplas**: `python 06_comparacoes_multiplas/multiplas.py <pasta_dos_CSVs>`.
7. **Figuras e relatório**: scripts de `07_figuras_e_relatorio/` (cada um traz o uso no cabeçalho).

## Arquivos do servidor

Estes arquivos fazem parte do código, mas estão no servidor interno e são copiados para cá à parte:

- `02_rede_e_metricas/serie_extracao.py` — extração mensal das notas e montagem das redes por período;
- `02_rede_e_metricas/serie_metricas.py` — métricas trimestrais e redes aleatórias com os mesmos graus;
- `04_relacoes_fronteira_nucleo/projeto_redes_ruben_final.ipynb` — retenção anual, continuidade das relações
  (tabela de destino das relações de 2017) e decomposição entre períodos, com as saídas limpas antes da cópia.

## Requisitos

Python 3.11 ou superior com os pacotes de `requirements.txt`. A extração usa a conexão interna com o data
warehouse (Sybase IQ), que não faz parte deste repositório.
