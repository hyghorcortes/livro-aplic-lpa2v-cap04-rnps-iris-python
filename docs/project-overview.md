# Visão geral do projeto

Este projeto representa um exemplo didático de classificação supervisionada usando a base Iris e quatro variantes computacionais:

1. `NN`: rede neural didática convencional;
2. `CNAPa`: rede/célula neural artificial paraconsistente;
3. `CNAPap`: variante paraconsistente aplicada ao problema Iris;
4. `CNAPCa`: composição paraconsistente de evidências das quatro variáveis de entrada.

## Núcleo computacional

Todas as variantes partem da mesma base e seguem uma estrutura comparável:

- carregamento da base Iris;
- partição estratificada manual;
- normalização;
- treinamento ou construção do classificador;
- avaliação em treino, validação e teste;
- geração de tabelas e figuras.

## Arquivos mais importantes

- `run_all_examples.py`
- `NN/main_iris_nn_didatico.py`
- `CNAPa/main_iris_cnapa_didatico.py`
- `CNAPap/main_iris_cnapap_didatico_v2_autocontido.py`
- `CNAPCa/main_iris_cnapca_didatico_autocontido.py`
