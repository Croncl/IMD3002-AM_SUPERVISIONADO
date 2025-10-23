# projeto_ML_re

Este projeto contém experimentos de aprendizado supervisionado com foco em classificação de imagens de gatos e cachorros.  
Inclui extração de características com LBP e HOG, aplicação de PCA, e testes com algoritmos como KNN, Decision Tree e MLP.

## Estrutura

- `001_extrator_de_caractecisticas/`: scripts para extração de características
- `002_KNN_acuracia/`: avaliação com KNN
- `003_Dtree/`: avaliação com Decision Tree
- `005_rede_neural_mlp/`: testes com MLP e GridSearch

## Observações

- Arquivos `.csv` com bases geradas foram ignorados via `.gitignore` por serem grandes
- Resultados podem ser reproduzidos com os scripts disponíveis

---

## ✅ 2. Verificar se o `.gitignore` está funcionando

Dentro de `projeto_ML_re`, o `.gitignore` deve conter:

```gitignore
*.csv
bases_geradas/
.ipynb_checkpoints/
tempCodeRunnerFile*
imagens/
