# Titanic Machine Learning Project

Este repositório contém um código de aprendizado de máquina para resolver o problema de previsão de sobrevivência dos passageiros do Titanic, utilizando o dataset fornecido pelo Kaggle. O objetivo é prever se um passageiro sobreviveu ou não ao naufrágio do Titanic com base em informações como idade, classe social, sexo, etc.

O código utiliza a regressão logística, um modelo de aprendizado supervisionado, para prever a sobrevivência dos passageiros do Titanic. A abordagem inclui os seguintes passos:

**Pré-processamento dos dados**: 
   - Carregamento dos dados.
   - Transformação das variáveis categóricas (como "Sex" e "Social_Class").
   - Preenchimento de valores ausentes.
   - Normalização das variáveis "Age" e "Fare".
   
## Como rodar o código

1. Baixe os dados do Titanic [aqui no Kaggle](https://www.kaggle.com/c/titanic/data).
2. Coloque o arquivo `train.csv` e `test.csv` na mesma pasta do código.
3. Importe os arquivos no código.
4. Execute o script `titanic.py` para treinar o modelo e gerar as previsões.

## Sobre a Predição

A predição presente no código está sendo feita através dos dados obtidos do "train_test_split()" para depois gerar a acurácia do modelo. Para gerar uma previsão com os dados de teste do Kaggle, apenas coloque os dados de "teste" dentro do metódo de predição e imprima a própria predição