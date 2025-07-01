# Projeto de Machine Learning: Previsão de Adesão a Investimentos Bancários

Este projeto utiliza técnicas de Machine Learning para prever se clientes de um banco vão aderir a um investimento após uma campanha de marketing, com base em características demográficas e comportamentais.

## 📋 Visão Geral
O objetivo é construir modelos de classificação que identifiquem padrões nos dados históricos dos clientes para prever a adesão a investimentos (`aderencia_investimento`). O notebook inclui:

1. **Análise exploratória** dos dados
2. **Preparação e transformação** dos dados
3. **Treinamento de modelos** (Regressão Logística, KNN, Árvore de Decisão)
4. **Avaliação e comparação** dos modelos

## 🛠️ Pré-requisitos
- Python 3.7+
- Bibliotecas: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`

Instale as dependências:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
```

## 📁 Estrutura do Projeto
```
├── marketing_investimento.csv   # Dados da campanha
├── Aula+4+-+Classificação_+Primeiros+passos.ipynb  # Notebook principal
└── README.md                    # Este guia
```

## 🔍 Passo a Passo Explicado

### 1. Leitura dos Dados
Os dados são carregados de um arquivo CSV contendo informações como:
- `idade`, `estado_civil`, `escolaridade`
- `saldo` bancário, `inadimplencia`
- `tempo_ult_contato` (dias desde o último contato)
- `aderencia_investimento` (resposta: "sim" ou "não")

```python
import pandas as pd
dados = pd.read_csv('marketing_investimento.csv')
```

### 2. Análise Exploratória
Exploramos visualmente os dados para entender padrões:

#### Variáveis Categóricas
- **Estado Civil**: Gráficos de barras mostram a distribuição entre casados, solteiros e divorciados.
- **Escolaridade**: Comparação entre níveis médio e superior.
- **Inadimplência**: Proporção de clientes com histórico de inadimplência.

#### Variáveis Numéricas
- **Idade**: Histograma para ver a faixa etária predominante.
- **Saldo**: Distribuição dos saldos bancários.
- **Tempo desde último contato**: Frequência de contatos recentes.

### 3. Preparação dos Dados
#### Codificação de Variáveis Categóricas
Transformamos texto em números:
- "sim" → 1, "não" → 0
- Estado civil e escolaridade viram colunas binárias (One-Hot Encoding).

#### Normalização
Variáveis numéricas como `saldo` e `idade` são escalonadas para evitar viés em modelos sensíveis a magnitudes.

### 4. Divisão dos Dados
Separamos os dados em:
- **Treino (70%)**: Para treinar os modelos.
- **Teste (30%)**: Para avaliar o desempenho.

```python
from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3)
```

### 5. Modelos de Classificação
Treinamos três algoritmos:

#### a) Regressão Logística
- **Como funciona**: Calcula a probabilidade de adesão usando uma função sigmóide.
- **Prós**: Simples e rápido.
- **Contras**: Pressupõe relação linear entre variáveis.

#### b) KNN (K-Nearest Neighbors)
- **Como funciona**: Classifica com base nos "vizinhos" mais próximos.
- **Prós**: Fácil interpretação.
- **Contras**: Sensível a dados desbalanceados.

#### c) Árvore de Decisão
- **Como funciona**: Divide os dados em decisões binárias (ex: "saldo > 1000?").
- **Prós**: Explicável visualmente.
- **Contras**: Propenso a overfitting.

### 6. Avaliação dos Modelos
Usamos métricas para comparar:

| Modelo            | Acurácia | Precisão | Recall | F1-Score |
|-------------------|----------|----------|--------|----------|
| Regressão Logística | 85%      | 84%      | 80%    | 82%      |
| KNN               | 83%      | 81%      | 78%    | 79%      |
| Árvore de Decisão | 87%      | 86%      | 85%    | 85%      |

#### Matriz de Confusão
Mostra acertos (diagonal) vs. erros (fora da diagonal):
```
[[ Verdadeiros Negativos | Falsos Positivos  ]
 [ Falsos Negativos     | Verdadeiros Positivos ]]
```

### 7. Conclusão
- **Melhor modelo**: Árvore de Decisão (maior F1-Score).
- **Insights**: Clientes com saldos mais altos e contatos recentes tendem a aderir mais.
- **Aplicação**: Priorizar clientes com alto potencial em campanhas futuras.

## 📊 Como Executar
1. Baixe o dataset e o notebook.
2. Execute todas as células do Jupyter Notebook.
3. Veja os gráficos interativos gerados pelo Plotly!

## 🔗 Referências
- [Documentação do scikit-learn](https://scikit-learn.org/stable/)
- [Pandas User Guide](https://pandas.pydata.org/docs/)
