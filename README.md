# Projeto de Previsão de Rotatividade de Funcionários (Attrition)

> Projeto completo de Machine Learning para prever a rotatividade de funcionários utilizando técnicas avançadas de Data Science. O projeto envolveu análise exploratória, engenharia de variáveis, tratamento de desbalanceamento de classes (SMOTE), modelagem com múltiplos algoritmos (Regressão Logística, Random Forest, XGBoost) e criação de dashboard interativo para visualização de resultados e insights de negócio.

![alt text](display_attrition.gif)


![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Status](https://img.shields.io/badge/Status-Conclu%C3%ADdo-success)

## 📊 Sobre o Projeto

Este projeto tem como objetivo **prever a rotatividade de funcionários** (Attrition) em uma empresa, permitindo que o RH identifique proativamente os funcionários em risco de saída e implemente ações de retenção. O modelo alcançou **97.18% de Recall** e **AUC de 1.0**, identificando quase todos os funcionários que realmente saem da empresa.

### Principais Resultados

| Modelo | Acurácia | Recall (Yes) | F1-Score | AUC |
|--------|----------|--------------|----------|-----|
| **XGBoost** 🏆 | 99.55% | **97.18%** | 98.57% | **1.0000** |
| Random Forest | 99.55% | 97.18% | 98.57% | 0.9978 |
| Regressão Logística | 83.56% | 16.20% | 24.08% | 0.7143 |

## 🎯 Funcionalidades Principais

O projeto é composto por **duas partes principais**:

### 1. Notebook Jupyter (`ML_Attrition_Projeto.ipynb`)

Análise completa do projeto de Machine Learning, incluindo:

- **Análise Exploratória de Dados (EDA)**: Visualizações e estatísticas descritivas para entender os padrões de rotatividade
- **Feature Engineering**: Criação de variáveis derivadas (`AgeGroup`, `IncomeGroup`, `PromotionRate`, `CompanyExperienceRatio`, etc.)
- **Tratamento de Desbalanceamento**: Aplicação de SMOTE (Synthetic Minority Over-sampling Technique) para equilibrar as classes
- **Modelagem**: Treinamento e avaliação de 3 modelos (Regressão Logística, Random Forest, XGBoost)
- **Análise de Importância de Variáveis**: Identificação dos fatores mais relevantes para a rotatividade
- **Salvamento de Artefatos**: Exportação de modelos, scaler e dados para uso no dashboard

### 2. Dashboard Interativo (`app_attrition.py`)

Aplicação web desenvolvida com **Streamlit** para visualização e análise dos resultados:

- **Seleção de Modelo**: Escolha entre Regressão Logística, Random Forest ou XGBoost
- **Ajuste de Threshold**: Controle do limiar de decisão para balancear Recall e Precisão
- **Métricas de Performance**: KPIs principais (Acurácia, Precisão, Recall, F1-Score, AUC)
- **Curva ROC Interativa**: Visualização do trade-off entre TPR e FPR com marcador de threshold
- **Matriz de Confusão**: Heatmap interativo com interpretação de TN, FP, FN, TP
- **Importância de Variáveis**: Gráfico de barras mostrando os fatores mais relevantes
- **Download de Previsões**: Exportação de CSV com previsões detalhadas por funcionário

## 🛠️ Tecnologias Utilizadas

### Data Science & Machine Learning
- **Python 3.11**: Linguagem de programação principal
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Operações numéricas e arrays
- **Scikit-learn**: Algoritmos de ML, pré-processamento e métricas
- **XGBoost**: Algoritmo de gradient boosting otimizado
- **Imbalanced-learn**: Técnicas de reamostragem (SMOTE)

### Visualização & Dashboard
- **Streamlit**: Framework para criação de aplicações web analíticas
- **Plotly**: Gráficos interativos (Curva ROC, Matriz de Confusão, Importância de Variáveis)
- **Matplotlib & Seaborn**: Visualizações estáticas no notebook

### Outras Ferramentas
- **Joblib**: Serialização de modelos e artefatos
- **JSON**: Armazenamento de métricas e importâncias

## 📦 Como Executar

Para configurar e executar o projeto localmente, siga os passos abaixo:

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/ml-attrition-projeto
cd ml-attrition-projeto
```

### 2. Instale as dependências

Certifique-se de ter o Python 3.7+ instalado. Em seguida, instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

**Conteúdo do `requirements.txt`:**
```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
streamlit
plotly
matplotlib
seaborn
joblib
```

### 3. Execute o Notebook (Treinamento do Modelo)

Abra o Jupyter Notebook e execute todas as células:

```bash
jupyter notebook ML_Attrition_Projeto.ipynb
```

**Importante:** Execute **todas as células na ordem** para gerar os artefatos necessários (`artifacts_attrition/`).

### 4. Execute o Dashboard

Após executar o notebook, navegue até o diretório do projeto e execute:

```bash
streamlit run app_attrition.py
```

### 5. Acesse o Dashboard

Após a execução, o dashboard estará disponível no seu navegador:

```
http://localhost:8501
```

## 📂 Estrutura de Diretórios

```
/ml-attrition-projeto
├── ML_Attrition_Projeto.ipynb      # Notebook principal com análise e modelagem
├── app_attrition.py                # Dashboard Streamlit
├── artifacts_attrition/            # Artefatos gerados pelo notebook
│   ├── log_reg.joblib              # Modelo Regressão Logística
│   ├── random_forest.joblib        # Modelo Random Forest
│   ├── xgboost.joblib              # Modelo XGBoost
│   ├── scaler.joblib               # StandardScaler para pré-processamento
│   ├── X_test_raw.csv              # Dados de teste (sem scaling)
│   ├── y_test.csv                  # Labels de teste
│   ├── metrics_test.json           # Métricas de avaliação
│   └── importances.json            # Importância de variáveis
├── data/                           # Dados brutos (não incluídos no repositório)
│   └── raw/
│       └── rh_data.csv             # Dataset IBM HR Analytics
├── apresentacao_attrition/         # Slides da apresentação (opcional)
├── requirements.txt                # Dependências do projeto
└── README.md
```

## 🎓 Como Usar o Dashboard

Ao acessar o dashboard, você encontrará:

### Barra Lateral (Configurações)
- **Seleção de Modelo**: Escolha entre Regressão Logística, Random Forest ou XGBoost
- **Ajuste de Threshold**: Slider para controlar o limiar de decisão (0.0 a 1.0)
  - **Threshold baixo (0.3)**: Captura mais funcionários em risco (alto Recall), mas com mais falsos positivos
  - **Threshold alto (0.7)**: Menos falsos positivos (alta Precisão), mas pode perder funcionários em risco

### Área Principal
1. **Métricas de Performance**: 5 KPIs principais exibidos em cards coloridos
2. **Curva ROC**: Gráfico interativo mostrando o trade-off entre TPR e FPR, com marcador do threshold atual
3. **Matriz de Confusão**: Heatmap e tabela com interpretação de TN, FP, FN, TP
4. **Variáveis Mais Relevantes**: Gráfico de barras com as top N variáveis mais importantes (ajustável via slider)
5. **Download de Previsões**: Botão para baixar CSV com previsões detalhadas

## 📊 Insights de Negócio

Com base na análise de importância de variáveis, os principais fatores que influenciam a rotatividade são:

### Top 5 Variáveis Mais Importantes (Random Forest/XGBoost)

1. **MaritalStatus (Estado Civil)**: Funcionários solteiros têm maior taxa de rotatividade
2. **BusinessTravel (Viagens a Trabalho)**: Viagens frequentes aumentam o risco de saída
3. **MonthlyIncome (Renda Mensal)**: Remuneração é um fator crítico de retenção
4. **YearsAtCompany (Anos na Empresa)**: Funcionários com menos tempo são mais propensos a sair
5. **TotalWorkingYears (Anos de Experiência Total)**: Experiência total impacta a decisão de permanecer

### Recomendações para o RH

1. **Remuneração Competitiva**: Revisar a política salarial, especialmente para cargos com maior taxa de rotatividade
2. **Equilíbrio Trabalho-Vida**: Revisar a política de viagens e horas extras para reduzir o burnout
3. **Desenvolvimento de Carreira**: Criar trilhas de carreira mais claras e programas de promoção
4. **Engajamento Inicial**: Focar em programas de mentoria e retenção nos primeiros anos de trabalho
5. **Segmentação de Ações**: Priorizar JobRoles/Departamentos com risco médio previsto mais alto

## 👤 Autor

**Ianna Lise Castro de Paiva**

- Link apresentação: [Rota 3 - ML](https://www.canva.com/design/DAG1olLnH50/sc9IwuKUID3-9280vY5sHg/edit?utm_content=DAG1olLnH50&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- GitHub: [iannacastro](https://github.com/iannacastro)
- LinkedIn: [Ianna Castro](https://linkedin.com/in/ianna-castro)
- Email: iannacastrop@gmail.com

