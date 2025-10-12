import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Attrition ML – Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================================================
ART = Path("artifacts_attrition")
MODEL_DISPLAY_NAMES = {
    "log_reg": "Regressão Logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost"
}

# ============================================================================
# TÍTULO E INTRODUÇÃO
# ============================================================================
st.title("🎯 Attrition ML – Dashboard de Resultados")
st.markdown("""
Este dashboard apresenta os resultados do modelo de Machine Learning para prever a **rotatividade de funcionários** (*Attrition*).
Use os controles na barra lateral para explorar diferentes modelos e ajustar o limiar de decisão.
""")

# ============================================================================
# SIDEBAR: SELEÇÃO DE MODELO E THRESHOLD
# ============================================================================
st.sidebar.header("⚙️ Configurações")

# Verificar quais modelos estão disponíveis
model_options = []
if (ART / "log_reg.joblib").exists():
    model_options.append("log_reg")
if (ART / "random_forest.joblib").exists():
    model_options.append("random_forest")
if (ART / "xgboost.joblib").exists():
    model_options.append("xgboost")

if not model_options:
    st.error("❌ Nenhum modelo encontrado em `artifacts_attrition/`. Execute o notebook para gerar os artefatos.")
    st.stop()

# Seleção do modelo
model_name = st.sidebar.selectbox(
    "Selecione o Modelo",
    model_options,
    format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x),
    index=len(model_options) - 1
)

# Ajuste do threshold
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚️ Limiar de Decisão")
threshold = st.sidebar.slider(
    "Attrition = 1 se probabilidade ≥ limiar",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Ajuste o limiar para controlar o trade-off entre Recall (capturar mais saídas) e Precisão (evitar alarmes falsos)."
)

st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Dica:** 
- **Threshold baixo (ex: 0.3):** Captura mais funcionários em risco (alto Recall), mas com mais falsos positivos.
- **Threshold alto (ex: 0.7):** Menos falsos positivos (alta Precisão), mas pode perder funcionários em risco.
""")

# ============================================================================
# CARREGAR ARTEFATOS
# ============================================================================
try:
    # Carregar métricas de referência (do notebook)
    with open(ART / "metrics_test.json", "r", encoding="utf-8") as f:
        metrics_all = json.load(f)
    metrics_ref = metrics_all.get(model_name, {})

    # Carregar importâncias de variáveis
    importances = {}
    if (ART / "importances.json").exists():
        with open(ART / "importances.json", "r", encoding="utf-8") as f:
            importances = json.load(f)

    # Carregar pré-processador (se existir)
    preproc = None
    if (ART / "preproc.joblib").exists():
        preproc = joblib.load(ART / "preproc.joblib")

    # Carregar modelo
    model = joblib.load(ART / f"{model_name}.joblib")

    # Carregar dados de teste
    y_test = pd.read_csv(ART / "y_test.csv")["y_test"].values
    X_test_raw = pd.read_csv(ART / "X_test_raw.csv")

    # Aplicar pré-processamento (se existir)
    if preproc is not None:
        X_test_trans = preproc.transform(X_test_raw)
    else:
        X_test_trans = X_test_raw.values

except FileNotFoundError as e:
    st.error(f"❌ Arquivo não encontrado: {e.filename}. Certifique-se de que o notebook gerou todos os artefatos.")
    st.stop()
except Exception as e:
    st.error(f"❌ Erro ao carregar artefatos: {str(e)}")
    st.stop()

# ============================================================================
# FUNÇÃO AUXILIAR: OBTER PROBABILIDADES
# ============================================================================
def proba_safe(mdl, X):
    """
    Obtém as probabilidades de forma segura, tratando diferentes tipos de modelos.
    
    Args:
        mdl: Modelo treinado (sklearn ou xgboost)
        X: Features de entrada
    
    Returns:
        Array de probabilidades para a classe positiva (Attrition = 1)
    """
    if hasattr(mdl, "predict_proba"):
        return mdl.predict_proba(X)[:, 1]
    elif hasattr(mdl, "decision_function"):
        from scipy.special import expit
        z = mdl.decision_function(X)
        if z.ndim == 1:
            return expit(z)
        from sklearn.utils.extmath import softmax
        return softmax(z)[:, 1]
    else:
        return mdl.predict(X)

# ============================================================================
# CALCULAR PROBABILIDADES E MÉTRICAS
# ============================================================================
proba = proba_safe(model, X_test_trans)
y_pred_thr = (proba >= threshold).astype(int)

# Calcular métricas com o threshold ajustado
acc = accuracy_score(y_test, y_pred_thr)
prec = precision_score(y_test, y_pred_thr, zero_division=0)
rec = recall_score(y_test, y_pred_thr, zero_division=0)
f1 = f1_score(y_test, y_pred_thr, zero_division=0)
auc = roc_auc_score(y_test, proba)

# ============================================================================
# EXIBIR KPIs PRINCIPAIS
# ============================================================================
st.markdown("---")
st.subheader("📊 Métricas de Performance")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Acurácia", f"{acc:.3f}", help="Proporção de previsões corretas (classe 0 e 1)")
kpi2.metric("Precisão (Classe 1)", f"{prec:.3f}", help="Dos que o modelo previu como 'Sai', quantos realmente saíram?")
kpi3.metric("Recall (Classe 1)", f"{rec:.3f}", help="Dos que realmente saíram, quantos o modelo conseguiu identificar?")
kpi4.metric("F1-Score (Classe 1)", f"{f1:.3f}", help="Média harmônica entre Precisão e Recall")
kpi5.metric("AUC", f"{auc:.4f}", help="Área sob a curva ROC (quanto mais próximo de 1, melhor)")

st.caption(f"**Modelo:** {MODEL_DISPLAY_NAMES.get(model_name, model_name)} | **Limiar:** {threshold:.2f} | **Observações de teste:** {len(y_test)}")

# ============================================================================
# CURVA ROC (VISUALIZAÇÃO APRIMORADA)
# ============================================================================
st.markdown("---")
st.subheader("📈 Curva ROC (Receiver Operating Characteristic)")

fpr, tpr, thresholds = roc_curve(y_test, proba)
roc_auc = roc_auc_score(y_test, proba)

# Criar gráfico interativo com Plotly
fig_roc = go.Figure()

# Curva ROC
fig_roc.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name=f'Curva ROC (AUC = {roc_auc:.4f})',
    line=dict(color='#1f77b4', width=2),
    hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
))

# Linha diagonal (modelo aleatório)
fig_roc.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Modelo Aleatório (AUC = 0.5)',
    line=dict(color='gray', width=1, dash='dash')
))

# Ponto correspondente ao threshold atual
idx_threshold = np.argmin(np.abs(thresholds - threshold))
fig_roc.add_trace(go.Scatter(
    x=[fpr[idx_threshold]],
    y=[tpr[idx_threshold]],
    mode='markers',
    name=f'Threshold = {threshold:.2f}',
    marker=dict(color='red', size=10, symbol='circle'),
    hovertemplate=f'Threshold: {threshold:.2f}<br>FPR: {fpr[idx_threshold]:.3f}<br>TPR: {tpr[idx_threshold]:.3f}<extra></extra>'
))

fig_roc.update_layout(
    xaxis_title='Taxa de Falsos Positivos (FPR)',
    yaxis_title='Taxa de Verdadeiros Positivos (TPR)',
    hovermode='closest',
    height=500,
    showlegend=True
)

st.plotly_chart(fig_roc, use_container_width=True)

st.caption("""
**Interpretação:** A curva ROC mostra o trade-off entre a taxa de verdadeiros positivos (TPR, Recall) e a taxa de falsos positivos (FPR) 
para diferentes valores de threshold. Quanto mais próxima a curva estiver do canto superior esquerdo, melhor o modelo.
""")

# ============================================================================
# MATRIZ DE CONFUSÃO (VISUALIZAÇÃO APRIMORADA)
# ============================================================================
st.markdown("---")
st.subheader("🔢 Matriz de Confusão")

cm = confusion_matrix(y_test, y_pred_thr)

# Criar heatmap interativo com Plotly
fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Previsto: Fica (0)', 'Previsto: Sai (1)'],
    y=['Real: Fica (0)', 'Real: Sai (1)'],
    colorscale='Blues',
    text=cm,
    texttemplate='%{text}',
    textfont={"size": 20},
    hovertemplate='Real: %{y}<br>Previsto: %{x}<br>Contagem: %{z}<extra></extra>'
))

fig_cm.update_layout(
    xaxis_title='Classe Prevista',
    yaxis_title='Classe Real',
    height=400
)

st.plotly_chart(fig_cm, use_container_width=True)

# Exibir a matriz também em formato de tabela
cm_df = pd.DataFrame(
    cm,
    index=["Real: Fica (0)", "Real: Sai (1)"],
    columns=["Previsto: Fica (0)", "Previsto: Sai (1)"]
)

col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(cm_df.style.format("{:}"), use_container_width=True)

with col2:
    st.markdown("""
    **Interpretação:**
    - **Verdadeiros Negativos (TN):** Funcionários que ficaram e o modelo previu corretamente.
    - **Falsos Positivos (FP):** Funcionários que ficaram, mas o modelo previu que sairiam (alarme falso).
    - **Falsos Negativos (FN):** Funcionários que saíram, mas o modelo previu que ficariam (perda de oportunidade de retenção).
    - **Verdadeiros Positivos (TP):** Funcionários que saíram e o modelo previu corretamente.
    """)

# ============================================================================
# IMPORTÂNCIA DE VARIÁVEIS / COEFICIENTES
# ============================================================================
st.markdown("---")
st.subheader("🔍 Variáveis Mais Relevantes")

imp_info = importances.get(model_name)

if imp_info:
    # Criar DataFrame com as importâncias
    imp_df = pd.DataFrame({
        'Variável': imp_info['features'],
        'Importância': imp_info['values']
    }).sort_values('Importância', ascending=False)
    
    # Slider para selecionar o número de variáveis a exibir
    topn = st.slider(
        "Quantidade de variáveis a exibir (Top N)",
        min_value=5,
        max_value=min(30, len(imp_df)),
        value=15,
        step=1
    )
    
    # Filtrar as top N variáveis
    imp_top = imp_df.head(topn)
    
    # Criar gráfico de barras interativo com Plotly
    fig_imp = px.bar(
        imp_top,
        x='Importância',
        y='Variável',
        orientation='h',
        title=f'Top {topn} Variáveis Mais Importantes',
        color='Importância',
        color_continuous_scale='Viridis'
    )
    
    fig_imp.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=max(400, topn * 25),
        showlegend=False
    )
    
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Nota de interpretação
    if model_name == "log_reg":
        st.info("""
        **Nota para Regressão Logística:** Os valores representam os **coeficientes** do modelo.
        - **Valores positivos:** Aumentam a probabilidade de *Attrition* (saída).
        - **Valores negativos:** Reduzem a probabilidade de *Attrition* (aumentam a retenção).
        """)
    else:
        st.info("""
        **Nota para Modelos de Árvore (Random Forest/XGBoost):** Os valores representam a **importância** de cada variável,
        medida pela redução média na impureza (Gini ou Entropia) ao usar essa variável para dividir os dados.
        """)
else:
    st.warning("⚠️ Importâncias/coeficientes não encontrados para este modelo.")

# ============================================================================
# ANÁLISE POR SEGMENTOS
# ============================================================================
st.markdown("---")
st.subheader("📊 Análise por Segmentos (Taxa de Attrition Prevista)")

# Identificar colunas categóricas disponíveis para segmentação
segment_cols = [
    c for c in ["JobRole", "Department", "BusinessTravel", "MaritalStatus", "Gender", "EducationField"]
    if c in X_test_raw.columns
]

if segment_cols:
    seg_col = st.selectbox(
        "Agrupar por",
        segment_cols,
        index=0,
        help="Selecione uma variável categórica para segmentar os resultados."
    )
    
    # Criar DataFrame temporário para análise
    tmp = X_test_raw[[seg_col]].copy()
    tmp["y_true"] = y_test
    tmp["proba"] = proba
    tmp["y_pred"] = y_pred_thr
    
    # Agrupar por segmento e calcular métricas
    grp = tmp.groupby(seg_col).agg(
        n=("y_true", "size"),
        attrition_real=("y_true", "mean"),
        attrition_prevista=("y_pred", "mean"),
        risco_medio_previsto=("proba", "mean")
    ).sort_values("risco_medio_previsto", ascending=False)
    
    # Exibir tabela formatada
    st.dataframe(
        grp.style.format({
            "attrition_real": "{:.2%}",
            "attrition_prevista": "{:.2%}",
            "risco_medio_previsto": "{:.2%}"
        }),
        use_container_width=True
    )
    
    # Criar gráfico de barras para visualizar o risco médio por segmento
    fig_seg = px.bar(
        grp.reset_index(),
        x=seg_col,
        y='risco_medio_previsto',
        title=f'Risco Médio de Attrition por {seg_col}',
        labels={'risco_medio_previsto': 'Risco Médio Previsto (%)', seg_col: seg_col},
        color='risco_medio_previsto',
        color_continuous_scale='Reds'
    )
    
    fig_seg.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig_seg, use_container_width=True)
    
    st.caption("""
    **Interpretação:** Use esta análise para priorizar ações de retenção. Segmentos com **risco médio previsto** mais alto
    devem ser o foco de intervenções do RH (ex.: revisão de remuneração, realocação, programas de desenvolvimento).
    """)
else:
    st.warning("⚠️ Nenhuma coluna categórica original disponível para análise segmentada no `X_test_raw`.")

# ============================================================================
# DOWNLOAD DE PREVISÕES DETALHADAS
# ============================================================================
st.markdown("---")
st.subheader("💾 Download de Previsões (Detalhe por Funcionário)")

export_df = X_test_raw.copy()
export_df["proba_attrition"] = proba
export_df["pred_attrition"] = y_pred_thr
export_df["y_true"] = y_test

st.download_button(
    label="📥 Baixar CSV com Previsões",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="attrition_predicoes.csv",
    mime="text/csv",
    help="Baixe um arquivo CSV com as previsões detalhadas para cada funcionário do conjunto de teste."
)

# ============================================================================
# NOTAS DE INTERPRETAÇÃO E RECOMENDAÇÕES
# ============================================================================
st.markdown("---")
with st.expander("📚 Notas de Interpretação e Recomendações"):
    st.markdown("""
    ### Como Usar Este Dashboard
    
    1. **Ajuste o Limiar de Decisão:**
       - Use o slider na barra lateral para controlar o trade-off entre **Recall** (capturar mais funcionários em risco) 
         e **Precisão** (evitar alarmes falsos).
       - Para ações preventivas de RH, um **Recall alto** é geralmente mais importante (ex.: threshold = 0.3).
    
    2. **Variáveis Mais Relevantes:**
       - Identifique os fatores que mais influenciam a rotatividade.
       - Use essas informações para direcionar **ações de retenção** (ex.: revisão de remuneração, alocação, carga de trabalho).
    
    3. **Segmentos de Maior Risco:**
       - Priorize **JobRoles/Departamentos** com **risco médio previsto** mais alto.
       - Implemente programas de retenção específicos para esses grupos.
    
    4. **AUC (Area Under the ROC Curve):**
       - Resume o poder de separação do modelo (quanto mais próximo de 1, melhor).
       - Um AUC > 0.8 é considerado excelente para problemas de negócio.
    
    ### Recomendações de Negócio
    
    - **Remuneração Competitiva:** Se `MonthlyIncome` for uma variável importante, revise a política salarial.
    - **Desenvolvimento de Carreira:** Se `YearsSinceLastPromotion` ou `JobLevel` forem importantes, crie trilhas de carreira mais claras.
    - **Equilíbrio Trabalho-Vida:** Se `OverTime` for importante, revise a política de horas extras e carga de trabalho.
    - **Engajamento Inicial:** Se `YearsAtCompany` for importante, foque em programas de mentoria e retenção nos primeiros anos.
    """)

# ============================================================================
# RODAPÉ
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Desenvolvido com ❤️ usando Streamlit | Projeto de Machine Learning - Attrition</p>
</div>
""", unsafe_allow_html=True)
