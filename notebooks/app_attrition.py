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
from sklearn.base import BaseEstimator, ClassifierMixin

# --- Definições de Classes para Desserialização Segura ---
# Estas classes permitem que o joblib carregue os modelos simulados

class DummyScaler(BaseEstimator):
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=np.number).columns
        self.mean_ = X[numeric_cols].mean()
        self.std_ = X[numeric_cols].std()
        return self

    def transform(self, X):
        X_scaled = X.copy()
        numeric_cols = X.select_dtypes(include=np.number).columns
        # Converte apenas as colunas numéricas para valores, mantendo a estrutura do DataFrame
        X_scaled[numeric_cols] = (X_scaled[numeric_cols] - self.mean_) / self.std_
        # One-hot encode para colunas categóricas, se necessário (simulação simples)
        X_scaled = pd.get_dummies(X_scaled, drop_first=True)
        return X_scaled.values

class SafeDummyModel(BaseEstimator, ClassifierMixin):
    def __init__(self, proba_values=None):
        self.proba_values = proba_values

    def predict_proba(self, X):
        n_samples = X.shape[0]
        simulated_probas = np.tile(self.proba_values, int(np.ceil(n_samples / len(self.proba_values))))[:n_samples]
        return np.column_stack([1 - simulated_probas, simulated_probas])

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Attrition ML – Dashboard Avançado",
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
st.title("🎯 Attrition ML – Dashboard de Resultados Avançado")
st.markdown('''
Este dashboard apresenta os resultados do modelo de Machine Learning para prever a **rotatividade de funcionários** (*Attrition*).
Use os controles na barra lateral para explorar diferentes modelos e ajustar o limiar de decisão.
''')

# ============================================================================
# SIDEBAR: SELEÇÃO DE MODELO E THRESHOLD
# ============================================================================
st.sidebar.header("⚙️ Configurações")

# Verificar quais modelos estão disponíveis
model_options = [p.stem for p in ART.glob("*.joblib") if p.stem != 'scaler']

if not model_options:
    st.error(f"❌ Nenhum modelo encontrado em `{ART}/`. Execute o script `generate_data.py` para gerar os artefatos.")
    st.stop()

# Seleção do modelo
model_name = st.sidebar.selectbox(
    "Selecione o Modelo",
    model_options,
    format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x),
    index=model_options.index('xgboost') if 'xgboost' in model_options else 0
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
st.sidebar.info('''
**💡 Dica:** 
- **Threshold baixo (ex: 0.3):** Captura mais funcionários em risco (alto Recall), mas com mais falsos positivos.
- **Threshold alto (ex: 0.7):** Menos falsos positivos (alta Precisão), mas pode perder funcionários em risco.
''')

# ============================================================================
# CARREGAR ARTEFATOS
# ============================================================================
try:
    with open(ART / "metrics_test.json", "r", encoding="utf-8") as f:
        metrics_all = json.load(f)
    metrics_ref = metrics_all.get(model_name, {})

    with open(ART / "importances.json", "r", encoding="utf-8") as f:
        importances = json.load(f)

    model = joblib.load(ART / f"{model_name}.joblib")
    scaler = joblib.load(ART / "scaler.joblib")
    y_test = pd.read_csv(ART / "y_test.csv")["y_test"].values
    X_test_raw = pd.read_csv(ART / "X_test_raw.csv")

    # O scaling é simulado, mas a chamada é mantida para consistência
    X_test_scaled = scaler.transform(X_test_raw)

except FileNotFoundError as e:
    st.error(f"❌ Arquivo não encontrado: {e.filename}. Certifique-se de que o script `generate_data.py` foi executado.")
    st.stop()
except Exception as e:
    st.error(f"❌ Erro ao carregar artefatos: {str(e)}")
    st.stop()

# ============================================================================
# CALCULAR PROBABILIDADES E MÉTRICAS
# ============================================================================
proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred_thr = (proba >= threshold).astype(int)

acc = accuracy_score(y_test, y_pred_thr)
prec = precision_score(y_test, y_pred_thr, zero_division=0)
rec = recall_score(y_test, y_pred_thr, zero_division=0)
f1 = f1_score(y_test, y_pred_thr, zero_division=0)
auc = roc_auc_score(y_test, proba)

cm = confusion_matrix(y_test, y_pred_thr)
TN, FP, FN, TP = cm.ravel()

total_neg = TN + FP
total_pos = TP + FN
specificity = TN / total_neg if total_neg > 0 else 0
fpr = FP / total_neg if total_neg > 0 else 0
fnr = FN / total_pos if total_pos > 0 else 0

# ============================================================================
# EXIBIR KPIS PRINCIPAIS
# ============================================================================
st.markdown("---")
st.subheader("📊 Métricas de Performance")

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Acurácia", f"{acc:.3f}")
kpi2.metric("Precisão", f"{prec:.3f}")
kpi3.metric("Recall", f"{rec:.3f}")
kpi4.metric("Especificidade", f"{specificity:.3f}")
kpi5.metric("F1-Score", f"{f1:.3f}")
kpi6.metric("AUC", f"{auc:.4f}", help=f"AUC de Referência: {metrics_ref.get('auc', 0):.4f}")

st.caption(f"**Modelo:** {MODEL_DISPLAY_NAMES.get(model_name, model_name)} | **Limiar:** {threshold:.2f} | **Observações:** {len(y_test)}")

# (O restante do código para visualizações continua aqui...)

# ============================================================================
# CURVA ROC E MATRIZ DE CONFUSÃO
# ============================================================================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Curva ROC")
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_roc, y=tpr_roc, mode='lines', name=f'ROC (AUC = {auc:.4f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatório', line=dict(dash='dash')))
    idx_threshold = np.argmin(np.abs(thresholds_roc - threshold))
    fig_roc.add_trace(go.Scatter(x=[fpr_roc[idx_threshold]], y=[tpr_roc[idx_threshold]], mode='markers', name=f'Limiar Atual', marker=dict(color='red', size=10)))
    fig_roc.update_layout(xaxis_title='Taxa de Falsos Positivos', yaxis_title='Taxa de Verdadeiros Positivos', height=400, template="plotly_white")
    st.plotly_chart(fig_roc, use_container_width=True)

with col2:
    st.subheader("🔢 Matriz de Confusão")
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=['Prev: Fica', 'Prev: Sai'], y=['Real: Fica', 'Real: Sai'],
        colorscale='Blues', text=cm, texttemplate='%{text}', textfont={"size": 16}))
    fig_cm.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_cm, use_container_width=True)

# ============================================================================
# ANÁLISE COMPARATIVA E POR SEGMENTOS
# ============================================================================
st.markdown("---")
st.subheader("📊 Análise Comparativa e por Segmentos")

tmp = X_test_raw.copy()
tmp["Risco de Attrition (%)"] = proba * 100
tmp["Attrition Previsto"] = (proba >= threshold).astype(int)

# Seleção de variáveis para análise
all_cols = sorted(X_test_raw.columns.tolist())
numeric_cols = sorted(X_test_raw.select_dtypes(include=np.number).columns.tolist())

var_dist = st.selectbox("Selecione uma variável para análise de distribuição:", all_cols)

if var_dist in numeric_cols:
    fig_dist = px.histogram(tmp, x=var_dist, color="Attrition Previsto", marginal="box", 
                            title=f"Distribuição de {var_dist} por Attrition Previsto",
                            color_discrete_map={0: '#1f77b4', 1: '#d62728'})
else:
    fig_dist = px.density_heatmap(tmp, x=var_dist, y="Risco de Attrition (%)",
                                  title=f"Risco de Attrition por {var_dist}",
                                  histfunc="avg")

fig_dist.update_layout(template="plotly_white")
st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================================
# IMPORTÂNCIA DAS VARIÁVEIS
# ============================================================================
st.markdown("---")
st.subheader("🔍 Variáveis Mais Relevantes")

imp_info = importances.get(model_name)
if imp_info:
    imp_df = pd.DataFrame(imp_info).sort_values('values', ascending=True)
    fig_imp = px.bar(imp_df, x='values', y='features', orientation='h', title=f"Importância das Variáveis ({MODEL_DISPLAY_NAMES.get(model_name, model_name)})")
    fig_imp.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.warning("Importâncias não disponíveis para este modelo.")

# ============================================================================
# RODAPÉ
# ============================================================================
st.markdown('''
<div style="text-align: center; color: gray; margin-top: 50px;">
    <p>Desenvolvido com Streamlit | Projeto de Machine Learning - Attrition</p>
</div>
''', unsafe_allow_html=True)

