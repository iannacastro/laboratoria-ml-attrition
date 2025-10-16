import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from sklearn.base import BaseEstimator, ClassifierMixin

# --- Definições de Classes para Desserialização Segura ---
# Estas classes precisam estar disponíveis no escopo ao carregar os artefatos

class DummyScaler(BaseEstimator):
    def fit(self, X, y=None):
        # Simula o fit do scaler
        numeric_cols = X.select_dtypes(include=np.number).columns
        self.mean_ = X[numeric_cols].mean()
        self.std_ = X[numeric_cols].std()
        return self

    def transform(self, X):
        # Simula a transformação de colunas numéricas
        X_scaled = X.copy()
        numeric_cols = X.select_dtypes(include=np.number).columns
        X_scaled[numeric_cols] = (X_scaled[numeric_cols] - self.mean_) / self.std_
        # Retorna um array numpy, como o scaler real faria
        return X_scaled.values

class SafeDummyModel(BaseEstimator, ClassifierMixin):
    def __init__(self, proba_values=None):
        self.proba_values = proba_values

    def predict_proba(self, X):
        # Retorna as probabilidades simuladas
        # Garante que o número de predições corresponda ao input X
        n_samples = X.shape[0]
        # Usa as probabilidades simuladas, repetindo ou truncando se necessário
        simulated_probas = np.tile(self.proba_values, int(np.ceil(n_samples / len(self.proba_values))))[:n_samples]
        return np.column_stack([1 - simulated_probas, simulated_probas])

# --- Geração dos Dados e Artefatos ---

# Configurações
np.random.seed(42)
N_SAMPLES = 1000

# Variáveis categóricas
job_roles = ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']
departments = ['Research & Development', 'Sales', 'Human Resources']
marital_status = ['Single', 'Married', 'Divorced']
business_travel = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
gender = ['Male', 'Female']
education_field = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources']

# Gerar dados brutos (simulados)
X_test_raw = pd.DataFrame({
    'Age': np.random.randint(18, 60, N_SAMPLES),
    'MonthlyIncome': np.random.randint(1500, 20000, N_SAMPLES),
    'YearsAtCompany': np.random.randint(0, 40, N_SAMPLES),
    'JobRole': np.random.choice(job_roles, N_SAMPLES),
    'Department': np.random.choice(departments, N_SAMPLES),
    'MaritalStatus': np.random.choice(marital_status, N_SAMPLES),
    'BusinessTravel': np.random.choice(business_travel, N_SAMPLES),
    'Gender': np.random.choice(gender, N_SAMPLES),
    'EducationField': np.random.choice(education_field, N_SAMPLES),
    'OverTime_Yes': np.random.randint(0, 2, N_SAMPLES),
    'DistanceFromHome': np.random.randint(1, 30, N_SAMPLES),
    'YearsSinceLastPromotion': np.random.randint(0, 15, N_SAMPLES),
})

# Gerar y_test (rótulos reais)
y_test = pd.DataFrame({'y_test': np.random.choice([0, 1], N_SAMPLES, p=[0.84, 0.16])})

# Gerar probabilidades (simuladas)
# A probabilidade é correlacionada com y_test para simular um modelo razoável
proba_simulated = np.clip(
    y_test['y_test'].values * 0.5 + np.random.rand(N_SAMPLES) * 0.4,
    0.01, 0.99
)

# Criar diretório de artefatos se não existir
ART = Path("artifacts_attrition")
ART.mkdir(exist_ok=True)

# Salvar dados
y_test.to_csv(ART / "y_test.csv", index=False)
X_test_raw.to_csv(ART / "X_test_raw.csv", index=False)

# Instanciar, treinar e salvar o scaler
dummy_scaler = DummyScaler()
dummy_scaler.fit(X_test_raw)
joblib.dump(dummy_scaler, ART / "scaler.joblib")

# Instanciar e salvar os modelos simulados
joblib.dump(SafeDummyModel(proba_values=proba_simulated), ART / "xgboost.joblib")
joblib.dump(SafeDummyModel(proba_values=proba_simulated * 0.9), ART / "random_forest.joblib") # RF um pouco diferente
joblib.dump(SafeDummyModel(proba_values=proba_simulated * 0.8), ART / "log_reg.joblib")     # LR um pouco diferente


# Simular importâncias de variáveis
importances = {
    "xgboost": {
        "features": ['MonthlyIncome', 'OverTime_Yes', 'Age', 'YearsAtCompany', 'DistanceFromHome', 'YearsSinceLastPromotion', 'JobRole', 'Department'],
        "values": [0.35, 0.28, 0.15, 0.1, 0.05, 0.04, 0.02, 0.01]
    },
    "random_forest": {
        "features": ['MonthlyIncome', 'Age', 'OverTime_Yes', 'YearsAtCompany', 'DistanceFromHome', 'YearsSinceLastPromotion', 'JobRole', 'Department'],
        "values": [0.33, 0.25, 0.18, 0.12, 0.06, 0.03, 0.02, 0.01]
    },
    "log_reg": {
        "features": ['MonthlyIncome', 'OverTime_Yes', 'YearsAtCompany', 'Age', 'DistanceFromHome', 'YearsSinceLastPromotion', 'JobRole', 'Department'],
        "values": [0.4, 0.3, 0.12, 0.1, 0.04, 0.02, 0.01, 0.01]
    }
}
with open(ART / "importances.json", "w") as f:
    json.dump(importances, f)

# Simular métricas de referência
metrics_test = {
    "xgboost": {"auc": 0.90},
    "random_forest": {"auc": 0.88},
    "log_reg": {"auc": 0.85}
}
with open(ART / "metrics_test.json", "w") as f:
    json.dump(metrics_test, f)

print(f"Dados de teste simulados e artefatos salvos em '{ART}/'.")

