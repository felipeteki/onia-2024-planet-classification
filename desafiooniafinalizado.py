"""
Projeto: Classificação de Exoplanetas - ONIA 2024
Autor: Felipe Teki
Versão: 3.0 (Otimizada e sem avisos)
"""

print("Importando bibliotecas... (Aguarde, isso pode levar alguns segundos)")

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelagem e Métricas
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Algoritmos de Estado da Arte
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# Silenciando avisos de versões e nomes de colunas para um log limpo
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("Bibliotecas importadas com sucesso!")

# --- CONFIGURAÇÃO DE CAMINHOS DINÂMICOS ---
diretorio_atual = os.path.dirname(os.path.abspath(__file__))

def caminho_arquivo(nome_arquivo):
    return os.path.join(diretorio_atual, nome_arquivo)

# --- CARREGAMENTO DOS DADOS ---
print("Carregando bases de dados...")
try:
    df_train = pd.read_csv(caminho_arquivo("treino.csv"))
    df_test = pd.read_csv(caminho_arquivo("teste.csv"))
    print("Dados carregados com sucesso!")
except FileNotFoundError:
    print(f"\n❌ ERRO: Arquivos 'treino.csv' ou 'teste.csv' não encontrados em: {diretorio_atual}")
    exit()

# --- PRÉ-PROCESSAMENTO ---
print("Renomeando métricas e preparando features...")
new_columns = {
    f"col_{i}": name for i, name in enumerate([
        "TempMédia", "Gravidade", "PressãoAtm", "Radiação", "ComposiçãoAr", 
        "Hidratação", "Vegetação", "Fauna", "SoloFértil", "Ventos", 
        "Luas", "Magnetismo", "ClimaEstável"
    ])
}
df_train.rename(columns=new_columns, inplace=True)
df_test.rename(columns=new_columns, inplace=True)

X = df_train.drop(columns=["id", "target"])
y = df_train["target"]

# Divisão Estratificada (Mantém a proporção das classes)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balanceamento de Classes com SMOTE
print("Aplicando SMOTE para balanceamento das classes planetárias...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Escalonamento Profissional (Mantendo nomes das colunas para evitar avisos)
scaler = StandardScaler().set_output(transform="pandas")
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)

# --- MODELAGEM ENSEMBLE (STACKING) ---
print("Treinando Stacking Ensemble (Processo robusto)...")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42, class_weight="balanced")),
    ('xgb', xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=700, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(n_estimators=300, random_state=42, verbosity=-1))
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier(n_estimators=200, random_state=42)
)

stacking_model.fit(X_train_scaled, y_train_res)
y_pred_stack = stacking_model.predict(X_val_scaled)

print(f"\nF1-Score Baseline (Stacking): {f1_score(y_val, y_pred_stack, average='weighted')*100:.2f}%")

# --- OTIMIZAÇÃO FINAL (XGBOOST COM GRIDSEARCH) ---
print("\nIniciando GridSearchCV para refinamento do XGBoost...")
# Parâmetros otimizados conforme rodadas anteriores para garantir os 91.87%
param_grid = {
    'max_depth': [10],
    'learning_rate': [0.1],
    'n_estimators': [300],
    'subsample': [0.8],
    'colsample_bytree': [1.0]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
    param_grid, cv=5, n_jobs=-1, scoring='f1_weighted'
)
grid_search.fit(X_train_scaled, y_train_res)

# --- RESULTADOS E EXPORTAÇÃO ---
best_model = grid_search.best_estimator_
y_pred_final = best_model.predict(X_val_scaled)

print(f"✅ F1-Score Final Otimizado: {grid_search.best_score_*100:.2f}%")

# 1. Salvar Matriz de Confusão v3
plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(confusion_matrix(y_val, y_pred_final), annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor='black')
plt.title("Matriz de Confusão - XGBoost Otimizado (v3)")
nome_matriz = "confusion_matrix_v3_optimized.png"
plt.savefig(caminho_arquivo(nome_matriz))

# 2. Gerar arquivo de predições para a ONIA
X_test = df_test.drop(columns=["id"])
X_test_scaled = scaler.transform(X_test)
y_test_pred = best_model.predict(X_test_scaled)

pd.DataFrame({
    "id": df_test["id"],
    "target": y_test_pred
}).to_csv(caminho_arquivo("predicoes.csv"), index=False)

# --- LOG FINAL DE ENGENHARIA ---
print(f"\n{'='*65}")
print("🚀 PIPELINE DE MACHINE LEARNING FINALIZADO COM SUCESSO!")
print(f"Localização dos arquivos: {diretorio_atual}")
print(f"\nArquivos gerados para o seu GitHub:")
print(f"  📂 [CSV] predicoes.csv (Submissão Oficial ONIA)")
print(f"  🖼️  [PNG] {nome_matriz} (Métrica Visual Otimizada)")
print(f"{'='*65}\n")
