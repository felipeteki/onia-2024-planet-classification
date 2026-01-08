print("Importando bibliotecas... (Aguarde, isso pode levar alguns segundos)")

import os
import pandas as pd  # Manipulação de dados
import numpy as np  # Cálculos numéricos
import matplotlib.pyplot as plt  # Gráficos
import seaborn as sns  # Gráficos estatísticos
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Balanceamento das classes
import xgboost as xgb  # XGBoost
import lightgbm as lgb  # LightGBM
from sklearn.neural_network import MLPClassifier  # Rede neural

print("Bibliotecas importadas com sucesso!")

# --- CONFIGURAÇÃO DE CAMINHOS DINÂMICOS ---
# Garante que o script encontre os arquivos CSV na mesma pasta onde ele está salvo
diretorio_atual = os.path.dirname(os.path.abspath(__file__))

def caminho_arquivo(nome_arquivo):
    return os.path.join(diretorio_atual, nome_arquivo)
# ------------------------------------------

# Carregar os dados
print("Carregando dados...")
try:
    df_train = pd.read_csv(caminho_arquivo("treino.csv"))
    df_test = pd.read_csv(caminho_arquivo("teste.csv"))
    print("Dados carregados!")
except FileNotFoundError as e:
    print(f"Erro: Os arquivos .csv não foram encontrados na pasta: {diretorio_atual}")
    exit()

# Renomear colunas conforme as métricas dos planetas
print("Renomeando colunas...")
new_column_names = {
    "col_0": "TempMédia", "col_1": "Gravidade", "col_2": "PressãoAtm", "col_3": "Radiação",
    "col_4": "ComposiçãoAr", "col_5": "Hidratação", "col_6": "Vegetação", "col_7": "Fauna",
    "col_8": "SoloFértil", "col_9": "Ventos", "col_10": "Luas", "col_11": "Magnetismo", "col_12": "ClimaEstável"
}
df_train.rename(columns=new_column_names, inplace=True)
df_test.rename(columns=new_column_names, inplace=True)
print("Colunas renomeadas!")

# Separar features e target
print("Tirando as colunas ID e Target...")
X = df_train.drop(columns=["id", "target"])
y = df_train["target"]
print("ID e Target removidos com sucesso!")

# Dividir os dados em treino e validação
print("Começando a divisão entre treino e teste...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Divisão feita com sucesso!")

# Balanceamento das classes usando SMOTE
print("Iniciando balanceamento das classes com SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Balanceamento concluído!")

# Escalonamento dos dados
print("Transformando os dados (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)
print("Dados transformados!")

# Modelos para combinação no Stacking
print("Configurando modelos para o Stacking...")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42, class_weight="balanced")),
    ('xgb', xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=700, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1))
]

# Criar o modelo Stacking
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier(n_estimators=200, random_state=42)
)

# Treinamento do modelo Stacking
print("Iniciando treinamento do modelo Stacking (Isso pode demorar)...")
stacking_model.fit(X_train_scaled, y_train_res)
y_pred = stacking_model.predict(X_val_scaled)
print("Treinamento do Stacking concluído!")

# F1-Score do Stacking
print("\n" + "="*30)
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"F1-Score Médio do Stacking: {f1 * 100:.2f}%")
print(classification_report(y_val, y_pred))
print("="*30 + "\n")

# Melhor ajuste dos hiperparâmetros para XGBoost usando GridSearchCV
print("Iniciando GridSearchCV para otimizar XGBoost...")
param_grid_xgb = {
    'max_depth': [6, 10],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search_xgb = GridSearchCV(xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
                               param_grid_xgb, cv=5, n_jobs=-1, scoring='f1_weighted')
grid_search_xgb.fit(X_train_scaled, y_train_res)

print(f"Melhores parâmetros XGBoost: {grid_search_xgb.best_params_}")
print(f"F1-Score do melhor XGBoost: {grid_search_xgb.best_score_ * 100:.2f}%")

# Matriz de Confusão para o melhor modelo XGBoost
print("Gerando Matriz de Confusão...")
y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_val_scaled)
conf_matrix_xgb = confusion_matrix(y_val, y_pred_xgb)

plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(conf_matrix_xgb, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor='black')
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - XGBoost (Valores Absolutos)")
plt.show()

# Preparar os dados de teste finais
print("Preparando dados de teste para predição final...")
X_test = df_test.drop(columns=["id"])
X_test_scaled = scaler.transform(X_test)

# Predições finais com o melhor XGBoost
y_test_pred = grid_search_xgb.best_estimator_.predict(X_test_scaled)

# Criar DataFrame com as predições
df_predicoes = pd.DataFrame({
    "id": df_test["id"],
    "target": y_test_pred
})

# Salvar arquivo CSV
caminho_saida = caminho_arquivo("predicoes.csv")
df_predicoes.to_csv(caminho_saida, index=False)

print(f"\nSucesso! Arquivo '{caminho_saida}' salvo.")
