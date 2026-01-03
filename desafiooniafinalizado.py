print("Importando bibliotecas..")
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
print("Bibliotecas importadas!")

# Carregar os dados
print("Carregando dados...")
df_train = pd.read_csv("C:/Users/felip/OneDrive/Área de Trabalho/ONIA/sklearn/treino.csv")
df_test = pd.read_csv("C:/Users/felip/OneDrive/Área de Trabalho/ONIA/sklearn/teste.csv")
print("Dados carregados!")

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
print("ID e Target removidos com sucesso! Target foi salva em uma variável!")

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
print("Transformando os dados em 0 e 1...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)
print("Dados transformados!")

# Modelos para combinação no Stacking
print("Escolhendo melhor modelo...")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42, class_weight="balanced")),
    ('xgb', xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=700, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42))
]
print("Melhor modelo escolhido!")

# Criar o modelo Stacking
print("Criando o modelo Stacking...")
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier(n_estimators=200, random_state=42)
)
print("Modelo Stacking criado!")

# Treinamento do modelo Stacking
print("Iniciando treinamento do modelo...")
stacking_model.fit(X_train_scaled, y_train_res)
y_pred = stacking_model.predict(X_val_scaled)
print("Treinamento concluído!")

# F1-Score do Stacking
print("Calculando o F1 Score...")
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"F1-Score Médio do Stacking: {f1 * 100:.2f}%")
print(classification_report(y_val, y_pred))
print("-" * 50)

# Melhor ajuste dos hiperparâmetros para XGBoost usando GridSearchCV
print("Definindo melhor ajuste dos hiperparâmetros para XGBoost usando GridSearchCV...")
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
print("Pronto, definido!")

# Melhor modelo XGBoost
print("Melhores parâmetros para XGBoost:", grid_search_xgb.best_params_)
print(f"F1-Score do melhor modelo XGBoost: {grid_search_xgb.best_score_ * 100:.2f}%")

# Matriz de Confusão para o melhor modelo XGBoost
print("Criando matriz de confusão para o melhor modelo XGBoost...")
y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_val_scaled)
conf_matrix_xgb = confusion_matrix(y_val, y_pred_xgb)

# Exibir a matriz de confusão com valores absolutos
print("Preparando a matriz de confusão")
plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(conf_matrix_xgb, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor='black')
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - XGBoost (Valores Absolutos)")
plt.show()
print("Matriz de confusão exibida com sucesso!")

# Validação cruzada com XGBoost
print("Fazendo a validação cruzada com XGBoost...")
f1_scores_xgb = cross_val_score(grid_search_xgb.best_estimator_, X_train_scaled, y_train_res, cv=5, scoring='f1_weighted')
print(f"F1-Score médio com Cross-Validation para XGBoost: {f1_scores_xgb.mean() * 100:.2f}%")

# Preparar os dados de teste
print("Preparando dados de teste...")
X_test = df_test.drop(columns=["id"])
X_test_scaled = scaler.transform(X_test)
print("Dados de teste preparados!")

# Fazer as predições no conjunto de teste
print("Fazendo as predições no conjunto de teste...")
y_test_pred = grid_search_xgb.best_estimator_.predict(X_test_scaled)
print("Predições feitas com sucesso!")

# Criar DataFrame com as predições
print("Código das predições começando...")
df_predicoes = pd.DataFrame({
    "id": df_test["id"],  # Manter os mesmos IDs do arquivo de teste
    "target": y_test_pred  # Predições do modelo
})

# Salvar as predições em um arquivo CSV
print("Salvando arquivo...")
df_predicoes.to_csv("predicoes.csv", index=False)

print("Arquivo 'predicoes.csv' salvo com sucesso!")