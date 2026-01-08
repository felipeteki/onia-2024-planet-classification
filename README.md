# Exoplanet Classification - ONIA 2024 🪐🚀

*(Português abaixo)*

Professional machine learning pipeline developed for the **1st National Artificial Intelligence Olympiad (ONIA)**. This project implements advanced ensemble techniques, hyperparameter tuning, and data balancing to classify habitability metrics in unknown planetary systems.

## 📊 Performance Results
* **Optimized XGBoost (GridSearchCV):** **91.87% F1-Score** (Weighted)
* **Stacking Ensemble (Baseline):** 79.07% F1-Score
* **Status:** `predicoes.csv` successfully generated for final submission.

## 🛠️ Technical Stack & Advanced Methods
* **Core:** Python, Pandas, Numpy.
* **Visualization:** Matplotlib, Seaborn.
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM.
* **Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure model fairness across all 5 classes.
* **Model Architecture:** **Stacking Classifier** (Ensemble) and **XGBoost** optimized via **GridSearchCV**.
* **Software Engineering:** Implementation of dynamic path handling (`os` library) for cross-environment portability.

## 🚀 How to Run
1. Clone this repository.
2. Ensure `treino.csv` and `teste.csv` are in the same directory as the script.
3. Install the complete dependency list:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the main script:
    ```bash
    python desafiooniafinalizado.py
    ```

## 🧠 Technical Evolution & Comparison
This repository documents the evolution from simple linear models to complex architectures. 
> **Note:** The file regarding the **1st Confusion Matrix generated in 2024** is included for comparative purposes, showcasing the performance gains (up to **91.87%**) achieved through current optimizations compared to the baseline.

---

# Classificação de Exoplanetas - ONIA 2024 🪐🚀

Pipeline profissional de Machine Learning desenvolvido para a **1ª Olimpíada Nacional de Inteligência Artificial (ONIA)**. O projeto implementa técnicas avançadas de ensemble, ajuste de hiperparâmetros e balanceamento de dados para classificação de métricas de habitabilidade em sistemas planetários desconhecidos.

## 📊 Resultados de Performance
* **XGBoost Otimizado (GridSearchCV):** **91.87% de F1-Score** (Weighted)
* **Stacking Ensemble (Baseline):** 79.07% de F1-Score
* **Status:** Arquivo `predicoes.csv` gerado com sucesso para submissão final.

## 🛠️ Tecnologias e Métodos Avançados
* **Core:** Python, Pandas, Numpy.
* **Visualização:** Matplotlib, Seaborn.
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM.
* **Tratamento de Dados:** Aplicação de **SMOTE** para balanceamento de classes minoritárias e **StandardScaler** para normalização.
* **Arquitetura do Modelo:** **Stacking Classifier** (Ensemble) e **XGBoost** de alta performance.
* **Otimização:** Busca exaustiva de hiperparâmetros via **GridSearchCV** com Validação Cruzada (K-Fold).
* **Engenharia de Software:** Gerenciamento dinâmico de diretórios para garantir portabilidade entre diferentes ambientes de execução.

## 🚀 Como Executar
1. Clone este repositório.
2. Certifique-se de que os arquivos `treino.csv` e `teste.csv` estejam no mesmo diretório do script.
3. Instale a lista completa de dependências:
    ```bash
    pip install -r requirements.txt
    ```
4. Execute o script principal:
    ```bash
    python desafiooniafinalizado.py
    ```

## 🧠 Evolução Técnica e Comparação
Este repositório documenta a evolução de modelos simples para arquiteturas robustas. 
> **Nota:** O arquivo da **1ª Matriz de Confusão gerada em 2024** foi mantido no repositório para fins de comparação, permitindo visualizar a evolução na identificação de padrões e a melhoria de performance (atingindo **91.87%**) em relação aos testes iniciais.

---
**Developed by [Felipe Teki](https://www.linkedin.com/in/felipeteki/)**
