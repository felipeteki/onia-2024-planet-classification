# Exoplanet Classification - ONIA 2024 🪐🚀

*(Português abaixo)*

Professional machine learning pipeline developed for the **1st National Artificial Intelligence Olympiad (ONIA)**. This project implements advanced ensemble techniques and data balancing to classify habitability metrics in unknown planetary systems.

## 📊 Performance Results
* **Optimized XGBoost (GridSearchCV):** **91.87% F1-Score** (Weighted)
* **Stacking Ensemble (Baseline):** 79.07% F1-Score
* **Status:** `predicoes.csv` successfully generated for final submission.

## 🛠️ Technical Stack & Advanced Methods
* **Core:** Python, Pandas, Numpy.
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM.
* **Class Imbalance:** Applied **SMOTE** to ensure model fairness across all 5 classes.
* **Optimization:** Exhaustive Hyperparameter Tuning via **GridSearchCV** (best params: `max_depth: 10`, `n_estimators: 300`, `learning_rate: 0.1`).

---

# Classificação de Exoplanetas - ONIA 2024 🪐🚀

Pipeline profissional de Machine Learning desenvolvido para a **1ª Olimpíada Nacional de Inteligência Artificial (ONIA)**. O projeto implementa técnicas avançadas de ensemble e balanceamento de dados para classificação de métricas de habitabilidade em sistemas planetários desconhecidos.

## 📊 Resultados de Performance
* **XGBoost Otimizado (GridSearchCV):** **91.87% de F1-Score** (Weighted)
* **Stacking Ensemble (Baseline):** 79.07% de F1-Score
* **Status:** Arquivo `predicoes.csv` gerado com sucesso para submissão final.

## 🛠️ Tecnologias e Métodos Avançados
* **Tratamento de Dados:** Aplicação de **SMOTE** para balanceamento das 5 classes planetárias.
* **Arquitetura:** Uso de **Stacking Classifier** e **XGBoost** de alta performance.
* **Otimização:** Busca exaustiva de hiperparâmetros (melhores parâmetros encontrados: profundidade 10, 300 estimadores).

## 🧠 Evolução Técnica e Comparação
Este repositório documenta a evolução de modelos lineares simples para arquiteturas complexas. 
> **Nota:** O arquivo da **1ª Matriz de Confusão (2024)** está mantido para fins de comparação. A versão atual (v2) alcançou **91.87%**, superando significativamente os testes preliminares.

---
**Developed by [Felipe Teki](https://www.linkedin.com/in/felipeteki/)**
