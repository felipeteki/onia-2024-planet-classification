# Exoplanet Classification - ONIA 2024 🪐🚀

*(Português abaixo)*

Professional machine learning pipeline developed for the **1st National Artificial Intelligence Olympiad (ONIA)**. This project implements advanced ensemble techniques and data balancing to classify habitability metrics in unknown planetary systems.

## 🛠️ Technical Stack & Advanced Methods
* **Core:** Python, Pandas, Numpy.
* **Visualization:** Matplotlib, Seaborn.
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM.
* **Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure model fairness.
* **Model Architecture:** **Stacking Classifier** (Ensemble) combining:
    * Random Forest & SVM (Radial Basis Function).
    * XGBoost & LightGBM (Gradient Boosting).
    * Multi-layer Perceptron (Neural Network).
* **Hyperparameter Tuning:** Optimized via **GridSearchCV** with 5-fold Cross-Validation.
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

---

# Classificação de Exoplanetas - ONIA 2024 🪐🚀

Pipeline profissional de Machine Learning desenvolvido para a **1ª Olimpíada Nacional de Inteligência Artificial (ONIA)**. O projeto implementa técnicas avançadas de ensemble e balanceamento de dados para classificação de métricas de habitabilidade em sistemas planetários desconhecidos.

## 🛠️ Tecnologias e Métodos Avançados
* **Core:** Python, Pandas, Numpy.
* **Visualização:** Matplotlib, Seaborn.
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM.
* **Tratamento de Dados:** Aplicação de **SMOTE** para balanceamento de classes minoritárias e **StandardScaler** para normalização.
* **Arquitetura do Modelo:** **Stacking Classifier** (Ensemble) integrando múltiplos estimadores:
    * Random Forest, SVM, XGBoost, LightGBM e Redes Neurais (MLP).
* **Otimização:** Busca exaustiva de hiperparâmetros via **GridSearchCV** com Validação Cruzada (K-Fold).
* **Engenharia de Software:** Gerenciamento dinâmico de diretórios para garantir que o código rode em qualquer máquina sem ajustes manuais de caminho.

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

## 🧠 Evolução Técnica e Carreira
Este repositório documenta a evolução de modelos lineares simples para arquiteturas complexas de **Stacking**. A escolha por algoritmos de estado da arte (XGBoost/LightGBM) e o rigor no tratamento estatístico dos dados refletem meu compromisso com a excelência técnica na minha trajetória na **Engenharia da Computação**.

---
**Developed by [Felipe Teki](https://www.linkedin.com/in/SEU-LINK-AQUI)** *Aspiring Software Engineer | Java & Python Enthusiast*
