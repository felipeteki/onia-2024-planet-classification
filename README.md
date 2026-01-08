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
