# Exoplanet Classification - ONIA 2024 🪐🚀

*(Português abaixo)*

This repository contains the code and results for the challenge developed for the **1st National Artificial Intelligence Olympiad (ONIA)** in 2024. This was my first real project using Python and Machine Learning libraries, developed at age 18 during my senior year of high school.

## 📌 Project Overview
The goal of the challenge was to create an AI model capable of classifying planets in unknown systems based on astronomical data provided in `.csv` files.

## 🛠️ Tech Stack & Methodology
* **Language:** Python.
* **Main Libraries:** Pandas, Scikit-Learn.
* **Data Preprocessing:** Handled missing values (NaN) by filling them with the mean, utilized `LabelEncoder` for categorical targets, and `StandardScaler` for feature scaling.
* **Model:** An **Ensemble (VotingClassifier)** combining **RandomForestClassifier** and **GradientBoostingClassifier** to achieve more robust and accurate predictions.

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have the input files (`dados_treino.csv` and `dados_teste.csv`) in the same folder.
3. Install dependencies: `pip install pandas scikit-learn`.
4. Run the script: `python desafiooniafinalizado.py`.

---

# Classificação de Exoplanetas - ONIA 2024 🪐🚀

Este repositório contém o código e os resultados do desafio desenvolvido para a **1ª Olimpíada Nacional de Inteligência Artificial (ONIA)** em 2024. Este foi o meu primeiro projeto real utilizando Python e bibliotecas de Machine Learning, desenvolvido aos 18 anos, durante o meu 3º ano do ensino médio.

## 📌 Sobre o Projeto
O objetivo do desafio era criar um modelo de IA capaz de classificar planetas em sistemas desconhecidos com base em dados astronómicos fornecidos em arquivos `.csv`.

## 🛠️ Tecnologias e Metodologia
* **Linguagem:** Python.
* **Bibliotecas:** Pandas, Scikit-Learn.
* **Processamento de Dados:** Tratamento de valores ausentes (NaN) com a média, uso de `LabelEncoder` para classes e `StandardScaler` para normalização dos dados.
* **Modelo:** Foi utilizado um **Ensemble (VotingClassifier)** que combina a votação dos modelos **RandomForestClassifier** e **GradientBoostingClassifier**.

## 🚀 Como Executar
1. Clone este repositório.
2. Certifique-se de que os arquivos de entrada (`dados_treino.csv` e `dados_teste.csv`) estão na mesma pasta.
3. Instale as dependências: `pip install pandas scikit-learn`.
4. Execute o script: `python desafiooniafinalizado.py`.

## 🧠 Reflexão Pessoal
Este projeto representou a minha transição da lógica de blocos (App Inventor) para o desenvolvimento profissional. Foi essa experiência que me deu a certeza de que queria seguir na **Engenharia da Computação** e me tornar um **Engenheiro de Software**.

---
*Developed by Felipe Teki*
