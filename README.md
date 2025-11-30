# 🩺 Cancer Machine
Este repositório contém um projeto completo de classificação de câncer de mama utilizando modelos de Machine Learning aplicados ao Breast Cancer Wisconsin (Diagnostic) Dataset.
O objetivo é comparar diferentes algoritmos de classificação e identificar qual apresenta melhor desempenho para apoiar os diagnósticos médicos sobre câncer de mama, apontando se os tumores são malignos ou benignos.

## 💻 Tecnologias Utilizadas
- Python

### 📓 Bibliotecas Requeridas
- pandas
- numpy
- scikit-learn
- matplotlib
  
## 📥 Dataset
O projeto utiliza o Breast Cancer Wisconsin (Diagnostic) Dataset, disponível diretamente no Scikit-Learn ou para download em:

🔗 https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Para carregar via Scikit-Learn:

```
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

## 🧠 Modelos de Machine Learning Utilizados
- Logistic Regression
- Support Vector Machine (SVM) com Kernel Linear
- Support Vector Machine (SVM) com Kernel Polinomial (grau 2 e grau 3)

## 🛠️ Como Executar os Scripts
1. Copie o link do repositório git:
```
https://github.com/LGAP159/cancer-machine
```

2. Instale as dependências:
```
pip install -r requirements.txt
```

3. Execute o script principal:
```
python src/train_models.py
```
### 📓 Bibliotecas Requeridas
- pandas
- numpy
- scikit-learn
- matplotlib

## 👩🏽‍💻 Código
O notebook principal se encontra em:

🔗 https://colab.research.google.com/drive/1z8ibAHjOsA3Ouc7Dn8tubyXotHzr2qFT?usp=sharing#scrollTo=jUdAD6MU3DXE 

## 💡 Sugestão 
O Jupyter é uma ferramenta perfeita para fazer análise de dados, estatística, machine learning e análises exploratórias. Caso queira utilizar em suas análises, faça assim:

- Para instalar o Jupyter
```
pip install jupyter
```

