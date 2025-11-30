🩺 Breast Cancer Classification Using Machine Learning

Este repositório contém um projeto completo de classificação de câncer de mama utilizando modelos de Machine Learning aplicados ao Breast Cancer Wisconsin (Diagnostic) Dataset.
O objetivo é comparar diferentes algoritmos de classificação e identificar qual apresenta melhor desempenho para apoiar diagnósticos médicos.

📌 Conteúdo do Repositório
/data            → dataset ou link para download  
/notebooks       → análise exploratória e experimentos  
/src             → scripts Python com a implementação dos modelos  
/results         → tabelas e figuras geradas  
requirements.txt → dependências do projeto  
README.md        → documentação geral  

📥 Dataset

O projeto utiliza o Breast Cancer Wisconsin (Diagnostic) Dataset, disponível diretamente no Scikit-Learn ou para download em:

🔗 https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Para carregar via Scikit-Learn:

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

🧠 Modelos Utilizados

Os seguintes modelos foram treinados:

Logistic Regression

SVM com Kernel Linear

SVM com Kernel Polinomial (grau 2 e grau 3)

Todos os dados passam por:

✔ Padronização (StandardScaler)
✔ Separação em treino/teste (398/171 amostras)
✔ Avaliação comparativa entre modelos

🛠️ Como Executar os Scripts
1. Clone o repositório:
git clone https://github.com/LGAP159/cancer-machine
cd breast-cancer-classification-ml

2. Instale as dependências:
pip install -r requirements.txt

3. Execute o script principal:
python src/train_models.py


Isso irá:

carregar e padronizar os dados

treinar todos os modelos

salvar métricas em /results/

gerar matrizes de confusão, curvas ROC e tabela comparativa

📓 Notebooks

O notebook principal se encontra em:

notebooks/breast_cancer_analysis.ipynb


Ele contém:

EDA (exploração do dataset)

Justificativa dos modelos

Treinamento

Avaliação com gráficos

Comparação final

📊 Resultados Obtidos (Resumo)

O modelo com melhor desempenho foi:

⭐ SVM com Kernel Polinomial (Grau 2)

Com:

Acurácia alta

F1 excelente

AUC superior

100% de precisão para tumores malignos

💻 Tecnologias Utilizadas

Python

pandas

numpy

scikit-learn

matplotlib

📄 requirements.txt sugerido
numpy
pandas
scikit-learn
matplotlib
jupyter
