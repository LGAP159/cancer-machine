Classificação de Câncer de Mama com Machine Learning

Este projeto utiliza algoritmos de Machine Learning para classificar tumores de mama como benignos ou malignos, usando o Breast Cancer Wisconsin (Diagnostic) Dataset. O objetivo é comparar diferentes modelos e identificar qual apresenta melhor desempenho no auxílio ao diagnóstico médico.

📌 Objetivo

Pré-processar os dados do conjunto Breast Cancer.

Treinar e comparar modelos de classificação:

Regressão Logística

SVM Linear

SVM Polinomial (grau 2 e grau 3)

Avaliar os modelos usando métricas relevantes como:

Acurácia

Precisão

Recall

F1-Score

Curva ROC e AUC

Identificar o modelo mais eficiente para distinguir tumores benignos e malignos.

🧠 Modelos Utilizados

Logistic Regression

SVM (Support Vector Machine) com:

Kernel linear

Kernel polinomial grau 2

Kernel polinomial grau 3

Os dados foram padronizados usando StandardScaler para garantir melhor desempenho dos modelos.

📊 Resultados Resumidos

O modelo com melhor desempenho geral foi o SVM com Kernel Polinomial Grau 2.

Ele alcançou 100% de Precisão para tumores malignos, além de excelente Recall e AUC.

A padronização das features foi essencial para o bom desempenho dos modelos.

pré-processamento,

treinamento dos modelos,

avaliação final.

Verifique as curvas ROC, matrizes de confusão e tabela de métricas geradas.

📂 Dataset

O projeto utiliza o Breast Cancer Wisconsin (Diagnostic) Dataset, que contém 30 features numéricas derivadas de imagens de biópsias.

📚 Tecnologias

Python

pandas

numpy

scikit-learn

matplotlib

📝 Conclusão

A análise mostrou que modelos baseados em SVM são muito eficazes para esse tipo de classificação, especialmente quando usam kernels polinomiais. O SVM grau 2 apresentou o melhor equilíbrio entre precisão, recall e capacidade de generalização, mostrando potencial para aplicações clínicas de apoio ao diagnóstico.
<img width="618" height="470" alt="image" src="https://github.com/user-attachments/assets/97a8e046-adf8-4b0c-83a6-d659ee8d0dd7" />



tipos de tumores detectáveis
