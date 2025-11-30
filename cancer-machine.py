"""
## 1) Exploração dos dados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("--- 1. Carregamento e Exploração Inicial ---")
print("Carregando o dataset 'data.csv'...")

try:
    df = pd.read_csv('breast-cancer.csv')
except FileNotFoundError:
    print("ERRO: O arquivo 'data.csv' não foi encontrado. Verifique o caminho.")
    exit()

print("\n1.1. As 5 primeiras linhas do dataset:")
print(df.head())

print("\n--- 2. Informações Gerais e Estatísticas Descritivas ---")

print("\n2.1. Informações Gerais do DataFrame (.info()):")
df.info()

print("\n2.2. Estatísticas Descritivas das Colunas Numéricas (.describe()):")
print(df.describe().T) 

print("\n--- 3. Verificação de Valores Nulos e Duplicados ---")

print("\n3.1. Contagem de Valores Nulos:")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0]) 

print(f"\n3.2. Total de linhas duplicadas: {df.duplicated().sum()}")

print("\n--- 4. Análise da Distribuição da Variável Alvo ('diagnosis') ---")

target_counts = df['diagnosis'].value_counts()
print("\n4.1. Distribuição de Frequência da 'diagnosis':")
print(target_counts)

plt.figure(figsize=(7, 5))
sns.countplot(x='diagnosis', data=df, palette='pastel')
plt.title('Distribuição de Tumores (Benignos vs. Malignos)')
plt.xlabel('Diagnosis (M=Maligno, B=Benigno)')
plt.ylabel('Contagem')
plt.grid(axis='y', alpha=0.5)
plt.show()

total = target_counts.sum()
pct_malignant = (target_counts['M'] / total) * 100
pct_benign = (target_counts['B'] / total) * 100

print(f"4.2. Porcentagem de Malignos (M): {pct_malignant:.2f}%")
print(f"Porcentagem de Benignos (B): {pct_benign:.2f}%")

print("\n--- 5. Identificação e Preparação de Variáveis (Features e Alvo) ---")

cols_to_drop = ['id'] 
df_clean = df.drop(columns=cols_to_drop)
print(f"Colunas removidas: {cols_to_drop}")

df_clean['diagnosis_encoded'] = df_clean['diagnosis'].map({'M': 1, 'B': 0})
# M (Maligno) -> 1
# B (Benigno) -> 0

# Variáveis Preditoras (Features)
# X: Todas as colunas numéricas (Features)
X = df_clean.drop(columns=['diagnosis', 'diagnosis_encoded'])

# Variável Alvo
# Y: Coluna 'diagnosis_encoded' (Alvo)
Y = df_clean['diagnosis_encoded']

print("\n5.4. Resultado da Separação:")
print(f"Variável Alvo (Y): {Y.name}")
print(f"Shape de X (Preditoras): {X.shape}")
print(f"Shape de Y (Alvo): {Y.shape}")
print("\nPrimeiras 3 Linhas das Features (X):")
print(X.head(3))
print("\nPrimeiras 3 Linhas do Alvo Codificado (Y):")
print(Y.head(3))

"""## 2) Pré-processamento dos Dados

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("--- 1. Carregamento e Limpeza Inicial dos Dados ---")

try:
    df = pd.read_csv('breast-cancer.csv')
except FileNotFoundError:
    print("ERRO: O arquivo 'data.csv' não foi encontrado. Certifique-se de que está no diretório correto.")
    exit()

cols_to_drop = ['id']
df_clean = df.drop(columns=cols_to_drop)

print(f"Colunas removidas: {cols_to_drop}")
print(f"Shape do DataFrame após limpeza: {df_clean.shape}")

print("\n--- 2. Conversão da Variável Alvo para Binário ---")

df_clean['diagnosis_encoded'] = df_clean['diagnosis'].map({'M': 1, 'B': 0})

X = df_clean.drop(columns=['diagnosis', 'diagnosis_encoded'])
Y = df_clean['diagnosis_encoded']

print(f"Alvo 'diagnosis' mapeado para 1 (M) e 0 (B).")
print(f"Variável Alvo (Y) definida: {Y.name}")
print(f"Número de Variáveis Preditoras (X): {X.shape[1]}")

print("\n--- 3. Separação de Dados em Treino e Teste (70/30) ---")

# Usando 70% para treino e 30% para teste (dentro da faixa 70-80% / 20-30%)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3, # 30% para teste
    random_state=42,
    stratify=Y 
)

print(f"Divisão Treino/Teste (70%/30%):")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_test shape: {Y_test.shape}")

print("\n--- 4. Padronização dos Dados (StandardScaler) ---")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("StandardScaler aplicado:")
print("  - Ajustado (fit) apenas no X_train.")
print("  - Transformado em X_train e X_test.")
print("Média (deve ser próximo a 0) e Desvio Padrão (deve ser próximo a 1) para X_train_scaled:")
print(f"Média: {X_train_scaled.mean().mean():.4f}")
print(f"Desvio Padrão: {X_train_scaled.std().mean():.4f}")

print("\n--- Processamento Concluído ---")
print("Os datasets X_train_scaled, X_test_scaled, Y_train, e Y_test estão prontos para a Modelagem.")

"""## 3) Treinamento dos Modelos

"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Definir os shapes de X_train/X_test e Y_train/Y_test
n_samples = 569
n_features = 30
test_size = 0.3 # 171 samples
train_size = n_samples - test_size # 398 samples

# Se rodou o código do passo 1.2, as variáveis X_train, X_test, Y_train, Y_test estão prontas.
# Dicionário para armazenar modelos treinados
models = {}
# Dicionário para armazenar resultados de acurácia
results = {}

print("--- 1. Inicializando Modelos ---")

# 1. Regressão Logística
models['LogisticRegression'] = LogisticRegression(max_iter=500, random_state=42)

# 2. SVM Linear
models['SVC_Linear'] = SVC(kernel='linear', random_state=42)

# 3. SVM Polinomial (Grau 2)
models['SVC_Poly_Degree_2'] = SVC(kernel='poly', degree=2, random_state=42)

# 4. SVM Polinomial (Grau 3)
models['SVC_Poly_Degree_3'] = SVC(kernel='poly', degree=3, random_state=42)

print(f"Total de modelos a serem treinados: {len(models)}")
print("-" * 50)

# --- 2. Treinamento e Avaliação em Loop ---

for name, model in models.items():
    print(f"Treinando modelo: {name}...")

    # 2.1. Treinamento
    model.fit(X_train, Y_train) 

    # 2.2. Previsão no conjunto de Teste
    y_pred = model.predict(X_test)

    # 2.3. Avaliação da Acurácia
    accuracy = accuracy_score(Y_test, y_pred) # Corrigido de y_test para Y_test
    results[name] = accuracy

    print(f"-> Treinamento concluído. Acurácia no Teste: {accuracy:.4f}")

    # 2.4. Salvando o modelo na memória (no dicionário 'models')
    # O modelo treinado já está salvo no dicionário 'models'
    print(f"-> Modelo {name} salvo no dicionário 'models'.")
    print("-" * 50)


# --- 3. Resultados Consolidados ---

print("--- 3. Resultados Consolidados de Acurácia ---")

results_df = pd.DataFrame(results.items(), columns=['Modelo', 'Acurácia'])
results_df = results_df.sort_values(by='Acurácia', ascending=False).reset_index(drop=True)

print(results_df.to_markdown(index=False))

best_model_name = results_df.iloc[0]['Modelo']
best_accuracy = results_df.iloc[0]['Acurácia']

print(f"\n✅ O modelo com maior acurácia no conjunto de teste é o **{best_model_name}** com **{best_accuracy:.4f}**.")

"""## 4) Avaliação

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

X_train = X_train_scaled
X_test = X_test_scaled

# --- 2. TREINAMENTO DOS MODELOS ---

models = {}

models['LogisticRegression'] = LogisticRegression(max_iter=500, random_state=42)
models['SVC_Linear'] = SVC(kernel='linear', random_state=42, probability=True)
models['SVC_Poly_Degree_2'] = SVC(kernel='poly', degree=2, random_state=42, probability=True)
models['SVC_Poly_Degree_3'] = SVC(kernel='poly', degree=3, random_state=42, probability=True)

print("--- 🧠 Treinamento dos Modelos Iniciado ---")
for name, model in models.items():
    model.fit(X_train, Y_train)
    print(f"✅ Modelo {name} treinado.")
print("-" * 50)


# --- 3. AVALIAÇÃO DOS MODELOS (Etapa 4 + Opcional: ROC/AUC) ---

all_metrics = []
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.50)')

for name, model in models.items():
    # Previsão das classes e das probabilidades
    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Métricas Básicas ---
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred, target_names=['Benigno (0)', 'Maligno (1)'], output_dict=True)

    acc = accuracy_score(Y_test, y_pred)
    prec = report['Maligno (1)']['precision']
    rec = report['Maligno (1)']['recall']
    f1 = report['Maligno (1)']['f1-score']

    # --- Curva ROC e AUC (Opcional) ---
    fpr, tpr, thresholds = roc_curve(Y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plotar a Curva ROC
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

    # Armazenar para a tabela comparativa
    all_metrics.append({
        'Modelo': name,
        'Acurácia': acc,
        'Precisão': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC': roc_auc
    })

    # Impressão da Matriz de Confusão e Métricas
    print(f"\n\n======== 📊 Avaliação Detalhada: {name} ========")
    cm_df = pd.DataFrame(cm, index=['Real B (0)', 'Real M (1)'], columns=['Previsto B (0)', 'Previsto M (1)'])
    print("\n1. Matriz de Confusão:")
    print(cm_df.to_markdown())
    print(f"\n2. Acurácia: {acc:.4f}")
    print(f"3. Precisão (Maligno/1): {prec:.4f}")
    print(f"4. Recall (Maligno/1): {rec:.4f}")
    print(f"5. F1-score (Maligno/1): {f1:.4f}")
    print(f"6. AUC (Area Under the Curve): {roc_auc:.4f}")


# --- 4. Tabela Comparativa de Desempenho (Incluindo AUC) ---

print("\n\n" + "=" * 80)
print("             🏆 Tabela Comparativa de Desempenho (Acurácia, Precisão, Recall, F1, AUC)")
print("=" * 80)

results_df = pd.DataFrame(all_metrics)
results_df = results_df.sort_values(by='AUC', ascending=False).reset_index(drop=True)

print(results_df.to_markdown(index=True, floatfmt=".4f"))


# --- 5. Visualização da Curva ROC ---
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('Curva ROC Comparativa dos Modelos')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("\n**Nota:** A AUC mede a capacidade de distinção do modelo, onde 1.0 é perfeito.")