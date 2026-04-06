"""
[PSI5790 aula 3. Lição de casa #2/2. Vale 5.] [Se quiser, pode usar IA para fazer este exer-
cício. Neste caso, declare se você usou ou não IA, qual foi o modelo e qual foi o prompt de
entrada.]
Leia arquivos irisdata.txt e iristarget.txt. Use as linhas pares da matriz lida para treinar e as li-
nhas ímpares para testar. Faça classificação usando vizinho mais próximo e árvore de decisão
e imprima as taxas de erro. Qual método teve a menor taxa de erro?
Resultado esperado:
A taxa de erro da árvore de decisão fica variando entre 5,33% e 4%:
A taxa de erro do vizinho mais próximo é sempre 4%.
Pode testar também outros métodos de aprendizado, assim como pode mudar parâmetros:
a) Regressão logística.
b) SVM.
c) Random forest.
d) Boost.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")



def taxa_erro(y_true, y_pred):
    return 100 * (1 - accuracy_score(y_true, y_pred))


def main():
    # Pula a primeira linha, que contém dimensões
    X = np.loadtxt("inputs/irisdata.txt", skiprows=1)
    y = np.loadtxt("inputs/iristarget.txt", skiprows=1).astype(int).ravel()

    print("Shape de X:", X.shape)
    print("Shape de y:", y.shape)

    # Linhas pares para treino, ímpares para teste
    X_train = X[::2]
    y_train = y[::2]

    X_test = X[1::2]
    y_test = y[1::2]

    modelos = {
        "Vizinho mais próximo (k=1)": KNeighborsClassifier(n_neighbors=1),
        "Árvore de decisão": DecisionTreeClassifier(random_state=42),
        "Regressão logística": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "Random forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Boost (AdaBoost)": AdaBoostClassifier(n_estimators=100, random_state=42),
    }

    resultados = []

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        erro = taxa_erro(y_test, y_pred)
        resultados.append((nome, erro))

        print(f"{nome}: taxa de erro = {erro:.2f}%")

    melhor = min(resultados, key=lambda x: x[1])
    print(f"\nMétodo com menor taxa de erro: {melhor[0]} ({melhor[1]:.2f}%)")


if __name__ == "__main__":
    main()