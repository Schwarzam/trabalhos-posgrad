"""
Escreva um programa que, dada uma imagem co-
lorida, as coordenadas (ls, cs) do pixel-semente s e um parâmetro de tolerância t, pinta de ver-
melho todos os pixels p conectados ao pixel-semente s cuja distância euclideana no espaço
das cores RGB seja menor que a tolerância t, isto é, distância(p, s)<t.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def pintar_regiao_vermelha(imagem, ls, cs, t):
    """
    imagem: imagem colorida no formato RGB
    ls, cs: linha e coluna do pixel-semente
    t: tolerância
    """

    h, w, _ = imagem.shape

    # Cópia da imagem para saída
    saida = imagem.copy()

    # Cor do pixel-semente
    semente = imagem[ls, cs].astype(np.float32)

    # Matriz de visitados
    visitado = np.zeros((h, w), dtype=bool)

    # Fila para BFS
    fila = deque()
    fila.append((ls, cs))
    visitado[ls, cs] = True

    # Guardar os pixels da região
    regiao = []

    # Vizinhança-4: cima, baixo, esquerda, direita
    vizinhos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while fila:
        l, c = fila.popleft()

        # Distância euclidiana RGB entre pixel atual e pixel-semente
        cor_atual = imagem[l, c].astype(np.float32)
        dist = np.linalg.norm(cor_atual - semente)

        if dist < t:
            regiao.append((l, c))

            # Explora vizinhos conectados
            for dl, dc in vizinhos:
                nl, nc = l + dl, c + dc

                if 0 <= nl < h and 0 <= nc < w and not visitado[nl, nc]:
                    visitado[nl, nc] = True
                    fila.append((nl, nc))

    # Pinta a região de vermelho
    for l, c in regiao:
        saida[l, c] = [255, 0, 0]

    return saida


# =========================
# Exemplo de uso
# =========================

# OpenCV lê em BGR
img_bgr = cv2.imread("images/ex2_teste.jpeg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Pixel-semente
ls = 250
cs = 230

# Tolerância
t = 60

resultado = pintar_regiao_vermelha(img_rgb, ls, cs, t)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Imagem original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Região pintada de vermelho")
plt.imshow(resultado)
plt.axis("off")

plt.savefig("outputs/ex2_crescimento_semente.png")
plt.show()