import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem em tons de cinza
img = cv2.imread("images/mickeyr.bmp", cv2.IMREAD_GRAYSCALE)

# Garante que a imagem fique binária: 0 ou 255
_, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Cópia para processar
img_clean = img_bin.copy()

h, w = img_bin.shape

# Percorre ignorando a borda
for i in range(1, h - 1):
    for j in range(1, w - 1):
        pixel = img_bin[i, j]

        # 8 vizinhos
        vizinhos = [
            img_bin[i-1, j-1], img_bin[i-1, j], img_bin[i-1, j+1],
            img_bin[i,   j-1],                 img_bin[i,   j+1],
            img_bin[i+1, j-1], img_bin[i+1, j], img_bin[i+1, j+1]
        ]

        brancos = sum(v == 255 for v in vizinhos)
        pretos  = sum(v == 0 for v in vizinhos)

        # Se o pixel é branco, mas está quase cercado de preto, é ruído branco
        if pixel == 255 and pretos >= 6:
            img_clean[i, j] = 0

        # Se o pixel é preto, mas está quase cercado de branco, é ruído preto
        elif pixel == 0 and brancos >= 6:
            img_clean[i, j] = 255

# Exibição
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img_bin, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Sem ruído")
plt.imshow(img_clean, cmap="gray")
plt.axis("off")

plt.savefig("outputs/ex1_remove_ruido.png")

plt.show()