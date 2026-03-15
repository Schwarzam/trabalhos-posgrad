"""
Escreva um programa que usa o filtro media-
na (usando a função medianBlur do OpenCV ou o filtro implementado “manualmente”) para
filtrar a imagem ruidosa fever-1.pgm e fever-2.pgm (que se encontram dentro do arquivo “fil-
tlin.zip” diretório “textura”) obtendo as imagens limpas. Figura 8 mostra a saída esperada fil-
trando fever-2.pgm.
"""

import cv2
import matplotlib.pyplot as plt

# Lista das imagens
arquivos = ["fever-1.pgm", "fever-2.pgm"]

for nome in arquivos:
    # Lê em tons de cinza
    img = cv2.imread(f"inputs/{nome}", cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Não foi possível abrir {nome}")
        continue

    # Aplica filtro da mediana
    img_filtrada = cv2.medianBlur(img, 5) # falar do pq 5.

    # Mostra original e filtrada
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title(f"{nome} original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"{nome} filtrada")
    plt.imshow(img_filtrada, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"outputs/ex1_{nome}_filtrada.png")  # Salva a imagem filtrada
    plt.show()
