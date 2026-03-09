import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# --------------------------------------------------
# Leitura da imagem binária
# --------------------------------------------------
def ler_binaria(path, limiar=127):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir {path}")

    # binariza
    _, bin_img = cv2.threshold(img, limiar, 255, cv2.THRESH_BINARY)

    # como os objetos são pretos no original, invertemos:
    # objeto -> 255
    # fundo  -> 0
    bin_img = 255 - bin_img

    # converte para 0 e 1 para facilitar
    bin_img = (bin_img > 0).astype(np.uint8)

    return bin_img


# --------------------------------------------------
# Busca em largura para encontrar um componente
# --------------------------------------------------
def bfs_componente(bin_img, visitado, li, ci):
    h, w = bin_img.shape
    fila = deque()
    fila.append((li, ci))
    visitado[li, ci] = True

    pixels = []
    
    vizinhos = [(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1)]

    while fila:
        l, c = fila.popleft()
        pixels.append((l, c))

        for dl, dc in vizinhos:
            nl, nc = l + dl, c + dc

            if 0 <= nl < h and 0 <= nc < w:
                if bin_img[nl, nc] == 1 and not visitado[nl, nc]:
                    visitado[nl, nc] = True
                    fila.append((nl, nc))

    return pixels


# --------------------------------------------------
# Monta a máscara mínima de um componente
# --------------------------------------------------
def recortar_componente(pixels):
    linhas = [p[0] for p in pixels]
    colunas = [p[1] for p in pixels]

    lmin, lmax = min(linhas), max(linhas)
    cmin, cmax = min(colunas), max(colunas)

    h = lmax - lmin + 1
    w = cmax - cmin + 1

    mascara = np.zeros((h, w), dtype=np.uint8)

    for l, c in pixels:
        mascara[l - lmin, c - cmin] = 1

    return mascara, lmin, cmin, lmax, cmax


# --------------------------------------------------
# BFS em uma região de fundo dentro do recorte
# --------------------------------------------------
def bfs_fundo(mask, visitado, li, ci):
    h, w = mask.shape
    fila = deque()
    fila.append((li, ci))
    visitado[li, ci] = True

    toca_borda = False
    pixels_fundo = []

    vizinhos = [(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1)]

    while fila:
        l, c = fila.popleft()
        pixels_fundo.append((l, c))

        if l == 0 or l == h - 1 or c == 0 or c == w - 1:
            toca_borda = True

        for dl, dc in vizinhos:
            nl, nc = l + dl, c + dc

            if 0 <= nl < h and 0 <= nc < w:
                if mask[nl, nc] == 0 and not visitado[nl, nc]:
                    visitado[nl, nc] = True
                    fila.append((nl, nc))

    return pixels_fundo, toca_borda


# --------------------------------------------------
# Conta furos manualmente
# --------------------------------------------------
def contar_furos_componente_manual(mascara):
    h, w = mascara.shape
    visitado = np.zeros((h, w), dtype=bool)
    furos = 0

    for i in range(h):
        for j in range(w):
            # procura regiões de fundo
            if mascara[i, j] == 0 and not visitado[i, j]:
                _, toca_borda = bfs_fundo(mascara, visitado, i, j)

                # se não toca borda, é furo
                if not toca_borda:
                    furos += 1

    return furos


# --------------------------------------------------
# Rotula todos os componentes e pinta por nº de furos
# --------------------------------------------------
def pintar_componentes_por_furos_manual(bin_img):
    h, w = bin_img.shape
    visitado = np.zeros((h, w), dtype=bool)

    # saída branca
    out = np.ones((h, w, 3), dtype=np.uint8) * 255

    componentes_info = []
    rotulo = 0

    for i in range(h):
        for j in range(w):
            if bin_img[i, j] == 1 and not visitado[i, j]:
                rotulo += 1

                # encontra pixels do componente
                pixels = bfs_componente(bin_img, visitado, i, j)

                # recorta componente
                mascara, lmin, cmin, lmax, cmax = recortar_componente(pixels)

                # conta furos manualmente
                furos = contar_furos_componente_manual(mascara)

                # escolhe cor
                if furos == 0:
                    cor = [255, 0, 0]   # vermelho
                elif furos == 1:
                    cor = [0, 255, 0]   # verde
                else:
                    cor = [0, 0, 255]   # azul

                # pinta na saída
                for l, c in pixels:
                    out[l, c] = cor

                componentes_info.append({
                    "rotulo": rotulo,
                    "area": len(pixels),
                    "furos": furos,
                    "bbox": (lmin, cmin, lmax, cmax)
                })

    return out, componentes_info


# --------------------------------------------------
# Teste
# --------------------------------------------------
for nome in ["images/c2.bmp", "images/c3.bmp"]:
    img_bin = ler_binaria(nome)
    img_color, infos = pintar_componentes_por_furos_manual(img_bin)

    print(f"\nResultados para {nome}")
    for info in infos:
        print(
            f"Componente {info['rotulo']}: "
            f"área = {info['area']}, "
            f"furos = {info['furos']}, "
            f"bbox = {info['bbox']}"
        )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"{nome} binária")
    plt.imshow(img_bin, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"{nome} colorida")
    plt.imshow(img_color)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"outputs/ex3_extra_{nome.split('/')[-1]}.png")
    plt.show()