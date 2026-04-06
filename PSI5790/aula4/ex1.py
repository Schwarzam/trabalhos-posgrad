"""
Faça um programa que efetua a sequência das
seguintes operações:
1) Recebe as imagens janei.pgm e janei-1.pgm como amostras de treinamento (AX, AY) e cria um
filtro pelo aprendizado de máquina.
2) Aplica o filtro aprendido na imagem julho.pgm (QX) gerando uma imagem semelhante a julho-
p1.pgm (QP).
3) Filtra essa imagem com filtro mediano adequado.
4) Sobrepõe a imagem filtrada à imagem original, obtendo uma imagem semelhante à julho-c1.png.
Dica 1: Usei filtro 3×3 e usei FlaNN (você pode usar outras técnicas).
Dica 2: Para sobrepor máscara vermelha, deixei componente R da imagem de saída 255.
"""

import os
import cv2
import numpy as np


def extract_patches_and_targets(img_x, img_y, ksize=3):
    """
    Extrai janelas ksize x ksize de img_x e o pixel central correspondente de img_y.
    Retorna:
        X: matriz (n_amostras, ksize*ksize)
        y: vetor (n_amostras,)
    """
    pad = ksize // 2
    h, w = img_x.shape

    X = []
    y = []

    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            patch = img_x[i - pad:i + pad + 1, j - pad:j + pad + 1]
            X.append(patch.flatten())
            y.append(img_y[i, j])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.uint8)
    return X, y


def apply_learned_filter_flann(train_X, train_y, query_img, ksize=3):
    """
    Aplica o filtro aprendido usando vizinho mais próximo via FLANN.
    Para cada patch da imagem de consulta, encontra o patch de treino mais parecido
    e usa o valor central correspondente aprendido.
    """
    pad = ksize // 2
    h, w = query_img.shape

    # saída inicial
    out = np.zeros((h, w), dtype=np.uint8)

    # FLANN para dados float32
    index_params = dict(algorithm=1, trees=5)   # KDTree
    search_params = dict(checks=50)
    flann = cv2.flann_Index(train_X, index_params)

    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            patch = query_img[i - pad:i + pad + 1, j - pad:j + pad + 1]
            q = patch.flatten().astype(np.float32).reshape(1, -1)

            idx, dists = flann.knnSearch(q, 1, params=search_params)
            nearest_idx = idx[0, 0]

            out[i, j] = train_y[nearest_idx]

    return out


def overlay_red_mask(gray_img, mask_bin):
    """
    Sobrepõe máscara vermelha à imagem original.
    Onde mask_bin > 0, coloca canal R = 255.
    """
    color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # OpenCV usa BGR; vermelho = (0,0,255)
    color[mask_bin > 0, 2] = 255

    return color


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1) Leitura das imagens
    ax = cv2.imread("inputs/janei.pgm", cv2.IMREAD_GRAYSCALE)
    ay = cv2.imread("inputs/janei-1.pgm", cv2.IMREAD_GRAYSCALE)
    qx = cv2.imread("inputs/julho.pgm", cv2.IMREAD_GRAYSCALE)

    if ax.shape != ay.shape:
        raise ValueError("janei.pgm e janei-1.pgm devem ter o mesmo tamanho")

    # 2) Aprende filtro com patches 3x3
    train_X, train_y = extract_patches_and_targets(ax, ay, ksize=7)

    # 3) Aplica o filtro aprendido em julho.pgm -> QP
    qp = apply_learned_filter_flann(train_X, train_y, qx, ksize=7)

    # binarizar QP para ficar mais parecido com a figura esperada
    _, qp_bin = cv2.threshold(qp, 127, 255, cv2.THRESH_BINARY_INV)

    # 4) Filtra com mediana
    qp_med = cv2.medianBlur(qp_bin, 13)

    # 5) Sobrepõe a máscara em vermelho na original
    overlay = overlay_red_mask(qx, qp_med)

    # 6) Salva saídas
    cv2.imwrite("outputs/julho-p1.png", qp_bin)
    cv2.imwrite("outputs/julho-p1-med.png", qp_med)
    cv2.imwrite("outputs/julho-c1.png", overlay)


if __name__ == "__main__":
    main()