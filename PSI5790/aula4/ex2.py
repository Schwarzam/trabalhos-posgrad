"""
[PSI5790 lição de casa #2/2. Vale 5] para classificar MNIST que atinge taxa de erro menor que 2,15% sem usar SVM ou taxa de erro menor
Implemente um programa de aprendizado de máquina clássico
que usando SVM .
Nota: Não pode usar deep learning nem rede neural convolucional. Não pode usar bibliotecas de deep learning como Keras,
Tensorflow ou PyTorch (exceto para ler MNIST).
1,85% Descreva (através de explicação falada no vídeo e também como comentários dentro do seu
programa .cpp ou .py) a taxa de erro que obteve, o tempo de processamento e as alterações feitas para
chegar ao seu programa com baixa taxa de erro.
Dica: É possível aumentar artificialmente os dados de treino deslocando as imagens originais um pixel
para as direções norte, sul, leste e oeste, obtendo um conjunto de treino 5 vezes maior que o original.
Com isso, cheguei à taxa de erro de 1,98% usando FlaNN e atingi taxa 1,78% com SVM (usando con-
junto 2 vezes maior – em vez de 5 vezes – que a original). A técnica de gerar imagens de treino artifici-
ais distorcidas chama-se “data augmentation”.
Nota: Você não é obrigado a usar imagens reduzidas para 14×14 nem é obrigado a eliminar linhas/colu-
nas brancas. Você pode copiar os conteúdos de mnist.AX, mnist.AY, mnist.QX, mnist.QY (e outros
membros da classe MNIST) para outras variáveis para que você possa alterá-las.

MELHOR RESULTADO OBTIDO:

Taxa de erro: 1.60%
Tempo de carga: 2.00 s
Tempo de augmentation: 2.39 s
Tempo de extração de atributos: 11.54 s
Tempo de indexação FLANN: 2.41 s
Tempo de classificação: 0.55 s
Tempo total: 18.87 s

"""

from sklearn.datasets import fetch_openml
import numpy as np
import time
import cv2


def load_mnist():
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.reshape(-1, 28, 28).astype(np.uint8)
    y = y.astype(np.int32)

    AX = X[:60000]
    AY = y[:60000]
    QX = X[60000:]
    QY = y[60000:]
    return AX, AY, QX, QY


def shift_image(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def bbox_crop(img):
    ys, xs = np.where(img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img.copy()
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return img[y1:y2 + 1, x1:x2 + 1]


def resize_keep_aspect(img, target=20):
    h, w = img.shape
    if h == 0 or w == 0:
        return np.zeros((target, target), dtype=np.uint8)

    if h > w:
        new_h = target
        new_w = max(1, int(round(w * target / h)))
    else:
        new_w = target
        new_h = max(1, int(round(h * target / w)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=np.uint8)

    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def center_by_mass(img):
    imgf = img.astype(np.float32)
    m = cv2.moments(imgf)
    if abs(m["m00"]) < 1e-8:
        return img.copy()

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    tx = img.shape[1] / 2.0 - cx
    ty = img.shape[0] / 2.0 - cy

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def deskew(img):
    m = cv2.moments(img.astype(np.float32))
    if abs(m["mu02"]) < 1e-8:
        return img.copy()

    skew = m["mu11"] / m["mu02"]
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew],
                    [0, 1, 0]])

    return cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def normalize_digit(img):
    """
    Pipeline geométrico:
    - crop do bounding box
    - resize para caber em 20x20
    - coloca no centro de uma imagem 28x28
    - deskew
    - centraliza pelo centro de massa
    """
    cropped = bbox_crop(img)
    small = resize_keep_aspect(cropped, target=20)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = small

    canvas = deskew(canvas)
    canvas = center_by_mass(canvas)
    return canvas


# HOG mais leve, mas melhor que pixels crus
hog = cv2.HOGDescriptor(
    _winSize=(28, 28),
    _blockSize=(14, 14),
    _blockStride=(7, 7),
    _cellSize=(7, 7),
    _nbins=9
)


def hog_feature(img):
    feat = hog.compute(img)
    return feat.reshape(-1)


def preprocess_and_extract(X):
    feats = []
    for img in X:
        norm = normalize_digit(img)
        feats.append(hog_feature(norm))
    return np.array(feats, dtype=np.float32)


def augment_with_shifts(X, y, diagonals=False):
    """
    Augmentation:
    original + N,S,L,O
    opcionalmente diagonais.
    """
    if diagonals:
        shifts = [
            (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
    else:
        shifts = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

    aug_imgs = []
    aug_y = []

    for img, label in zip(X, y):
        base = normalize_digit(img)
        for dx, dy in shifts:
            aug_imgs.append(shift_image(base, dx, dy))
            aug_y.append(label)

    return np.array(aug_imgs, dtype=np.uint8), np.array(aug_y, dtype=np.int32)


def weighted_vote(labels, dists):
    eps = 1e-8
    weights = 1.0 / (dists + eps)
    scores = {}
    for lab, w in zip(labels, weights):
        scores[lab] = scores.get(lab, 0.0) + w
    return max(scores.items(), key=lambda x: x[1])[0]


def build_flann(train_X, trees=8):
    index_params = dict(algorithm=1, trees=trees)
    return cv2.flann_Index(train_X, index_params)


def predict_flann_knn(flann, train_y, test_X, k=4, checks=128):
    preds = np.empty(len(test_X), dtype=np.int32)
    search_params = dict(checks=checks)

    for i, q in enumerate(test_X):
        q = q.reshape(1, -1).astype(np.float32)
        idx, dists = flann.knnSearch(q, k, params=search_params)

        nn_idx = idx[0]
        nn_dists = dists[0]
        nn_labels = train_y[nn_idx]

        preds[i] = weighted_vote(nn_labels, nn_dists)

    return preds


def error_rate(y_true, y_pred):
    return 100.0 * np.mean(y_true != y_pred)


def main():
    t0 = time.perf_counter()
    AX, AY, QX, QY = load_mnist()
    t_load = time.perf_counter()

    # augmentation 5x como na dica
    AX_aug, AY_aug = augment_with_shifts(AX, AY, diagonals=False)
    t_aug = time.perf_counter()

    train_X = preprocess_and_extract(AX_aug)
    test_X = preprocess_and_extract(QX)
    t_feat = time.perf_counter()

    flann = build_flann(train_X, trees=8)
    t_index = time.perf_counter()

    pred = predict_flann_knn(flann, AY_aug, test_X, k=4, checks=128)
    t_pred = time.perf_counter()

    err = error_rate(QY, pred)

    print(f"Taxa de erro: {err:.2f}%")
    print(f"Tempo de carga: {t_load - t0:.2f} s")
    print(f"Tempo de augmentation: {t_aug - t_load:.2f} s")
    print(f"Tempo de extração de atributos: {t_feat - t_aug:.2f} s")
    print(f"Tempo de indexação FLANN: {t_index - t_feat:.2f} s")
    print(f"Tempo de classificação: {t_pred - t_index:.2f} s")
    print(f"Tempo total: {t_pred - t0:.2f} s")


if __name__ == "__main__":
    main()
    
    