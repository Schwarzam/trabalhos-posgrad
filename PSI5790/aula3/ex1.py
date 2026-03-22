""" 
[PSI5790 aula 3. Lição de casa #1/2. Vale 5.] [Se quiser, pode usar IA para fazer este exer-
cício. Neste caso, declare se você usou ou não IA, qual foi o modelo e qual foi o prompt de
entrada.] Corrija a deformação em perspectiva do tabuleiro de xadrez abaixo, gerando uma
imagem onde cada casa do tabuleiro é um quadrado alinhado aos eixos do sistema de coorde-
nadas. Consequentemente, o tabuleiro todo será um retângulo alinhado aos eixos do sistema
de coordenadas.
Nota: Você não precisa determinar automaticamente as esquinas do tabuleiro. Pode colocar
no seu programa, manualmente, as suas coordenadas.
"""

    
import cv2
import numpy as np
import os


def corrigir_perspectiva_imagem_inteira(
    image_path,
    output_path,
    src_pts,
    board_cols=8,
    board_rows=8,
    square_size=80,
    margin=200
):
    """
    Corrige a perspectiva da imagem inteira usando os 4 cantos do tabuleiro.

    image_path: caminho da imagem original
    output_path: caminho da imagem de saída
    src_pts: 4 cantos do tabuleiro na imagem original, na ordem:
             [sup_esq, sup_dir, inf_dir, inf_esq]
    board_cols, board_rows: dimensões do tabuleiro
    square_size: tamanho desejado de cada casa na imagem corrigida
    margin: margem ao redor do tabuleiro na imagem final
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir {image_path}")

    h_img, w_img = img.shape[:2]

    # tamanho desejado do tabuleiro retificado
    board_w = board_cols * square_size
    board_h = board_rows * square_size

    # colocamos o tabuleiro dentro de uma tela maior
    dst_pts = np.array([
        [margin, margin],
        [margin + board_w - 1, margin],
        [margin + board_w - 1, margin + board_h - 1],
        [margin, margin + board_h - 1]
    ], dtype=np.float32)

    src_pts = np.array(src_pts, dtype=np.float32)

    # homografia
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # transformamos também os 4 cantos da imagem original
    corners_img = np.array([
        [0, 0],
        [w_img - 1, 0],
        [w_img - 1, h_img - 1],
        [0, h_img - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    transformed_corners = cv2.perspectiveTransform(corners_img, H).reshape(-1, 2)

    # junta cantos da imagem transformada com os pontos do tabuleiro destino
    all_pts = np.vstack([transformed_corners, dst_pts])

    min_x = int(np.floor(np.min(all_pts[:, 0])))
    min_y = int(np.floor(np.min(all_pts[:, 1])))
    max_x = int(np.ceil(np.max(all_pts[:, 0])))
    max_y = int(np.ceil(np.max(all_pts[:, 1])))

    # translada para garantir tudo dentro da imagem final
    tx = -min_x if min_x < 0 else 0
    ty = -min_y if min_y < 0 else 0

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)

    H_final = T @ H

    out_w = max_x - min_x
    out_h = max_y - min_y

    warped = cv2.warpPerspective(img, H_final, (out_w, out_h))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, warped)

    return warped, H_final


if __name__ == "__main__":
    # EXEMPLO: substitua pelos cantos reais do seu tabuleiro
    pontos_origem = [
        [137, 43],   # canto superior esquerdo
        [323, 33],  # canto superior direito
        [350, 297],  # canto inferior direito
        [107, 296]    # canto inferior esquerdo
    ]

    img_out, H = corrigir_perspectiva_imagem_inteira(
        image_path="inputs/calib_result.jpg",
        output_path="outputs/tabuleiro_imagem_inteira_corrigida.jpg",
        src_pts=pontos_origem,
        board_cols=8,
        board_rows=8,
        square_size=100,
        margin=250
    )

    print("Imagem salva em outputs/tabuleiro_imagem_inteira_corrigida.jpg")
    print("Homografia final:")
    print(H)