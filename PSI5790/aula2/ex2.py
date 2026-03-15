import os
import cv2
import numpy as np

"Inspirado na docs do openCV https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html"

def non_max_suppression_points(score_map, template_w, template_h, num_peaks=4):
    """
    Encontra os num_peaks melhores pontos no mapa de resposta,
    zerando uma região ao redor de cada pico para evitar duplicatas.
    """
    scores = score_map.copy().astype(np.float32)
    points = []

    for _ in range(num_peaks):
        _, max_val, _, max_loc = cv2.minMaxLoc(scores)
        x, y = max_loc
        points.append((x, y, max_val))

        x1 = max(0, x - template_w // 2)
        y1 = max(0, y - template_h // 2)
        x2 = min(scores.shape[1], x + template_w // 2)
        y2 = min(scores.shape[0], y + template_h // 2)

        scores[y1:y2, x1:x2] = -1.0

    return points


def main():
    os.makedirs("outputs", exist_ok=True)

    img = cv2.imread("inputs/a.png")
    template = cv2.imread("inputs/q.png")

    if img is None:
        raise FileNotFoundError("Não foi possível abrir a.png")
    if template is None:
        raise FileNotFoundError("Não foi possível abrir q.png")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    th, tw = template_gray.shape[:2]

    # Uma única chamada ao matchTemplate
    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Pega as 4 melhores ocorrências
    matches = non_max_suppression_points(result, tw, th, num_peaks=4)

    out = img.copy()

    for i, (x, y, score) in enumerate(matches, start=1):
        top_left = (x, y)
        bottom_right = (x + tw, y + th)

        cv2.rectangle(out, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(
            out,
            f"{i}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        print(f"Match {i}: x={x}, y={y}, score={score:.4f}")

    # Salva imagem final
    cv2.imwrite("outputs/ex2_urso_detectado.png", out)

    # Salva também um mapa visual da correlação
    result_norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result_norm = result_norm.astype(np.uint8)
    cv2.imwrite("outputs/ex2_mapa_correlacao.png", result_norm)
    
    

if __name__ == "__main__":
    main()