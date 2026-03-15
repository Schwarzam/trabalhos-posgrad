"""
Cada uma das 12 imagens q??.jpg aparece uma única vez na imagem a.jpg, possivelmente ro-
tacionado. Faça um programa que lê as imagens a.jpg e as 12 imagens-modelos q??.jpg e gera
a imagem p.jpg indicando onde está cada uma das 12 imagens-modelos juntamente com o ân-
gulo da rotação, como na figura abaixo à direita.
Sugestão: Você pode rotacionar as imagens q??.jpg em vários ângulos e buscá-los todos na
imagem a.jpg.
"""


import os
import cv2
import numpy as np


def rotate_image_bound(image, angle_deg):
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return rotated, M


def rotate_points(points, angle_deg, w, h):
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    pts = np.array(points, dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    rotated_pts = (M @ pts_h.T).T

    return rotated_pts, new_w, new_h


def template_mask_from_rotated(rotated_bgr):
    gray = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)

    # fundo branco -> 0, objeto -> 255
    mask = np.where(gray < 250, 255, 0).astype(np.uint8)

    if np.count_nonzero(mask) == 0:
        mask = np.ones_like(gray, dtype=np.uint8) * 255

    return mask


def match_single_template(scene_gray, template_bgr, angle_step=5):
    best = {
        "score": -np.inf,
        "angle": None,
        "top_left": None,
        "size": None,
        "corners_scene": None,
    }

    h0, w0 = template_bgr.shape[:2]
    original_corners = np.array([
        [0, 0],
        [w0 - 1, 0],
        [w0 - 1, h0 - 1],
        [0, h0 - 1]
    ], dtype=np.float32)

    for angle in range(0, 360, angle_step):
        rotated_bgr, _ = rotate_image_bound(template_bgr, angle)
        rotated_gray = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)

        rh, rw = rotated_gray.shape[:2]

        if rh > scene_gray.shape[0] or rw > scene_gray.shape[1]:
            continue

        mask = template_mask_from_rotated(rotated_bgr)

        try:
            result = cv2.matchTemplate(
                scene_gray,
                rotated_gray,
                cv2.TM_CCORR_NORMED,
                mask=mask
            )
        except cv2.error:
            result = cv2.matchTemplate(
                scene_gray,
                rotated_gray,
                cv2.TM_CCOEFF_NORMED
            )

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best["score"]:
            rotated_corners, _, _ = rotate_points(original_corners, angle, w0, h0)
            rotated_corners[:, 0] += max_loc[0]
            rotated_corners[:, 1] += max_loc[1]

            best = {
                "score": float(max_val),
                "angle": angle,
                "top_left": max_loc,
                "size": (rw, rh),
                "corners_scene": rotated_corners.copy(),
            }

    return best


def draw_rotated_box(img, corners, color=(0, 0, 255), thickness=2):
    corners_int = np.round(corners).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [corners_int], isClosed=True, color=color, thickness=thickness)


def put_label(img, text, corners, color=(0, 0, 255)):
    xs = corners[:, 0]
    ys = corners[:, 1]
    x = int(np.min(xs))
    y = int(np.min(ys)) - 8
    y = max(y, 20)

    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA
    )


def process_scene(scene_path, template_paths, output_path, angle_step=5):
    scene = cv2.imread(scene_path)
    if scene is None:
        print(f"[ERRO] Não foi possível abrir {scene_path}")
        return

    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    output = scene.copy()

    print(f"\nProcessando {scene_path}")

    for template_path in template_paths:
        template = cv2.imread(template_path)
        if template is None:
            print(f"[AVISO] Não foi possível abrir {template_path}")
            continue

        best = match_single_template(scene_gray, template, angle_step=angle_step)

        if best["angle"] is None:
            print(f"[AVISO] Nenhum match encontrado para {template_path}")
            continue

        draw_rotated_box(output, best["corners_scene"], color=(0, 0, 255), thickness=2)
        put_label(
            output,
            f"{os.path.basename(template_path)} | {best['angle']}°",
            best["corners_scene"],
            color=(0, 0, 255)
        )

        print(
            f"{os.path.basename(template_path)} -> "
            f"score={best['score']:.4f}, "
            f"angulo={best['angle']}°, "
            f"pos={best['top_left']}"
        )

    cv2.imwrite(output_path, output)
    print(f"Salvo em {output_path}")


def main():
    os.makedirs("outputs/extra", exist_ok=True)

    scene_paths = [f"inputs/extra/a{i}.jpg" for i in range(1, 9)]
    template_paths = [f"inputs/extra/q{i:02d}.jpg" for i in range(1, 13)]

    for idx, scene_path in enumerate(scene_paths, start=1):
        output_path = os.path.join("outputs/extra", f"p{idx}.jpg")
        process_scene(scene_path, template_paths, output_path, angle_step=5)


if __name__ == "__main__":
    main()