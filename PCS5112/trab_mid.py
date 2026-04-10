import math
import sys
import numpy as np
import pandas as pd
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *

import adss


BASE_URL = "https://ai-scope.cbpf.br/"
USERNAME = ""
PASSWORD = ""

QUERY = """
select top 20000
    g.source_id,
    g.ra,
    g.dec,
    g.phot_g_mean_mag,
    g.bp_rp,
    d.r_med_geo,
    d.r_lo_geo,
    d.r_hi_geo,
    d.r_med_photogeo,
    d.flag
from public.gaia_dr3 as g
join public.geodist_gaia_dr3 as d
    on g.source_id = d.source_id
where g.phot_g_mean_mag < 15
  and d.r_med_geo is not null
"""

SCREEN_W = 1400
SCREEN_H = 900
FOVY = 60.0
NEAR = 0.1
FAR = 5000.0
DISTANCE_SCALE = 0.002
POINT_SIZE_MIN = 1.0
POINT_SIZE_MAX = 6.0
AUTO_PICK_RADIUS_PX = 30
MAX_STARS = 20000


def fetch_gaia_data():
    cl = adss.ADSSClient(
        base_url=BASE_URL,
        username=USERNAME,
        password=PASSWORD,
    )

    result = cl.query_and_wait(
        query_text=QUERY,
        mode="astroql",
        file=None,
        table_name=None,
    )

    if hasattr(result, "data"):
        df = result.data
    else:
        df = pd.DataFrame(result)

    if len(df) == 0:
        raise RuntimeError("Query returned no rows.")

    return df


def spherical_to_cartesian(ra_deg, dec_deg, dist_pc):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    x = dist_pc * np.cos(dec) * np.cos(ra)
    y = dist_pc * np.sin(dec)
    z = dist_pc * np.cos(dec) * np.sin(ra)
    return x, y, z


def bp_rp_to_rgb(bp_rp):
    if np.isnan(bp_rp):
        return (1.0, 1.0, 1.0)

    x = max(-0.5, min(3.5, bp_rp))

    if x < 0.0:
        t = (x + 0.5) / 0.5
        return (0.7 + 0.3 * t, 0.8 + 0.2 * t, 1.0)
    elif x < 0.8:
        t = x / 0.8
        return (1.0, 1.0, 1.0 - 0.1 * t)
    elif x < 1.6:
        t = (x - 0.8) / 0.8
        return (1.0, 1.0 - 0.15 * t, 0.9 - 0.2 * t)
    elif x < 2.4:
        t = (x - 1.6) / 0.8
        return (1.0, 0.85 - 0.2 * t, 0.7 - 0.2 * t)
    else:
        t = (x - 2.4) / 1.1
        return (1.0, 0.65 - 0.2 * t, 0.5 - 0.2 * t)


def mag_to_size(gmag):
    if np.isnan(gmag):
        return 2.0
    t = 1.0 - np.clip((gmag - 5.0) / 11.0, 0.0, 1.0)
    return POINT_SIZE_MIN + t * (POINT_SIZE_MAX - POINT_SIZE_MIN)


def prepare_dataframe(df):
    df = df.copy()

    dist = df["r_med_geo"].to_numpy(dtype=float)

    x, y, z = spherical_to_cartesian(
        df["ra"].to_numpy(dtype=float),
        df["dec"].to_numpy(dtype=float),
        dist
    )

    df["x"] = x * DISTANCE_SCALE
    df["y"] = y * DISTANCE_SCALE
    df["z"] = z * DISTANCE_SCALE

    colors = np.array([bp_rp_to_rgb(v) for v in df["bp_rp"].to_numpy(dtype=float)])
    df["r"] = colors[:, 0]
    df["g"] = colors[:, 1]
    df["b"] = colors[:, 2]
    df["size"] = np.array([mag_to_size(v) for v in df["phot_g_mean_mag"].to_numpy(dtype=float)])

    if len(df) > MAX_STARS:
        df = df.nsmallest(MAX_STARS, "phot_g_mean_mag").copy()

    return df.reset_index(drop=True)


class Camera:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 4.0], dtype=np.float64)
        self.yaw = 0.0
        self.pitch = 0.0
        self.move_speed = 0.03
        self.mouse_sensitivity = 0.15

    def forward(self):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        fx = math.cos(pitch_rad) * math.sin(yaw_rad)
        fy = math.sin(pitch_rad)
        fz = -math.cos(pitch_rad) * math.cos(yaw_rad)

        v = np.array([fx, fy, fz], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def right(self):
        f = self.forward()
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r = np.cross(f, up)
        n = np.linalg.norm(r)
        return r / n if n > 0 else r

    def apply_view(self):
        f = self.forward()
        center = self.pos + f
        gluLookAt(
            self.pos[0], self.pos[1], self.pos[2],
            center[0], center[1], center[2],
            0.0, 1.0, 0.0
        )


def setup_opengl():
    glViewport(0, 0, SCREEN_W, SCREEN_H)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(FOVY, SCREEN_W / SCREEN_H, NEAR, FAR)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

    glClearColor(0.02, 0.02, 0.05, 1.0)


def begin_2d():
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, SCREEN_W, SCREEN_H, 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)


def end_2d():
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()

    glMatrixMode(GL_MODELVIEW)


def draw_text(font, text, x, y, color=(255, 255, 255, 255)):
    surf = font.render(text, True, color[:3], None)
    text_data = pygame.image.tostring(surf, "RGBA", True)
    w, h = surf.get_width(), surf.get_height()

    glRasterPos2f(x, y + h)
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)


def draw_crosshair():
    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    glVertex2f(SCREEN_W / 2 - 10, SCREEN_H / 2)
    glVertex2f(SCREEN_W / 2 + 10, SCREEN_H / 2)
    glVertex2f(SCREEN_W / 2, SCREEN_H / 2 - 10)
    glVertex2f(SCREEN_W / 2, SCREEN_H / 2 + 10)
    glEnd()


def draw_axes(length=0.5):
    glLineWidth(2.0)
    glBegin(GL_LINES)

    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(length, 0, 0)

    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, length, 0)

    glColor3f(0, 0.5, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, length)

    glEnd()


def draw_stars(df):
    sizes = sorted(df["size"].round(1).unique())

    for s in sizes:
        sub = df[np.isclose(df["size"], s, atol=0.05)]
        glPointSize(float(s))
        glBegin(GL_POINTS)
        for row in sub.itertuples(index=False):
            glColor3f(row.r, row.g, row.b)
            glVertex3f(row.x, row.y, row.z)
        glEnd()


def draw_selected_marker(row, radius=0.03):
    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(1.5)
    glBegin(GL_LINE_LOOP)
    for i in range(40):
        a = 2.0 * math.pi * i / 40.0
        glVertex3f(
            row.x + radius * math.cos(a),
            row.y + radius * math.sin(a),
            row.z
        )
    glEnd()


def project_point(x, y, z):
    model = glGetDoublev(GL_MODELVIEW_MATRIX)
    proj = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)

    win = gluProject(x, y, z, model, proj, viewport)
    if win is None:
        return None

    sx, sy, sz = win
    sy = SCREEN_H - sy
    return sx, sy, sz


def pick_star_near_screen_center(df):
    cx = SCREEN_W / 2
    cy = SCREEN_H / 2

    best_idx = None
    best_d2 = None

    for i, row in df.iterrows():
        p = project_point(row["x"], row["y"], row["z"])
        if p is None:
            continue

        sx, sy, sz = p
        if sz < 0 or sz > 1:
            continue

        dx = sx - cx
        dy = sy - cy
        d2 = dx * dx + dy * dy

        if d2 <= AUTO_PICK_RADIUS_PX * AUTO_PICK_RADIUS_PX:
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = i

    return best_idx


def format_info(row):
    return [
        f"source_id: {int(row['source_id'])}",
        f"ra / dec: {row['ra']:.6f} / {row['dec']:.6f}",
        f"G mag: {row['phot_g_mean_mag']:.3f}" if pd.notna(row['phot_g_mean_mag']) else "G mag: -",
        f"BP-RP: {row['bp_rp']:.3f}" if pd.notna(row['bp_rp']) else "BP-RP: -",
        f"r_med_geo: {row['r_med_geo']:.2f} pc" if pd.notna(row['r_med_geo']) else "r_med_geo: -",
        f"r_lo_geo: {row['r_lo_geo']:.2f} pc" if pd.notna(row['r_lo_geo']) else "r_lo_geo: -",
        f"r_hi_geo: {row['r_hi_geo']:.2f} pc" if pd.notna(row['r_hi_geo']) else "r_hi_geo: -",
        f"flag: {row['flag']}",
    ]


def draw_hud(font, camera, df, selected_idx):
    begin_2d()

    draw_crosshair()
    draw_text(font, "WASD move | R/F up/down | mouse look | TAB autopick | Q/E speed | ESC quit", 10, 10)
    draw_text(font, f"stars: {len(df)}", 10, 35)
    draw_text(font, f"camera: ({camera.pos[0]:.2f}, {camera.pos[1]:.2f}, {camera.pos[2]:.2f})", 10, 60)
    draw_text(font, f"yaw/pitch: {camera.yaw:.1f} / {camera.pitch:.1f}", 10, 85)
    draw_text(font, f"speed: {camera.move_speed:.3f}", 10, 110)

    if selected_idx is not None:
        row = df.iloc[selected_idx]
        lines = format_info(row)
        x = SCREEN_W - 420
        y = 20

        draw_text(font, "Selected object", x, y, (255, 220, 120, 255))
        for i, line in enumerate(lines):
            draw_text(font, line, x, y + 28 + i * 22)

    end_2d()


def main():
    print("Fetching Gaia data from ADSS...")
    df = fetch_gaia_data()
    df = prepare_dataframe(df)
    print(f"Loaded {len(df)} stars")

    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((SCREEN_W, SCREEN_H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Gaia 3D Viewer - ADSS + PyOpenGL")

    font = pygame.font.SysFont("Consolas", 18)
    clock = pygame.time.Clock()

    setup_opengl()

    camera = Camera()
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    selected_idx = None
    auto_pick = True
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    auto_pick = not auto_pick
                elif event.key == pygame.K_q:
                    camera.move_speed = max(0.005, camera.move_speed * 0.8)
                elif event.key == pygame.K_e:
                    camera.move_speed = min(1.0, camera.move_speed * 1.25)

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.rel
                camera.yaw += mx * camera.mouse_sensitivity
                camera.pitch -= my * camera.mouse_sensitivity
                camera.pitch = max(-89.0, min(89.0, camera.pitch))

        keys = pygame.key.get_pressed()
        forward = camera.forward()
        right = camera.right()
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        move = np.zeros(3, dtype=np.float64)
        if keys[pygame.K_w]:
            move += forward
        if keys[pygame.K_s]:
            move -= forward
        if keys[pygame.K_d]:
            move += right
        if keys[pygame.K_a]:
            move -= right
        if keys[pygame.K_r]:
            move += up
        if keys[pygame.K_f]:
            move -= up

        norm = np.linalg.norm(move)
        if norm > 0:
            move = move / norm
            camera.pos += move * camera.move_speed * max(1.0, dt * 60.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera.apply_view()

        draw_axes(length=0.3)
        draw_stars(df)

        if auto_pick:
            selected_idx = pick_star_near_screen_center(df)

        if selected_idx is not None:
            row = df.iloc[selected_idx]
            draw_selected_marker(row)

        draw_hud(font, camera, df, selected_idx)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        pygame.quit()
        sys.exit(1)