import math
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *

import adss


# ============================================================
# CONFIG
# ============================================================
BASE_URL = "https://ai-scope.cbpf.br/"
USERNAME = ""
PASSWORD = ""

QUERY = """
select top 80000
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
where g.phot_g_mean_mag < 15.8
  and d.r_med_geo is not null
  and d.r_med_geo < 500
"""

SCREEN_W = 1600
SCREEN_H = 950

FOVY = 60.0
NEAR = 0.02
FAR = 10000.0

# Bigger visual scale so movement feels more obvious
DISTANCE_SCALE = 0.01

# Stars
POINT_SIZE_BASE = 3.5
POINT_SIZE_BRIGHT = 8.0

# Chunking
CHUNK_SIZE = 2.0
LOAD_RADIUS_CHUNKS = 4

# Camera feel
MOUSE_SENSITIVITY = 0.12
ACCEL = 0.045
FRICTION = 0.88
MAX_SPEED = 0.22

# Picking
PICK_RADIUS_PX = 18

# Earth
EARTH_RADIUS = 0.13

# Background
N_BG_STARS = 3500
BG_RADIUS = 2500.0

# Travel
TRAVEL_STOP_DISTANCE = 0.22
TRAVEL_SPEED = 0.055

# HUD button
BUTTON_W = 220
BUTTON_H = 36

# Minimap
MINIMAP_SIZE = 220
MINIMAP_PADDING = 16
MINIMAP_RANGE = 12.0


# ============================================================
# HELPERS
# ============================================================
def spherical_to_cartesian(ra_deg, dec_deg, dist_pc):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    x = dist_pc * np.cos(dec) * np.cos(ra)
    y = dist_pc * np.sin(dec)
    z = dist_pc * np.cos(dec) * np.sin(ra)
    return x, y, z


def bp_rp_to_rgb_scalar(bp_rp):
    if pd.isna(bp_rp):
        return (1.0, 1.0, 1.0)

    x = max(-0.5, min(3.5, float(bp_rp)))

    if x < 0.0:
        t = (x + 0.5) / 0.5
        return (0.7 + 0.3 * t, 0.8 + 0.2 * t, 1.0)
    elif x < 0.8:
        t = x / 0.8
        return (1.0, 1.0, 1.0 - 0.08 * t)
    elif x < 1.6:
        t = (x - 0.8) / 0.8
        return (1.0, 1.0 - 0.12 * t, 0.92 - 0.18 * t)
    elif x < 2.4:
        t = (x - 1.6) / 0.8
        return (1.0, 0.86 - 0.18 * t, 0.72 - 0.18 * t)
    else:
        t = (x - 2.4) / 1.1
        return (1.0, 0.68 - 0.16 * t, 0.52 - 0.16 * t)


def mag_to_size_scalar(gmag):
    if pd.isna(gmag):
        return POINT_SIZE_BASE
    t = 1.0 - np.clip((float(gmag) - 5.0) / 11.0, 0.0, 1.0)
    return POINT_SIZE_BASE + t * (POINT_SIZE_BRIGHT - POINT_SIZE_BASE)


def clamp_norm(v, max_norm):
    n = np.linalg.norm(v)
    if n > max_norm and n > 0:
        return v / n * max_norm
    return v


def draw_text(font, text, x, y, color=(255, 255, 255, 255)):
    surf = font.render(text, True, color[:3], None)
    data = pygame.image.tostring(surf, "RGBA", True)
    w, h = surf.get_width(), surf.get_height()
    glRasterPos2f(x, y + h)
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)


def draw_rect(x, y, w, h, rgba):
    glColor4f(*rgba)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


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


# ============================================================
# DATA MODEL
# ============================================================
@dataclass(frozen=True)
class ChunkKey:
    ix: int
    iy: int
    iz: int


class StarCatalog:
    def __init__(self, df):
        self.source_id = df["source_id"].to_numpy(np.int64)
        self.ra = df["ra"].to_numpy(np.float64)
        self.dec = df["dec"].to_numpy(np.float64)
        self.phot_g = df["phot_g_mean_mag"].to_numpy(np.float64)
        self.bp_rp = df["bp_rp"].to_numpy(np.float64)
        self.r_med_geo = df["r_med_geo"].to_numpy(np.float64)
        self.r_lo_geo = df["r_lo_geo"].to_numpy(np.float64)
        self.r_hi_geo = df["r_hi_geo"].to_numpy(np.float64)
        self.r_med_photogeo = df["r_med_photogeo"].to_numpy(np.float64)
        self.flag = df["flag"].astype(str).to_numpy()

        x, y, z = spherical_to_cartesian(self.ra, self.dec, self.r_med_geo)
        self.x = x * DISTANCE_SCALE
        self.y = y * DISTANCE_SCALE
        self.z = z * DISTANCE_SCALE

        colors = np.array([bp_rp_to_rgb_scalar(v) for v in self.bp_rp], dtype=np.float32)
        self.r = colors[:, 0]
        self.g = colors[:, 1]
        self.b = colors[:, 2]

        self.size = np.array([mag_to_size_scalar(v) for v in self.phot_g], dtype=np.float32)

        self.n = len(self.source_id)

        self.chunk_map = {}
        self._build_chunks()

    def _chunk_of(self, x, y, z):
        return ChunkKey(
            int(math.floor(x / CHUNK_SIZE)),
            int(math.floor(y / CHUNK_SIZE)),
            int(math.floor(z / CHUNK_SIZE)),
        )

    def _build_chunks(self):
        for i in range(self.n):
            key = self._chunk_of(self.x[i], self.y[i], self.z[i])
            self.chunk_map.setdefault(key, []).append(i)

        for k, v in self.chunk_map.items():
            self.chunk_map[k] = np.array(v, dtype=np.int32)

    def visible_indices(self, cam_pos):
        ck = self._chunk_of(cam_pos[0], cam_pos[1], cam_pos[2])
        idxs = []

        for dx in range(-LOAD_RADIUS_CHUNKS, LOAD_RADIUS_CHUNKS + 1):
            for dy in range(-LOAD_RADIUS_CHUNKS, LOAD_RADIUS_CHUNKS + 1):
                for dz in range(-LOAD_RADIUS_CHUNKS, LOAD_RADIUS_CHUNKS + 1):
                    k = ChunkKey(ck.ix + dx, ck.iy + dy, ck.iz + dz)
                    if k in self.chunk_map:
                        idxs.append(self.chunk_map[k])

        if not idxs:
            return np.empty(0, dtype=np.int32)

        return np.concatenate(idxs)

    def info_lines(self, i):
        return [
            f"source_id: {int(self.source_id[i])}",
            f"ra / dec: {self.ra[i]:.6f} / {self.dec[i]:.6f}",
            f"G mag: {self.phot_g[i]:.3f}",
            f"BP-RP: {self.bp_rp[i]:.3f}" if not np.isnan(self.bp_rp[i]) else "BP-RP: -",
            f"r_med_geo: {self.r_med_geo[i]:.2f} pc",
            f"r_lo_geo: {self.r_lo_geo[i]:.2f} pc" if not np.isnan(self.r_lo_geo[i]) else "r_lo_geo: -",
            f"r_hi_geo: {self.r_hi_geo[i]:.2f} pc" if not np.isnan(self.r_hi_geo[i]) else "r_hi_geo: -",
            f"r_med_photogeo: {self.r_med_photogeo[i]:.2f} pc" if not np.isnan(self.r_med_photogeo[i]) else "r_med_photogeo: -",
            f"flag: {self.flag[i]}",
        ]


# ============================================================
# FETCH
# ============================================================
def fetch_catalog():
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

    return StarCatalog(df)


# ============================================================
# CAMERA
# ============================================================
class Camera:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.55], dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.yaw = 180.0
        self.pitch = 0.0
        self.mouse_sensitivity = MOUSE_SENSITIVITY

    def forward(self):
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        fx = math.cos(pitch) * math.sin(yaw)
        fy = math.sin(pitch)
        fz = -math.cos(pitch) * math.cos(yaw)
        v = np.array([fx, fy, fz], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def right(self):
        f = self.forward()
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r = np.cross(f, up)
        n = np.linalg.norm(r)
        return r / n if n > 0 else r

    def update(self, dt, keys):
        move_dir = np.zeros(3, dtype=np.float64)
        f = self.forward()
        r = self.right()
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        if keys[pygame.K_w]:
            move_dir += f
        if keys[pygame.K_s]:
            move_dir -= f
        if keys[pygame.K_d]:
            move_dir += r
        if keys[pygame.K_a]:
            move_dir -= r
        if keys[pygame.K_r]:
            move_dir += up
        if keys[pygame.K_f]:
            move_dir -= up

        n = np.linalg.norm(move_dir)
        if n > 0:
            move_dir /= n
            self.vel += move_dir * ACCEL * max(dt * 60.0, 1.0)

        self.vel *= FRICTION
        self.vel = clamp_norm(self.vel, MAX_SPEED)
        self.pos += self.vel * max(dt * 60.0, 1.0)

    def apply_view(self):
        f = self.forward()
        c = self.pos + f
        gluLookAt(
            self.pos[0], self.pos[1], self.pos[2],
            c[0], c[1], c[2],
            0.0, 1.0, 0.0
        )


# ============================================================
# OPENGL
# ============================================================
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

    glClearColor(0.01, 0.01, 0.03, 1.0)


# ============================================================
# BACKGROUND / EARTH
# ============================================================
def make_background_stars(n=N_BG_STARS, radius=BG_RADIUS, seed=42):
    rng = np.random.default_rng(seed)
    pts = []
    cols = []
    sizes = []

    for _ in range(n):
        u = rng.uniform(-1, 1)
        phi = rng.uniform(0, 2 * np.pi)
        s = math.sqrt(max(0.0, 1 - u * u))

        x = radius * s * math.cos(phi)
        y = radius * u
        z = radius * s * math.sin(phi)

        pts.append((x, y, z))
        b = rng.uniform(0.55, 1.0)
        cols.append((b, b, b))
        sizes.append(rng.uniform(1.0, 2.0))

    return np.array(pts), np.array(cols), np.array(sizes)


def draw_background(bg_pts, bg_cols, bg_sizes, cam_pos):
    glPushMatrix()
    glTranslatef(cam_pos[0], cam_pos[1], cam_pos[2])

    unique_sizes = np.unique(np.round(bg_sizes, 1))
    for s in unique_sizes:
        mask = np.isclose(bg_sizes, s, atol=0.05)
        pts = bg_pts[mask]
        cols = bg_cols[mask]

        glPointSize(float(s))
        glBegin(GL_POINTS)
        for (x, y, z), (r, g, b) in zip(pts, cols):
            glColor3f(r, g, b)
            glVertex3f(x, y, z)
        glEnd()

    glPopMatrix()


def draw_earth():
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)

    glPushMatrix()
    glTranslatef(0.0, 0.0, 0.0)

    glColor3f(0.08, 0.28, 0.70)
    gluSphere(quad, EARTH_RADIUS, 42, 42)

    # simple land-ish blobs
    glColor3f(0.20, 0.62, 0.25)
    for ox, oy, oz, s in [
        (0.04, 0.03, 0.09, 0.030),
        (-0.05, -0.01, 0.08, 0.028),
        (0.03, -0.05, -0.10, 0.024),
        (-0.02, 0.06, -0.07, 0.020),
    ]:
        glPushMatrix()
        glTranslatef(ox, oy, oz)
        gluSphere(quad, s, 18, 18)
        glPopMatrix()

    glPopMatrix()
    gluDeleteQuadric(quad)


def draw_axes(length=0.25):
    glLineWidth(2.0)
    glBegin(GL_LINES)

    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(length, 0, 0)

    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, length, 0)

    glColor3f(0, 0.6, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, length)

    glEnd()


# ============================================================
# DRAW STARS
# ============================================================
def draw_star_points(cat, idx):
    if len(idx) == 0:
        return

    sizes = cat.size[idx]
    unique_sizes = np.unique(np.round(sizes, 1))

    for s in unique_sizes:
        mask = np.isclose(sizes, s, atol=0.05)
        sel = idx[mask]

        glPointSize(float(s))
        glBegin(GL_POINTS)
        for i in sel:
            glColor3f(cat.r[i], cat.g[i], cat.b[i])
            glVertex3f(cat.x[i], cat.y[i], cat.z[i])
        glEnd()


def draw_selected_star_as_sphere(cat, i):
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)

    glPushMatrix()
    glTranslatef(cat.x[i], cat.y[i], cat.z[i])
    glColor3f(cat.r[i], cat.g[i], cat.b[i])
    gluSphere(quad, 0.040, 22, 22)
    glPopMatrix()

    gluDeleteQuadric(quad)


def draw_selected_marker(cat, i, radius=0.065):
    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(1.6)
    glBegin(GL_LINE_LOOP)
    for k in range(48):
        a = 2.0 * math.pi * k / 48.0
        glVertex3f(
            cat.x[i] + radius * math.cos(a),
            cat.y[i] + radius * math.sin(a),
            cat.z[i]
        )
    glEnd()


# ============================================================
# PICKING
# ============================================================
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


def pick_star(cat, visible_idx, mouse_x, mouse_y):
    best_i = None
    best_d2 = None

    for i in visible_idx:
        p = project_point(cat.x[i], cat.y[i], cat.z[i])
        if p is None:
            continue

        sx, sy, sz = p
        if sz < 0 or sz > 1:
            continue

        dx = sx - mouse_x
        dy = sy - mouse_y
        d2 = dx * dx + dy * dy

        if d2 <= PICK_RADIUS_PX * PICK_RADIUS_PX:
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_i = i

    return best_i


# ============================================================
# MINIMAP
# ============================================================
def world_to_minimap(cam_pos, x, z, mx, my, size, span):
    dx = x - cam_pos[0]
    dz = z - cam_pos[2]

    nx = np.clip(dx / span, -1, 1)
    nz = np.clip(dz / span, -1, 1)

    px = mx + size / 2 + nx * (size / 2)
    py = my + size / 2 + nz * (size / 2)
    return px, py


def draw_minimap(camera, cat, visible_idx):
    mx = SCREEN_W - MINIMAP_SIZE - MINIMAP_PADDING
    my = MINIMAP_PADDING
    size = MINIMAP_SIZE

    draw_rect(mx, my, size, size, (0.0, 0.0, 0.0, 0.45))

    glColor4f(1, 1, 1, 0.25)
    glLineWidth(1.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(mx, my)
    glVertex2f(mx + size, my)
    glVertex2f(mx + size, my + size)
    glVertex2f(mx, my + size)
    glEnd()

    # earth
    ex, ey = world_to_minimap(camera.pos, 0.0, 0.0, mx, my, size, MINIMAP_RANGE)
    glColor3f(0.2, 0.7, 1.0)
    glPointSize(7.0)
    glBegin(GL_POINTS)
    glVertex2f(ex, ey)
    glEnd()

    # stars
    glPointSize(2.0)
    glBegin(GL_POINTS)
    for i in visible_idx:
        px, py = world_to_minimap(camera.pos, cat.x[i], cat.z[i], mx, my, size, MINIMAP_RANGE)
        glColor3f(cat.r[i], cat.g[i], cat.b[i])
        glVertex2f(px, py)
    glEnd()

    # camera
    glColor3f(1.0, 0.85, 0.1)
    glPointSize(8.0)
    glBegin(GL_POINTS)
    glVertex2f(mx + size / 2, my + size / 2)
    glEnd()

    # heading
    f = camera.forward()
    glColor3f(1.0, 0.85, 0.1)
    glBegin(GL_LINES)
    glVertex2f(mx + size / 2, my + size / 2)
    glVertex2f(mx + size / 2 + f[0] * 18, my + size / 2 + f[2] * 18)
    glEnd()


# ============================================================
# HUD
# ============================================================
def button_rect():
    x = 18
    y = SCREEN_H - BUTTON_H - 18
    return x, y, BUTTON_W, BUTTON_H


def draw_hud(font, camera, cat, visible_idx, selected_i, ui_mode, fps):
    begin_2d()

    # crosshair only when mouse-look mode is active
    if not ui_mode:
        glColor3f(1, 1, 1)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glVertex2f(SCREEN_W / 2 - 10, SCREEN_H / 2)
        glVertex2f(SCREEN_W / 2 + 10, SCREEN_H / 2)
        glVertex2f(SCREEN_W / 2, SCREEN_H / 2 - 10)
        glVertex2f(SCREEN_W / 2, SCREEN_H / 2 + 10)
        glEnd()

    draw_minimap(camera, cat, visible_idx)

    draw_text(font, "WASD move | R/F up-down | mouse look | M toggle mouse/UI | click star | ESC quit", 12, 10)
    draw_text(font, f"camera: ({camera.pos[0]:.2f}, {camera.pos[1]:.2f}, {camera.pos[2]:.2f})", 12, 36)
    draw_text(font, f"velocity: ({camera.vel[0]:.3f}, {camera.vel[1]:.3f}, {camera.vel[2]:.3f})", 12, 62)
    draw_text(font, f"yaw/pitch: {camera.yaw:.1f} / {camera.pitch:.1f}", 12, 88)
    draw_text(font, f"visible stars: {len(visible_idx)}", 12, 114)
    draw_text(font, f"fps: {fps:.1f}", 12, 140)

    mode_txt = "UI mode ON" if ui_mode else "Mouse-look mode"
    draw_text(font, mode_txt, 12, 166, (255, 220, 120, 255))

    bx, by, bw, bh = button_rect()
    if selected_i is not None:
        draw_rect(bx, by, bw, bh, (0.1, 0.5, 0.15, 0.85))
        draw_text(font, "Travel to selected star", bx + 16, by + 7)
    else:
        draw_rect(bx, by, bw, bh, (0.2, 0.2, 0.2, 0.6))
        draw_text(font, "Select a star first", bx + 38, by + 7)

    if selected_i is not None:
        x = 12
        y = 220
        draw_text(font, "Selected object", x, y, (255, 220, 120, 255))
        for j, line in enumerate(cat.info_lines(selected_i)):
            draw_text(font, line, x, y + 28 + j * 23)

    end_2d()


# ============================================================
# TRAVEL
# ============================================================
def update_travel(camera, cat, selected_i):
    target = np.array([cat.x[selected_i], cat.y[selected_i], cat.z[selected_i]], dtype=np.float64)

    v = target - camera.pos
    d = np.linalg.norm(v)
    if d < TRAVEL_STOP_DISTANCE:
        camera.vel[:] = 0.0
        return False

    direction = v / d
    desired = target - direction * TRAVEL_STOP_DISTANCE
    step = desired - camera.pos

    step_norm = np.linalg.norm(step)
    if step_norm > TRAVEL_SPEED:
        step = step / step_norm * TRAVEL_SPEED

    camera.pos += step
    camera.vel[:] = 0.0

    # also gently look at target
    dx, dy, dz = (target - camera.pos)
    yaw = math.degrees(math.atan2(dx, -dz))
    horiz = math.sqrt(dx * dx + dz * dz)
    pitch = math.degrees(math.atan2(dy, horiz))
    camera.yaw = yaw
    camera.pitch = max(-89.0, min(89.0, pitch))

    return True


# ============================================================
# MAIN
# ============================================================
def main():
    print("Fetching Gaia data...")
    cat = fetch_catalog()
    print(f"Loaded {cat.n} stars")

    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((SCREEN_W, SCREEN_H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Gaia 3D Explorer")

    font = pygame.font.SysFont("Consolas", 18)
    clock = pygame.time.Clock()

    setup_opengl()
    bg_pts, bg_cols, bg_sizes = make_background_stars()

    camera = Camera()

    ui_mode = False
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    selected_i = None
    travel_active = False

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        fps = clock.get_fps()

        visible_idx = cat.visible_indices(camera.pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_m:
                    ui_mode = not ui_mode
                    pygame.event.set_grab(not ui_mode)
                    pygame.mouse.set_visible(ui_mode)

            elif event.type == pygame.MOUSEMOTION and not ui_mode:
                mx, my = event.rel
                camera.yaw += mx * camera.mouse_sensitivity
                camera.pitch -= my * camera.mouse_sensitivity
                camera.pitch = max(-89.0, min(89.0, camera.pitch))

            elif event.type == pygame.MOUSEBUTTONDOWN and ui_mode:
                mx, my = pygame.mouse.get_pos()

                if event.button == 1:
                    bx, by, bw, bh = button_rect()
                    if selected_i is not None and (bx <= mx <= bx + bw and by <= my <= by + bh):
                        travel_active = True
                    else:
                        picked = pick_star(cat, visible_idx, mx, my)
                        if picked is not None:
                            selected_i = picked
                            travel_active = False

        keys = pygame.key.get_pressed()
        if not ui_mode and not travel_active:
            camera.update(dt, keys)

        if travel_active and selected_i is not None:
            travel_active = update_travel(camera, cat, selected_i)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera.apply_view()

        draw_background(bg_pts, bg_cols, bg_sizes, camera.pos)
        draw_earth()
        draw_axes()
        draw_star_points(cat, visible_idx)

        if selected_i is not None:
            draw_selected_star_as_sphere(cat, selected_i)
            draw_selected_marker(cat, selected_i)

        draw_hud(font, camera, cat, visible_idx, selected_i, ui_mode, fps)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        pygame.quit()
        sys.exit(1)