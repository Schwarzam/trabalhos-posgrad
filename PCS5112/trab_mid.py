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

import os
import json
import time
import gzip
import hashlib
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
BASE_URL = "https://ai-scope.cbpf.br/"
USERNAME = ""
PASSWORD = ""

QUERY = """
select 
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
where d.r_med_geo < 40
"""

CACHE_DIR = Path("./cache_gaia")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USE_CACHE = True
FORCE_REFRESH = False

SCREEN_W = 1200
SCREEN_H = 720

FOVY = 60.0
NEAR = 0.02
FAR = 10000.0

# Bigger visual scale so movement feels more obvious
DISTANCE_SCALE = 1

# Stars
POINT_SIZE_BASE = 3.5
POINT_SIZE_BRIGHT = 8.0

# Chunking
CHUNK_SIZE = 2.0
LOAD_RADIUS_CHUNKS = 4

# Camera feel
MOUSE_SENSITIVITY = 0.12
ACCEL = 0.0045
FRICTION = 0.88
MAX_SPEED = 0.01

# Picking
PICK_RADIUS_PX = 18

# Earth
EARTH_RADIUS = 0.01

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
        self.source_id = pd.to_numeric(df["source_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)

        self.ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        self.dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        self.phot_g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        self.bp_rp = pd.to_numeric(df["bp_rp"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        self.r_med_geo = pd.to_numeric(df["r_med_geo"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        self.r_lo_geo = pd.to_numeric(df["r_lo_geo"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        self.r_hi_geo = pd.to_numeric(df["r_hi_geo"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        self.r_med_photogeo = pd.to_numeric(df["r_med_photogeo"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)

        self.flag = df["flag"].astype("string").fillna("").to_numpy()

        valid = (
            np.isfinite(self.ra) &
            np.isfinite(self.dec) &
            np.isfinite(self.r_med_geo)
        )

        self.source_id = self.source_id[valid]
        self.ra = self.ra[valid]
        self.dec = self.dec[valid]
        self.phot_g = self.phot_g[valid]
        self.bp_rp = self.bp_rp[valid]
        self.r_med_geo = self.r_med_geo[valid]
        self.r_lo_geo = self.r_lo_geo[valid]
        self.r_hi_geo = self.r_hi_geo[valid]
        self.r_med_photogeo = self.r_med_photogeo[valid]
        self.flag = self.flag[valid]

        x, y, z = spherical_to_cartesian(self.ra, self.dec, self.r_med_geo)
        self.x = (x * DISTANCE_SCALE).astype(np.float32)
        self.y = (y * DISTANCE_SCALE).astype(np.float32)
        self.z = (z * DISTANCE_SCALE).astype(np.float32)

        colors = np.array([bp_rp_to_rgb_scalar(v) for v in self.bp_rp], dtype=np.float32)
        self.r = colors[:, 0]
        self.g = colors[:, 1]
        self.b = colors[:, 2]

        self.size = np.array([mag_to_size_scalar(v) for v in self.phot_g], dtype=np.float32)

        self.positions = np.column_stack([self.x, self.y, self.z]).astype(np.float32)
        self.colors = np.column_stack([self.r, self.g, self.b]).astype(np.float32)

        # bucket point sizes so OpenGL can draw each bucket in one call
        self.size_bucket = np.clip(np.round(self.size).astype(np.int32), 1, 32)

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
            f"G mag: {self.phot_g[i]:.3f}" if np.isfinite(self.phot_g[i]) else "G mag: -",
            f"BP-RP: {self.bp_rp[i]:.3f}" if np.isfinite(self.bp_rp[i]) else "BP-RP: -",
            f"r_med_geo: {self.r_med_geo[i]:.2f} pc" if np.isfinite(self.r_med_geo[i]) else "r_med_geo: -",
            f"r_lo_geo: {self.r_lo_geo[i]:.2f} pc" if np.isfinite(self.r_lo_geo[i]) else "r_lo_geo: -",
            f"r_hi_geo: {self.r_hi_geo[i]:.2f} pc" if np.isfinite(self.r_hi_geo[i]) else "r_hi_geo: -",
            f"r_med_photogeo: {self.r_med_photogeo[i]:.2f} pc" if np.isfinite(self.r_med_photogeo[i]) else "r_med_photogeo: -",
            f"flag: {self.flag[i]}",
        ]


# ============================================================
# FETCH
# ============================================================
def query_cache_key():
    payload = {
        "base_url": BASE_URL,
        "query": QUERY.strip(),
    }
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()

def query_cache_paths():
    key = query_cache_key()
    return {
        "data": CACHE_DIR / f"{key}.pkl.gz",
        "meta": CACHE_DIR / f"{key}.json",
    }

def load_cached_dataframe():
    paths = query_cache_paths()
    if not paths["data"].exists():
        return None

    print(f"Loading cached query: {paths['data']}")
    return pd.read_pickle(paths["data"], compression="gzip")

def save_cached_dataframe(df):
    paths = query_cache_paths()
    df.to_pickle(paths["data"], compression="gzip")

    meta = {
        "created_at_unix": time.time(),
        "rows": int(len(df)),
        "query": QUERY.strip(),
        "base_url": BASE_URL,
    }
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved cache: {paths['data']}")
    
def fetch_catalog():
    if USE_CACHE and not FORCE_REFRESH:
        df = load_cached_dataframe()
        if df is not None and len(df) > 0:
            return StarCatalog(df)

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

    num_cols = [
        "source_id", "ra", "dec", "phot_g_mean_mag", "bp_rp",
        "r_med_geo", "r_lo_geo", "r_hi_geo", "r_med_photogeo"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if USE_CACHE:
        save_cached_dataframe(df)

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

    u = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(np.maximum(0.0, 1 - u * u))

    x = radius * s * np.cos(phi)
    y = radius * u
    z = radius * s * np.sin(phi)

    pts = np.column_stack([x, y, z]).astype(np.float32)

    b = rng.uniform(0.55, 1.0, n).astype(np.float32)
    cols = np.column_stack([b, b, b]).astype(np.float32)

    sizes = np.clip(np.round(rng.uniform(1.0, 2.5, n)).astype(np.int32), 1, 4)

    return pts, cols, sizes


def draw_background(bg_pts, bg_cols, bg_sizes, cam_pos):
    glPushMatrix()
    glTranslatef(cam_pos[0], cam_pos[1], cam_pos[2])

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)

    try:
        for s in np.unique(bg_sizes):
            mask = (bg_sizes == s)
            pts = bg_pts[mask]
            cols = bg_cols[mask]

            glPointSize(float(s))
            glVertexPointer(3, GL_FLOAT, 0, pts)
            glColorPointer(3, GL_FLOAT, 0, cols)
            glDrawArrays(GL_POINTS, 0, len(pts))
    finally:
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

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

    pos = cat.positions[idx]
    col = cat.colors[idx]
    size_buckets = cat.size_bucket[idx]

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)

    try:
        glVertexPointer(3, GL_FLOAT, 0, pos)
        glColorPointer(3, GL_FLOAT, 0, col)

        for s in np.unique(size_buckets):
            mask = (size_buckets == s)
            count = int(mask.sum())
            if count == 0:
                continue

            glPointSize(float(s))
            sub_pos = pos[mask]
            sub_col = col[mask]

            glVertexPointer(3, GL_FLOAT, 0, sub_pos)
            glColorPointer(3, GL_FLOAT, 0, sub_col)
            glDrawArrays(GL_POINTS, 0, len(sub_pos))
    finally:
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)


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

    ex, ey = world_to_minimap(camera.pos, 0.0, 0.0, mx, my, size, MINIMAP_RANGE)
    glColor3f(0.2, 0.7, 1.0)
    glPointSize(7.0)
    glBegin(GL_POINTS)
    glVertex2f(ex, ey)
    glEnd()

    if len(visible_idx) > 1500:
        step = max(1, len(visible_idx) // 1500)
        mini_idx = visible_idx[::step]
    else:
        mini_idx = visible_idx

    glPointSize(2.0)
    glBegin(GL_POINTS)
    for i in mini_idx:
        px, py = world_to_minimap(camera.pos, cat.x[i], cat.z[i], mx, my, size, MINIMAP_RANGE)
        glColor3f(cat.r[i], cat.g[i], cat.b[i])
        glVertex2f(px, py)
    glEnd()

    glColor3f(1.0, 0.85, 0.1)
    glPointSize(8.0)
    glBegin(GL_POINTS)
    glVertex2f(mx + size / 2, my + size / 2)
    glEnd()

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

    print("OpenGL vendor:", glGetString(GL_VENDOR).decode())
    print("OpenGL renderer:", glGetString(GL_RENDERER).decode())
    print("OpenGL version:", glGetString(GL_VERSION).decode())

    paths = query_cache_paths()
    if paths["meta"].exists():
        try:
            with open(paths["meta"], "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"Cache rows: {meta.get('rows')}")
            print(f"Cache created: {time.ctime(meta.get('created_at_unix', 0))}")
        except Exception as e:
            print(f"Warning: could not read cache metadata: {e}")

    font = pygame.font.SysFont("Consolas", 18)
    clock = pygame.time.Clock()

    setup_opengl()
    bg_pts, bg_cols, bg_sizes = make_background_stars()

    camera = Camera()

    ui_mode = False
    show_hud = True

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

                elif event.key == pygame.K_h:
                    show_hud = not show_hud

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

        if show_hud:
            draw_hud(font, camera, cat, visible_idx, selected_i, ui_mode, fps)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()