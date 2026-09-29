"""Meshcat renders of the full-order scene for figures: a logged plant state (or a taut-cable state with the
attitudes implied by the commanded thrust) is replayed in the visualization diagram, cables, commanded thrust
and payload velocity are drawn as lines, the wall as a slab, the camera is placed relative to the payload,
and the Meshcat page is screenshotted through Chromium (Playwright). Nothing here touches the simulation."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from pydrake.geometry import Box, Cylinder, MeshcatVisualizer, Rgba, Sphere, StartMeshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import DiagramBuilder

from authority_barriers.theory.params import E3, Params
from authority_barriers.theory.state import State
from .harness import attitudes_from_thrust
from .plant import QUAD_COLORS, build_plant, set_initial_state


class SceneRenderer:
    def __init__(self, p: Params, width: int = 1400, height: int = 800, meshcat=None, walls=None):
        """walls: list of Params (one per wall plane) to draw; default the single wall of p."""
        self.p, self.width, self.height = p, width, height
        self.walls = list(walls) if walls else [p]
        self.meshcat = meshcat or StartMeshcat()
        builder = DiagramBuilder()
        self.info = build_plant(p, builder, visualize=True, detail=True, wall=False)
        MeshcatVisualizer.AddToBuilder(builder, self.info.scene_graph, self.meshcat)
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_ctx = self.info.plant.GetMyMutableContextFromRoot(self.context)
        m = self.meshcat
        m.SetProperty("/Background", "visible", False)
        m.SetProperty("/Grid", "visible", False)
        m.SetProperty("/Axes", "visible", False)
        for j, pw in enumerate(self.walls):                     # each wall as a slab drawn directly in Meshcat
            n = pw.n_vec
            m.SetObject(f"/wall/{j}", Box(0.06, 30.0, 14.0), Rgba(0.80, 0.80, 0.78, 0.28))
            m.SetTransform(f"/wall/{j}", RigidTransform(RotationMatrix.MakeFromOneVector(n, 0), pw.d0 * n + 15.0 * E3 + 0.03 * n))
        self.fov = 42.0
        self._pw = self._browser = self._page = None

    # ---- drawing helpers
    def _rod(self, name: str, a: np.ndarray, b: np.ndarray, radius: float, rgba: Rgba):
        """A segment drawn as a thin cylinder (WebGL draws every line 1 px wide, which vanishes at a distance)."""
        a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; L = float(np.linalg.norm(d))
        if L < 1e-6:
            self.meshcat.Delete(name); return
        self.meshcat.SetObject(name, Cylinder(radius, L), rgba)
        self.meshcat.SetTransform(name, RigidTransform(RotationMatrix.MakeFromOneVector(d / L, 2), 0.5 * (a + b)))

    # ---- state
    def show(self, x_plant: np.ndarray | None = None, st: State | None = None, u: np.ndarray | None = None, v_L: np.ndarray | None = None,
             path: np.ndarray | None = None, ghosts=None, D: float | None = None, h: float | None = None):
        """Either the full plant state (positions and velocities) or a taut state with attitudes from the thrust; `path`
        (3 x K) draws the payload's past positions; `ghosts` (list of payload positions) draws the payload's earlier
        positions of a sequence as translucent spheres; `D` and `h` draw the distance line to the wall with the
        stopping point x_L + D n marked (the gap between the mark and the wall is H). Returns the payload and quadrotor positions."""
        plant, p = self.info.plant, self.p
        self.meshcat.Delete("/ghost"); self.meshcat.Delete("/Dmark"); self.meshcat.Delete("/hline")
        if x_plant is not None:
            plant.SetPositionsAndVelocities(self.plant_ctx, np.asarray(x_plant, float))
        else:
            set_initial_state(self.info, self.plant_ctx, st, p, attitudes=attitudes_from_thrust(u) if u is not None else None)
        self.diagram.ForcedPublish(self.context)
        x = plant.GetPositionsAndVelocities(self.plant_ctx)
        xL = self.info.pos(x, "payload")
        for i in range(p.N):
            pi = self.info.pos(x, f"quad{i}")
            c = QUAD_COLORS[i % len(QUAD_COLORS)]
            self._rod(f"/cables/{i}", xL, pi, 0.012, Rgba(0.25, 0.25, 0.25, 1.0))
            if u is not None:
                tip = pi + 1.0 * np.asarray(u[i], float) / p.f_max_arr[i]           # 1 m at full thrust
                self._rod(f"/thrust/{i}", pi, tip, 0.022, Rgba(*c, 1.0))
                self.meshcat.SetObject(f"/thrust_tip/{i}", Sphere(0.045), Rgba(*c, 1.0)); self.meshcat.SetTransform(f"/thrust_tip/{i}", RigidTransform(tip))
        vl = v_L if v_L is not None else (self.info.lin_vel(x, "payload") if x_plant is not None else st.v_L)
        self._rod("/velocity", xL, xL + 0.35 * np.asarray(vl, float), 0.02, Rgba(0.05, 0.05, 0.05, 1.0))
        if path is not None and path.shape[1] >= 2:
            self.meshcat.SetLine("/path", np.asarray(path, float), 2.0, Rgba(0.45, 0.45, 0.45, 1.0))
        for j, g in enumerate(ghosts or []):
            self.meshcat.SetObject(f"/ghost/{j}", Sphere(0.12), Rgba(0.15, 0.15, 0.15, 0.22))
            self.meshcat.SetTransform(f"/ghost/{j}", RigidTransform(np.asarray(g, float)))
        if h is not None and np.isfinite(h) and h > 0:
            self._rod("/hline", xL, xL + float(h) * p.n_vec, 0.012, Rgba(0.45, 0.45, 0.45, 1.0))
        if D is not None and np.isfinite(D) and D > 0:
            base = xL + float(D) * p.n_vec                                       # where the maneuver would stop the payload
            self._rod("/Dmark", base - 0.5 * E3, base + 0.5 * E3, 0.03, Rgba(0.48, 0.25, 0.75, 1.0))
        quads = np.array([self.info.pos(x, f"quad{i}") for i in range(p.N)])
        return {"xL": xL, "quads": quads, "centroid": quads.mean(axis=0)}

    def set_arena(self, path: np.ndarray):
        """The whole approach (3 x K payload positions) plus the wall, for the arena views."""
        p = self.p; n = p.n_vec
        pts = np.asarray(path, float)
        h = p.d0 - n @ pts                                   # distances to the wall along the path
        alt, lat = pts[2], p.r_vec @ pts
        self.arena = {"h_max": float(max(h.max(), 1.0)), "alt": (float(alt.min()), float(alt.max())), "lat": (float(lat.min()), float(lat.max()))}

    def set_wall_grid(self, center: np.ndarray, half_w: float = 6.0, half_h: float = 4.0, step: float = 1.0):
        """A 1 m grid drawn on the face of every wall around `center` (projected onto the wall), so that the wall reads
        as a wall and gives the scale of the scene."""
        self.meshcat.Delete("/wallgrid")
        for j, pw in enumerate(self.walls):
            n, r = pw.n_vec, pw.r_vec
            c = np.asarray(center, float); c = c + (pw.d0 - n @ c) * n - 0.01 * n
            self.meshcat.SetObject(f"/wall/{j}", Box(0.06, 2 * half_w, 2 * half_h), Rgba(0.93, 0.93, 0.91, 0.45))   # the slab shrinks to the gridded panel
            Rw = RotationMatrix(np.column_stack([n, r, np.cross(n, r)]))                                        # x = normal, y = lateral, z = up (or down)
            self.meshcat.SetTransform(f"/wall/{j}", RigidTransform(Rw, c + 0.04 * n))
            col = Rgba(0.50, 0.50, 0.48, 1.0)
            for k, s in enumerate(np.arange(-half_w, half_w + 1e-9, step)):
                self.meshcat.SetLine(f"/wallgrid/{j}/v{k}", np.column_stack([c + s * r - half_h * E3, c + s * r + half_h * E3]), 1.2, col)
            for k, s in enumerate(np.arange(-half_h, half_h + 1e-9, step)):
                self.meshcat.SetLine(f"/wallgrid/{j}/h{k}", np.column_stack([c - half_w * r + s * E3, c + half_w * r + s * E3]), 1.2, col)

    # ---- camera
    FOV = {"side": 42.0, "rear": 42.0, "close_side": 50.0, "close_front": 50.0, "close_low": 55.0, "close_top": 50.0, "arena_side": 45.0, "arena_high": 45.0}

    def _front(self, pt: np.ndarray, h_min: float = 0.3) -> np.ndarray:
        """The point moved back along the wall normal if it lies less than h_min in front of the wall."""
        p = self.p; h = p.d0 - p.n_vec @ pt
        return pt if h >= h_min else pt - (h_min - h) * p.n_vec

    def camera(self, view: str, focus: dict):
        p = self.p; n, r = p.n_vec, p.r_vec
        f = np.asarray(focus["xL"], float); c = np.asarray(focus["centroid"], float)
        mid = 0.5 * (f + c)                                   # between the payload and the vehicles
        h_f = p.d0 - n @ f
        ahead = lambda d: (h_f - max(h_f - d, 0.5)) * n
        if view == "side":                                    # medium, along the lateral axis; wall on the right
            cam, tgt = f + 5.2 * r + 0.9 * E3 + ahead(0.6), f + ahead(0.6) + 0.3 * E3
        elif view == "rear":                                  # medium three-quarter from behind and above
            cam, tgt = f - 4.2 * n + 3.0 * r + 2.2 * E3, f + ahead(0.8) + 0.3 * E3
        elif view == "close_side":                            # close, lateral, at the height of the vehicles
            cam, tgt = mid + 3.0 * r + 0.6 * E3, mid + 0.35 * E3
        elif view == "close_front":                           # close, from the wall side looking back at the team (three-quarter)
            cam, tgt = self._front(c + 2.2 * n + 1.4 * r + 0.6 * E3), c - 0.3 * E3
        elif view == "close_low":                             # close, from below and beside, looking up at rotors and cables
            cam, tgt = f + 2.3 * r - 0.9 * n - 0.6 * E3, c - 0.4 * E3
        elif view == "close_top":                             # close, from above and slightly behind: the formation
            cam, tgt = c - 1.2 * n + 0.4 * r + 3.2 * E3, c - 0.2 * E3
        elif view.startswith("arena"):                        # the whole approach and the wall
            A = getattr(self, "arena", None) or {"h_max": max(h_f, 1.0), "alt": (f[2], f[2]), "lat": (r @ f, r @ f)}
            L = max(A["h_max"] + 1.5, 6.0)                    # length of the approach [m]
            alt_c = 0.5 * (A["alt"][0] + A["alt"][1]) if A["alt"][1] - A["alt"][0] < 0.6 * L else f[2]   # the maneuver's altitude band, or the current one
            center = f + (h_f - 0.5 * L) * n + (alt_c - f[2] + 0.6) * E3
            if view == "arena_side":
                cam, tgt = center + 1.35 * L * r + 0.25 * L * E3, center
            else:                                             # arena_high: from behind the start, high
                cam, tgt = center - 0.9 * L * n + 0.6 * L * r + 0.7 * L * E3, center
        else:
            raise ValueError(view)
        cam, tgt = self._front(np.asarray(cam, float)), self._front(np.asarray(tgt, float), 0.2)
        self.meshcat.SetCameraPose(cam, tgt)
        self.meshcat.SetProperty("/Cameras/default/rotated/<object>", "fov", self.FOV.get(view, 42.0))
        return {"view": view, "camera": cam.tolist(), "target": tgt.tolist(), "fov": self.FOV.get(view, 42.0)}

    def frame(self, pts: np.ndarray, fov: float = 50.0, pad: float = 1.03, direction=None):
        """A fixed camera that keeps every point of `pts` (K x 3) in view: three-quarter from behind the approach,
        beside and above, at the distance at which the whole set fits the frame. Used for the sequences, where the
        camera must not move between consecutive frames. Returns the camera record."""
        p = self.p; n, r = p.n_vec, p.r_vec
        pts = np.asarray(pts, float); c = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        cands = [direction] if direction is not None else [-0.60 * n + 0.70 * r + 0.30 * E3, 0.90 * r + 0.40 * E3, -0.90 * n + 0.40 * E3]
        aspect = self.width / self.height; tv = np.tan(np.radians(fov / 2)); th = aspect * tv
        for d in cands:
            d = np.asarray(d, float) / np.linalg.norm(d)
            right = np.cross(E3, d); right /= np.linalg.norm(right); up = np.cross(d, right)
            rel = pts - c; wx = np.abs(rel @ right).max(); wy = np.abs(rel @ up).max(); wz = (rel @ d).max()
            dist = max(wx / th, wy / tv) + wz + 0.5
            for _ in range(40):                                                    # grow the distance until every point projects inside the padded frame, re-centring the target on the projected extents
                depth = dist - rel @ d
                if np.any(depth <= 0.5):
                    dist *= 1.08; continue
                px, py = rel @ right / depth, rel @ up / depth
                mx, my = 0.5 * (px.max() + px.min()), 0.5 * (py.max() + py.min())
                if abs(mx) > 0.02 or abs(my) > 0.02:                                # shift the target so that the scene is centred, then re-fit
                    c = c + (mx * right + my * up) * float(np.median(depth)); rel = pts - c; continue
                if (px.max() - px.min()) * 0.5 * pad <= th and (py.max() - py.min()) * 0.5 * pad <= tv:
                    break
                dist *= 1.06
            cam = c + dist * d
            if all(pw.d0 - pw.n_vec @ cam > 0.5 for pw in self.walls):          # the camera stays in front of every wall
                break
        self.meshcat.SetCameraPose(cam, c)
        self.meshcat.SetProperty("/Cameras/default/rotated/<object>", "fov", fov)
        return {"view": "sequence", "camera": cam.tolist(), "target": c.tolist(), "fov": fov}

    # ---- screenshots
    def _ensure_browser(self):
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        self._page = self._browser.new_page(viewport={"width": self.width, "height": self.height})
        self._page.goto(self.meshcat.web_url())
        time.sleep(2.0)
        # hide the viewer's widgets (dat.GUI controls, stats box) so that only the scene is captured
        self._page.add_style_tag(content="#stats-plot, #status-message, .dg, div.dg { display: none !important; }")

    def screenshot(self, path: Path, settle: float = 0.6, timeout_s: float = 120.0):
        self._ensure_browser()
        time.sleep(settle)
        try:
            self._page.screenshot(path=str(path), timeout=1000.0 * timeout_s)
        except Exception:                          # a starved browser on a loaded machine: re-open the page once and retry
            self.close(); self._ensure_browser(); time.sleep(max(settle, 1.5))
            self._page.screenshot(path=str(path), timeout=1000.0 * timeout_s)
        return path

    def close(self):
        if self._browser is not None:
            self._browser.close(); self._pw.stop()
            self._pw = self._browser = self._page = None


def stamp_size(width: int, text: str) -> int:
    """Font size at which `text` spans a frame of the given width, capped so that short texts do not shout."""
    return int(max(14, min(width / 14.0, 0.96 * width / (0.58 * max(len(text), 1)))))


def stamp(path: Path, text: str, size: int | None = None):
    """Write a caption bar above the frame (Meshcat has no text). The font is scaled to the frame so that the text spans
    the width and stays legible when the frame is printed small; pass `size` to give a set of frames one font size."""
    from PIL import Image, ImageDraw, ImageFont
    src = Image.open(path).convert("RGB")
    size = size or stamp_size(src.width, text)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    bar = int(1.7 * size)
    im = Image.new("RGB", (src.width, src.height + bar), (255, 255, 255)); im.paste(src, (0, bar))
    draw = ImageDraw.Draw(im)
    w = draw.textlength(text, font=font)
    draw.text((max((src.width - w) / 2, 4), int(0.3 * size)), text, fill=(20, 20, 20), font=font)
    im.save(path)


def crop_to_content(paths, margin: float = 0.04, bar: int = 0):
    """Crop a set of frames to the union bounding box of their non-white content (the same box for every frame, so a
    fixed camera stays fixed), with a relative margin; `bar` rows at the top are ignored. Returns the box."""
    import numpy as np
    from PIL import Image
    box = None
    for f in paths:
        a = np.asarray(Image.open(f).convert("L"))[bar:]
        ys, xs = np.where(a < 245)
        if xs.size == 0: continue
        b = [xs.min(), ys.min() + bar, xs.max(), ys.max() + bar]
        box = b if box is None else [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
    if box is None: return None
    w, h = box[2] - box[0], box[3] - box[1]; mx, my = int(margin * w), int(margin * h)
    im0 = Image.open(paths[0]); box = (max(box[0] - mx, 0), max(box[1] - my, 0), min(box[2] + mx, im0.width), min(box[3] + my, im0.height))
    for f in paths:
        Image.open(f).crop(box).save(f)
    return box
