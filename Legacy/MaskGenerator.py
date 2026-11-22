# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:40:22 2026

@author: Alexander Ortmann
"""

"""
Micropattern Mask Generator
============================
Für DMD/SLM-basiertes Structured Illumination / Micropatterning.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import copy

PROJECTOR_W = 1920
PROJECTOR_H = 1080
MAX_PATCH_SIZE = 150
RECOMMENDED_PATCH = 100


@dataclass
class ProjectorConfig:
    width: int = PROJECTOR_W
    height: int = PROJECTOR_H
    offset_x: int = 85
    offset_y: int = -5
    px_per_um: float = 0.77


@dataclass
class ShapeParams:
    shape_type: str = "square"
    width: float = 10.0
    height: float = 10.0
    angle: float = 0.0
    text: str = "A"
    font_size: int = 20


@dataclass
class GridParams:
    n_cols: int = 3
    n_rows: int = 3
    gap_x: float = 30.0
    gap_y: float = 30.0
    shape: ShapeParams = field(default_factory=ShapeParams)


@dataclass
class PatchConfig:
    enabled: bool = False
    patch_size: int = RECOMMENDED_PATCH
    overlap: int = 0


@dataclass
class MaskLayer:
    name: str = "Layer 1"
    enabled: bool = True
    grid: GridParams = field(default_factory=GridParams)
    patch: PatchConfig = field(default_factory=PatchConfig)
    intensity: int = 255


# ─────────────────────────────────────────────
# KERN: MASKEN-RENDERING
# ─────────────────────────────────────────────

class MaskRenderer:
    def __init__(self, config: ProjectorConfig):
        self.config = config

    def render_layers(self, layers: List[MaskLayer]) -> np.ndarray:
        W, H = self.config.width, self.config.height
        A = np.zeros((H, W), dtype=np.uint8)
        for layer in layers:
            if layer.enabled:
                layer_mask = self.render_layer(layer)
                A = np.maximum(A, layer_mask)
        return A

    def render_layer(self, layer: MaskLayer) -> np.ndarray:
        W, H = self.config.width, self.config.height
        A = np.zeros((H, W), dtype=np.uint8)
        g = layer.grid
        s = g.shape
        cx = W // 2 + self.config.offset_x
        cy = H // 2 + self.config.offset_y

        for ix in range(g.n_cols):
            for iy in range(g.n_rows):
                spacing_x = s.width + g.gap_x
                spacing_y = s.height + g.gap_y
                col0 = cx + (ix - (g.n_cols - 1) / 2) * spacing_x - s.width / 2
                row0 = cy + (iy - (g.n_rows - 1) / 2) * spacing_y - s.height / 2
                self._draw_shape(A, s, col0, row0, layer.intensity)
        return A

    def _draw_shape(self, A: np.ndarray, s: ShapeParams, col0: float, row0: float, intensity: int):
        H, W = A.shape
        img_pil = Image.fromarray(A)
        draw = ImageDraw.Draw(img_pil)

        x0, y0 = col0, row0
        x1, y1 = col0 + s.width, row0 + s.height
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            return

        if s.shape_type in ("square", "rectangle"):
            if s.angle == 0:
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=intensity)
            else:
                pts = self._rotated_rect_pts(cx, cy, s.width, s.height, s.angle)
                draw.polygon(pts, fill=intensity)

        elif s.shape_type == "circle":
            r = min(s.width, s.height) / 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=intensity)

        elif s.shape_type == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=intensity)

        elif s.shape_type == "triangle":
            pts = self._triangle_pts(cx, cy, s.width, s.height, s.angle)
            draw.polygon(pts, fill=intensity)

        elif s.shape_type == "line":
            lw = max(1, int(s.height))
            draw.line([x0, cy, x1, cy], fill=intensity, width=lw)

        elif s.shape_type == "cross":
            arm_w = max(1, s.width // 3)
            arm_h = max(1, s.height // 3)
            draw.rectangle([x0, cy - arm_h / 2, x1, cy + arm_h / 2], fill=intensity)
            draw.rectangle([cx - arm_w / 2, y0, cx + arm_w / 2, y1], fill=intensity)

        elif s.shape_type == "ring":
            r_outer = min(s.width, s.height) / 2
            r_inner = r_outer * 0.6
            tmp = Image.new("L", (W, H), 0)
            d2 = ImageDraw.Draw(tmp)
            d2.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], fill=intensity)
            d2.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=0)
            arr = np.maximum(np.array(img_pil), np.array(tmp))
            img_pil = Image.fromarray(arr)
            draw = ImageDraw.Draw(img_pil)

        elif s.shape_type == "text":
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s.font_size)
            except Exception:
                font = ImageFont.load_default()
            tmp = Image.new("L", (W, H), 0)
            dtmp = ImageDraw.Draw(tmp)
            dtmp.text((0, 0), s.text, fill=255, font=font)
            arr = np.array(tmp)
            ys, xs = np.where(arr > 0)
            if len(xs) == 0:
                return
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            tw = max_x - min_x + 1
            th = max_y - min_y + 1
            tx = cx - tw / 2
            ty = cy - th / 2
            draw.text((tx - min_x, ty - min_y), s.text, fill=intensity, font=font)

        A[:] = np.array(img_pil)

    def _rotated_rect_pts(self, cx, cy, w, h, angle_deg):
        a = math.radians(angle_deg)
        corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        return [(dx*math.cos(a) - dy*math.sin(a) + cx,
                 dx*math.sin(a) + dy*math.cos(a) + cy) for dx, dy in corners]

    def _triangle_pts(self, cx, cy, w, h, angle_deg):
        a = math.radians(angle_deg)
        corners = [(0, -h/2), (w/2, h/2), (-w/2, h/2)]
        return [(dx*math.cos(a) - dy*math.sin(a) + cx,
                 dx*math.sin(a) + dy*math.cos(a) + cy) for dx, dy in corners]


# ─────────────────────────────────────────────
# PATCH-DATENSTRUKTUR
# ─────────────────────────────────────────────

@dataclass
class Patch:
    mask: np.ndarray          # Ausschnitt (patch_size × patch_size) aus der Gesamtmaske
    proj_mask: np.ndarray     # 1920×1080 Vollbild – Inhalt auf Projektor-Mittelpunkt (Offset) zentriert
    proj_x: int               # linke obere Ecke des Ausschnitts im Gesamtbild
    proj_y: int
    patch_idx: Tuple[int, int]
    # Mittelpunkt des tatsächlichen Inhalts dieses Patches im Gesamt-Koordinatensystem
    content_cx_px: float
    content_cy_px: float
    # Verschiebung relativ zu Patch (0,0)
    stage_dx_px: float
    stage_dy_px: float
    stage_dx_um: float
    stage_dy_um: float
    stage_dx_mm: float
    stage_dy_mm: float


# ─────────────────────────────────────────────
# PATCH-ZERLEGUNG
# ─────────────────────────────────────────────

class PatchSplitter:
    def __init__(self, config: ProjectorConfig):
        self.config = config

    def split(self, full_mask: np.ndarray, patch_cfg: PatchConfig) -> List[Patch]:
        ps   = patch_cfg.patch_size
        ov   = patch_cfg.overlap
        stride = ps - ov
        H, W = full_mask.shape

        proj_cx = W // 2 + self.config.offset_x   # Projektor-Mittelpunkt in Pixel
        proj_cy = H // 2 + self.config.offset_y

        # ── 1) Tight Bounding Box des Inhalts ────────────────────────────────
        rows = np.any(full_mask > 0, axis=1)
        cols = np.any(full_mask > 0, axis=0)
        if not rows.any():
            return []

        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        # ── 2) Symmetrisch um den Projektor-Mittelpunkt erweitern ─────────────
        half_w = max(proj_cx - c0, c1 - proj_cx)
        half_h = max(proj_cy - r0, r1 - proj_cy)

        c0s = max(0, proj_cx - half_w)
        c1s = min(W - 1, proj_cx + half_w)
        r0s = max(0, proj_cy - half_h)
        r1s = min(H - 1, proj_cy + half_h)

        # ── 3) Patch-Grid iterieren ───────────────────────────────────────────
        patches = []
        base_cx = None   # Mittelpunkt des Referenz-Patches (0,0) – wird beim ersten gesetzt
        base_cy = None

        patch_col = 0
        x = int(c0s)
        while x <= int(c1s):
            patch_row = 0
            y = int(r0s)
            while y <= int(r1s):
                x_end = min(x + ps, W)
                y_end = min(y + ps, H)

                # Ausschnitt aus der Gesamtmaske
                crop = full_mask[y:y_end, x:x_end]
                padded = np.zeros((ps, ps), dtype=np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop

                # ── Schwerpunkt des Inhalts in diesem Patch ──────────────────
                # (für leere Patches = geometrischer Mittelpunkt)
                ys_idx, xs_idx = np.where(padded > 0)
                if len(xs_idx) > 0:
                    # Mittelpunkt des tatsächlichen Inhalts im globalen System
                    ccx = x + float(xs_idx.mean())
                    ccy = y + float(ys_idx.mean())
                else:
                    ccx = x + ps / 2.0
                    ccy = y + ps / 2.0

                if base_cx is None:
                    base_cx, base_cy = ccx, ccy

                dx_px = ccx - base_cx
                dy_px = ccy - base_cy
                px_per_um = self.config.px_per_um

                # ── Proj-Maske: Inhalt auf Projektor-Mittelpunkt zentrieren ───
                proj_mask = self._make_projector_mask(
                    padded, ccx - x, ccy - y, proj_cx, proj_cy, W, H
                )

                patches.append(Patch(
                    mask=padded,
                    proj_mask=proj_mask,
                    proj_x=x,
                    proj_y=y,
                    patch_idx=(patch_col, patch_row),
                    content_cx_px=ccx,
                    content_cy_px=ccy,
                    stage_dx_px=dx_px,
                    stage_dy_px=dy_px,
                    stage_dx_um=dx_px / px_per_um,
                    stage_dy_um=dy_px / px_per_um,
                    stage_dx_mm=dx_px / px_per_um / 1000.0,
                    stage_dy_mm=dy_px / px_per_um / 1000.0,
                ))

                y += stride
                patch_row += 1
            x += stride
            patch_col += 1

        return patches

    def _make_projector_mask(
        self,
        patch_crop: np.ndarray,
        local_cx: float, local_cy: float,
        proj_cx: int, proj_cy: int,
        W: int, H: int
    ) -> np.ndarray:
        """
        Baut eine 1920×1080-Maske, in der der Patch-Inhalt so verschoben wird,
        dass sein Inhalts-Mittelpunkt auf (proj_cx, proj_cy) liegt.
        """
        out = np.zeros((H, W), dtype=np.uint8)

        # Verschiebung: Inhaltsmittelpunkt → Projektor-Mittelpunkt
        shift_x = int(round(proj_cx - local_cx))
        shift_y = int(round(proj_cy - local_cy))

        ps_h, ps_w = patch_crop.shape

        # Quell- und Zielkoordinaten mit Clipping
        src_x0, src_y0 = 0, 0
        dst_x0 = shift_x
        dst_y0 = shift_y

        if dst_x0 < 0:
            src_x0 = -dst_x0
            dst_x0 = 0
        if dst_y0 < 0:
            src_y0 = -dst_y0
            dst_y0 = 0

        src_x1 = ps_w
        src_y1 = ps_h
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        dst_x1 = min(dst_x1, W)
        dst_y1 = min(dst_y1, H)
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = patch_crop[src_y0:src_y1, src_x0:src_x1]

        return out

    def export_stage_coordinates(self, patches: List[Patch]) -> List[dict]:
        coords = []
        for p in patches:
            coords.append({
                "patch_idx": list(p.patch_idx),
                "proj_origin_px": [p.proj_x, p.proj_y],
                "content_center_px": [round(p.content_cx_px, 2), round(p.content_cy_px, 2)],
                "stage_offset_px":   [round(p.stage_dx_px, 2),   round(p.stage_dy_px, 2)],
                "stage_offset_um":   [round(p.stage_dx_um, 3),   round(p.stage_dy_um, 3)],
                "stage_offset_mm":   [round(p.stage_dx_mm, 6),   round(p.stage_dy_mm, 6)],
            })
        return coords


# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────

SHAPES = ["square", "rectangle", "circle", "ellipse", "triangle", "line", "cross", "ring", "text"]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Micropattern Mask Generator")
        self.configure(bg="#1a1a2e")
        self.geometry("1400x880")

        self.proj_cfg = ProjectorConfig()
        self.layers: List[MaskLayer] = [MaskLayer()]
        self.current_layer_idx = 0
        self.patch_cfg = PatchConfig()
        self.patches: List[Patch] = []
        self.current_mask: Optional[np.ndarray] = None

        self._build_ui()
        self._refresh_preview()

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = tk.Frame(self, bg="#16213e", width=380)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_propagate(False)

        canvas_left = tk.Canvas(left, bg="#16213e", highlightthickness=0)
        scroll_left = tk.Scrollbar(left, orient="vertical", command=canvas_left.yview)
        canvas_left.configure(yscrollcommand=scroll_left.set)
        scroll_left.pack(side="right", fill="y")
        canvas_left.pack(side="left", fill="both", expand=True)

        self.inner_left = tk.Frame(canvas_left, bg="#16213e")
        canvas_left.create_window((0, 0), window=self.inner_left, anchor="nw")
        self.inner_left.bind("<Configure>",
            lambda e: canvas_left.configure(scrollregion=canvas_left.bbox("all")))

        self._build_projector_section(self.inner_left)
        self._build_layer_section(self.inner_left)
        self._build_grid_section(self.inner_left)
        self._build_shape_section(self.inner_left)
        self._build_patch_section(self.inner_left)
        self._build_export_section(self.inner_left)

        right = tk.Frame(self, bg="#0f3460")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        tk.Label(right, text="PREVIEW", font=("Courier", 11, "bold"),
                 bg="#0f3460", fg="#e94560").grid(row=0, column=0, pady=8)

        fig_frame = tk.Frame(right, bg="#0f0f0f")
        fig_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        fig_frame.rowconfigure(0, weight=1)
        fig_frame.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(facecolor="#0f0f0f")
        self.ax.set_facecolor("#0f0f0f")
        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas_mpl.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = tk.Frame(right, bg="#0f3460")
        toolbar_frame.grid(row=2, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas_mpl, toolbar_frame)

        self.status_var = tk.StringVar(value="Bereit")
        tk.Label(right, textvariable=self.status_var, font=("Courier", 9),
                 bg="#0f3460", fg="#aaa").grid(row=3, column=0, pady=4)

    def _section_label(self, parent, text):
        tk.Label(parent, text=f"▸ {text}", font=("Courier", 10, "bold"),
                 bg="#16213e", fg="#e94560", anchor="w").pack(fill="x", padx=10, pady=(14, 2))
        tk.Frame(parent, bg="#e94560", height=1).pack(fill="x", padx=10)

    def _row(self, parent, label, widget_factory, **kw):
        f = tk.Frame(parent, bg="#16213e")
        f.pack(fill="x", padx=10, pady=2)
        tk.Label(f, text=label, width=18, anchor="w",
                 bg="#16213e", fg="#ccc", font=("Courier", 9)).pack(side="left")
        w = widget_factory(f, **kw)
        w.pack(side="left", fill="x", expand=True)
        return w

    def _build_projector_section(self, parent):
        self._section_label(parent, "PROJEKTOR")
        self.var_offset_x = self._intvar(self.proj_cfg.offset_x)
        self.var_offset_y = self._intvar(self.proj_cfg.offset_y)
        self.var_px_um    = self._dblvar(self.proj_cfg.px_per_um)
        self._row(parent, "Offset X [px]", self._spin,
                  textvariable=self.var_offset_x, from_=-500, to=500, command=self._on_change)
        self._row(parent, "Offset Y [px]", self._spin,
                  textvariable=self.var_offset_y, from_=-500, to=500, command=self._on_change)
        self._row(parent, "px / µm", self._spin,
                  textvariable=self.var_px_um, from_=0.01, to=100, increment=0.01, command=self._on_change)

        # Abgeleitete Werte anzeigen
        self.lbl_scale_info = tk.Label(parent, text="", bg="#16213e", fg="#7ecfff",
                                        font=("Courier", 8), anchor="w")
        self.lbl_scale_info.pack(fill="x", padx=10)
        self._update_scale_info()

    def _update_scale_info(self):
        try:
            pum = float(self.var_px_um.get())
            um_per_px = 1.0 / pum
            mm_per_px = um_per_px / 1000.0
            self.lbl_scale_info.config(
                text=f"  1 px = {um_per_px:.3f} µm = {mm_per_px:.6f} mm")
        except Exception:
            pass

    def _build_layer_section(self, parent):
        self._section_label(parent, "LAYER")
        f = tk.Frame(parent, bg="#16213e")
        f.pack(fill="x", padx=10, pady=4)

        self.layer_listbox = tk.Listbox(f, height=4, bg="#0f3460", fg="#fff",
                                         selectbackground="#e94560", font=("Courier", 9))
        self.layer_listbox.pack(side="left", fill="x", expand=True)
        self.layer_listbox.bind("<<ListboxSelect>>", self._on_layer_select)

        btn_f = tk.Frame(f, bg="#16213e")
        btn_f.pack(side="left", padx=4)
        for txt, cmd in [("＋", self._add_layer), ("－", self._del_layer),
                          ("↑", self._move_layer_up), ("↓", self._move_layer_down)]:
            tk.Button(btn_f, text=txt, command=cmd, bg="#e94560", fg="white",
                      font=("Courier", 9), width=2, relief="flat").pack(pady=1)

        self._refresh_layer_list()

        self.var_layer_name      = tk.StringVar()
        self.var_layer_intensity = self._intvar(255)
        self.var_layer_enabled   = tk.BooleanVar(value=True)

        self._row(parent, "Name",
                  lambda p, **kw: tk.Entry(p, textvariable=self.var_layer_name,
                                           bg="#0f3460", fg="white",
                                           insertbackground="white", font=("Courier", 9)))
        self.var_layer_name.trace_add("write", lambda *_: self._sync_layer_name())

        self._row(parent, "Intensität [0-255]", self._spin,
                  textvariable=self.var_layer_intensity, from_=0, to=255, command=self._on_change)

        f2 = tk.Frame(parent, bg="#16213e")
        f2.pack(fill="x", padx=10, pady=2)
        tk.Checkbutton(f2, text="Layer aktiv", variable=self.var_layer_enabled,
                       bg="#16213e", fg="#ccc", selectcolor="#0f3460",
                       activebackground="#16213e", font=("Courier", 9),
                       command=self._on_change).pack(side="left")

    def _build_grid_section(self, parent):
        self._section_label(parent, "GRID")
        self.var_ncols = self._intvar(3)
        self.var_nrows = self._intvar(3)
        self.var_gapx  = self._dblvar(30)
        self.var_gapy  = self._dblvar(30)
        self._row(parent, "Spalten", self._spin,
                  textvariable=self.var_ncols, from_=1, to=50, command=self._on_change)
        self._row(parent, "Zeilen", self._spin,
                  textvariable=self.var_nrows, from_=1, to=50, command=self._on_change)
        self._row(parent, "Gap X [px]", self._spin,
                  textvariable=self.var_gapx, from_=0, to=500, increment=1, command=self._on_change)
        self._row(parent, "Gap Y [px]", self._spin,
                  textvariable=self.var_gapy, from_=0, to=500, increment=1, command=self._on_change)

    def _build_shape_section(self, parent):
        self._section_label(parent, "FORM")

        self.var_shape = tk.StringVar(value="square")
        f = tk.Frame(parent, bg="#16213e")
        f.pack(fill="x", padx=10, pady=2)
        tk.Label(f, text="Form", width=18, anchor="w",
                 bg="#16213e", fg="#ccc", font=("Courier", 9)).pack(side="left")
        cb = ttk.Combobox(f, textvariable=self.var_shape, values=SHAPES,
                          state="readonly", width=14)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_change())

        self.var_width     = self._dblvar(10)
        self.var_height    = self._dblvar(10)
        self.var_angle     = self._dblvar(0)
        self.var_text      = tk.StringVar(value="A")
        self.var_font_size = self._intvar(20)

        self._row(parent, "Breite [px]", self._spin,
                  textvariable=self.var_width, from_=1, to=500, increment=1, command=self._on_change)
        self._row(parent, "Höhe [px]", self._spin,
                  textvariable=self.var_height, from_=1, to=500, increment=1, command=self._on_change)
        self._row(parent, "Rotation [°]", self._spin,
                  textvariable=self.var_angle, from_=-180, to=180, increment=5, command=self._on_change)

        f2 = tk.Frame(parent, bg="#16213e")
        f2.pack(fill="x", padx=10, pady=2)
        tk.Label(f2, text="Text", width=18, anchor="w",
                 bg="#16213e", fg="#ccc", font=("Courier", 9)).pack(side="left")
        tk.Entry(f2, textvariable=self.var_text, bg="#0f3460", fg="white",
                 insertbackground="white", font=("Courier", 9), width=12).pack(side="left")
        self.var_text.trace_add("write", lambda *_: self._on_change())

        self._row(parent, "Schriftgröße [px]", self._spin,
                  textvariable=self.var_font_size, from_=4, to=200, command=self._on_change)

        self.size_warn = tk.Label(parent, text="", bg="#16213e",
                                   fg="#ff6b6b", font=("Courier", 8))
        self.size_warn.pack(fill="x", padx=10)

    def _build_patch_section(self, parent):
        self._section_label(parent, "PATCH-MODUS (Vignettierung)")

        self.var_patch_enabled = tk.BooleanVar(value=False)
        self.var_patch_size    = self._intvar(RECOMMENDED_PATCH)
        self.var_patch_overlap = self._intvar(0)

        f = tk.Frame(parent, bg="#16213e")
        f.pack(fill="x", padx=10, pady=2)
        tk.Checkbutton(f, text="Patch-Modus aktiv", variable=self.var_patch_enabled,
                       bg="#16213e", fg="#ccc", selectcolor="#0f3460",
                       activebackground="#16213e", font=("Courier", 9),
                       command=self._on_change).pack(side="left")

        self._row(parent, "Patch-Größe [px]", self._spin,
                  textvariable=self.var_patch_size, from_=20, to=MAX_PATCH_SIZE, command=self._on_change)
        self._row(parent, "Überlapp [px]", self._spin,
                  textvariable=self.var_patch_overlap, from_=0, to=50, command=self._on_change)

        tk.Label(parent,
                 text=f"  ⚠ Max empfohlen: {MAX_PATCH_SIZE} px (besser: {RECOMMENDED_PATCH} px)",
                 bg="#16213e", fg="#ffa07a", font=("Courier", 8), anchor="w").pack(fill="x", padx=10)

        tk.Button(parent, text="Patches generieren & anzeigen",
                  command=self._show_patches, bg="#e94560", fg="white",
                  font=("Courier", 9), relief="flat", pady=4).pack(fill="x", padx=10, pady=4)

    def _build_export_section(self, parent):
        self._section_label(parent, "EXPORT")

        tk.Button(parent, text="Gesamtmaske speichern (BMP/PNG/TIFF)",
                  command=self._export_full, bg="#533483", fg="white",
                  font=("Courier", 9), relief="flat", pady=4).pack(fill="x", padx=10, pady=3)

        tk.Button(parent, text="Patches als Projektormasken speichern",
                  command=self._export_patches, bg="#533483", fg="white",
                  font=("Courier", 9), relief="flat", pady=4).pack(fill="x", padx=10, pady=3)

        tk.Button(parent, text="Stage-Koordinaten (JSON + CSV)",
                  command=self._export_coords, bg="#2c7873", fg="white",
                  font=("Courier", 9), relief="flat", pady=4).pack(fill="x", padx=10, pady=3)

        tk.Button(parent, text="Config speichern / laden",
                  command=self._config_dialog, bg="#2c7873", fg="white",
                  font=("Courier", 9), relief="flat", pady=4).pack(fill="x", padx=10, pady=3)

    def _spin(self, parent, **kw):
        kw.setdefault("bg", "#0f3460")
        kw.setdefault("fg", "white")
        kw.setdefault("buttonbackground", "#e94560")
        kw.setdefault("font", ("Courier", 9))
        kw.setdefault("width", 8)
        kw.setdefault("relief", "flat")
        return tk.Spinbox(parent, **kw)

    def _intvar(self, val):
        v = tk.IntVar(value=val)
        v.trace_add("write", lambda *_: self._on_change())
        return v

    def _dblvar(self, val):
        v = tk.DoubleVar(value=val)
        v.trace_add("write", lambda *_: self._on_change())
        return v

    # ── Layer-Verwaltung ─────────────────────────────────────────────────────

    def _refresh_layer_list(self):
        self.layer_listbox.delete(0, tk.END)
        for l in self.layers:
            prefix = "✓" if l.enabled else "✗"
            self.layer_listbox.insert(tk.END, f"{prefix} {l.name}")
        if self.layers:
            self.layer_listbox.selection_set(self.current_layer_idx)

    def _on_layer_select(self, event=None):
        sel = self.layer_listbox.curselection()
        if sel:
            self._sync_layer_to_ui()
            self.current_layer_idx = sel[0]
            self._load_layer_to_ui()

    def _add_layer(self):
        self._sync_layer_to_ui()
        self.layers.append(MaskLayer(name=f"Layer {len(self.layers)+1}"))
        self.current_layer_idx = len(self.layers) - 1
        self._refresh_layer_list()
        self._load_layer_to_ui()
        self._refresh_preview()

    def _del_layer(self):
        if len(self.layers) <= 1:
            messagebox.showwarning("Hinweis", "Mindestens ein Layer muss vorhanden sein.")
            return
        self.layers.pop(self.current_layer_idx)
        self.current_layer_idx = max(0, self.current_layer_idx - 1)
        self._refresh_layer_list()
        self._load_layer_to_ui()
        self._refresh_preview()

    def _move_layer_up(self):
        i = self.current_layer_idx
        if i > 0:
            self.layers[i], self.layers[i-1] = self.layers[i-1], self.layers[i]
            self.current_layer_idx = i - 1
            self._refresh_layer_list()

    def _move_layer_down(self):
        i = self.current_layer_idx
        if i < len(self.layers) - 1:
            self.layers[i], self.layers[i+1] = self.layers[i+1], self.layers[i]
            self.current_layer_idx = i + 1
            self._refresh_layer_list()

    def _sync_layer_name(self):
        if 0 <= self.current_layer_idx < len(self.layers):
            self.layers[self.current_layer_idx].name = self.var_layer_name.get()
            self._refresh_layer_list()

    def _sync_layer_to_ui(self):
        if not self.layers or self.current_layer_idx >= len(self.layers):
            return
        l = self.layers[self.current_layer_idx]
        try:
            l.name              = self.var_layer_name.get()
            l.enabled           = self.var_layer_enabled.get()
            l.intensity         = int(self.var_layer_intensity.get())
            l.grid.n_cols       = int(self.var_ncols.get())
            l.grid.n_rows       = int(self.var_nrows.get())
            l.grid.gap_x        = float(self.var_gapx.get())
            l.grid.gap_y        = float(self.var_gapy.get())
            l.grid.shape.shape_type = self.var_shape.get()
            l.grid.shape.width  = float(self.var_width.get())
            l.grid.shape.height = float(self.var_height.get())
            l.grid.shape.angle  = float(self.var_angle.get())
            l.grid.shape.text   = self.var_text.get()
            l.grid.shape.font_size = int(self.var_font_size.get())
        except (tk.TclError, ValueError):
            pass

    def _load_layer_to_ui(self):
        if not self.layers or self.current_layer_idx >= len(self.layers):
            return
        l = self.layers[self.current_layer_idx]
        self.var_layer_name.set(l.name)
        self.var_layer_enabled.set(l.enabled)
        self.var_layer_intensity.set(l.intensity)
        self.var_ncols.set(l.grid.n_cols)
        self.var_nrows.set(l.grid.n_rows)
        self.var_gapx.set(l.grid.gap_x)
        self.var_gapy.set(l.grid.gap_y)
        self.var_shape.set(l.grid.shape.shape_type)
        self.var_width.set(l.grid.shape.width)
        self.var_height.set(l.grid.shape.height)
        self.var_angle.set(l.grid.shape.angle)
        self.var_text.set(l.grid.shape.text)
        self.var_font_size.set(l.grid.shape.font_size)

    # ── Preview & Rendering ───────────────────────────────────────────────────

    def _on_change(self, *args):
        try:
            self._sync_layer_to_ui()
            self._update_proj_cfg()
            self._check_size_warning()
            self._update_scale_info()
            self._refresh_layer_list()
            self._refresh_preview()
        except Exception:
            pass

    def _update_proj_cfg(self):
        try:
            self.proj_cfg.offset_x  = int(self.var_offset_x.get())
            self.proj_cfg.offset_y  = int(self.var_offset_y.get())
            self.proj_cfg.px_per_um = float(self.var_px_um.get())
        except (tk.TclError, ValueError):
            pass

    def _check_size_warning(self):
        try:
            w = float(self.var_width.get())
            h = float(self.var_height.get())
            if w > MAX_PATCH_SIZE or h > MAX_PATCH_SIZE:
                self.size_warn.config(text=f"⚠ Form > {MAX_PATCH_SIZE}px! → Patch-Modus empfohlen")
            elif w > RECOMMENDED_PATCH or h > RECOMMENDED_PATCH:
                self.size_warn.config(text=f"⚠ Form > {RECOMMENDED_PATCH}px – Vignettierung möglich")
            else:
                self.size_warn.config(text="✓ Größe OK")
        except Exception:
            pass

    def _refresh_preview(self):
        renderer = MaskRenderer(self.proj_cfg)
        self.current_mask = renderer.render_layers(self.layers)

        self.ax.clear()
        self.ax.imshow(self.current_mask, cmap="gray", vmin=0, vmax=255,
                       aspect="auto", interpolation="nearest")

        cx = self.proj_cfg.width  // 2 + self.proj_cfg.offset_x
        cy = self.proj_cfg.height // 2 + self.proj_cfg.offset_y
        self.ax.axhline(cy, color="#e94560", linewidth=0.5, alpha=0.5)
        self.ax.axvline(cx, color="#e94560", linewidth=0.5, alpha=0.5)
        self.ax.plot(cx, cy, "r+", markersize=12, markeredgewidth=1.5)

        if self.var_patch_enabled.get() and self.patches:
            ps = int(self.var_patch_size.get())
            for p in self.patches:
                rect = plt.Rectangle((p.proj_x, p.proj_y), ps, ps,
                                     linewidth=0.8, edgecolor="#00ff9f",
                                     facecolor="none", alpha=0.6)
                self.ax.add_patch(rect)
                self.ax.plot(p.content_cx_px, p.content_cy_px, "c+",
                             markersize=6, markeredgewidth=0.8)

        self.ax.set_title(
            f"Projektor-Maske  ({self.proj_cfg.width}×{self.proj_cfg.height})",
            color="#ccc", fontsize=9)
        self.ax.tick_params(colors="#555")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#333")
        self.fig.tight_layout()
        self.canvas_mpl.draw_idle()

        n_white = int(np.sum(self.current_mask > 0))
        self.status_var.set(
            f"Aktive Pixel: {n_white} | Offset: ({self.proj_cfg.offset_x}, {self.proj_cfg.offset_y}) | "
            f"px/µm: {self.proj_cfg.px_per_um}")

    # ── Patch-Modus ───────────────────────────────────────────────────────────

    def _show_patches(self):
        if self.current_mask is None:
            self._refresh_preview()
        ps = int(self.var_patch_size.get())
        ov = int(self.var_patch_overlap.get())
        cfg = PatchConfig(enabled=True, patch_size=ps, overlap=ov)
        splitter = PatchSplitter(self.proj_cfg)
        self.patches = splitter.split(self.current_mask, cfg)

        if not self.patches:
            messagebox.showinfo("Info", "Keine Patches – Maske ist leer.")
            return

        self._refresh_preview()

        n = len(self.patches)
        ncols = min(4, n)
        nrows = max(1, math.ceil(n / ncols))

        win = tk.Toplevel(self)
        win.title(f"Patch-Übersicht ({n} Patches)")
        win.configure(bg="#16213e")

        fig2, axes = plt.subplots(nrows, ncols, facecolor="#0f0f0f", squeeze=False)

        for i, p in enumerate(self.patches):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            ax.imshow(p.mask, cmap="gray", vmin=0, vmax=255)
            ax.set_title(
                f"P{p.patch_idx}\n"
                f"Δ {p.stage_dx_px:.0f}/{p.stage_dy_px:.0f} px\n"
                f"Δ {p.stage_dx_um:.1f}/{p.stage_dy_um:.1f} µm\n"
                f"Δ {p.stage_dx_mm:.4f}/{p.stage_dy_mm:.4f} mm",
                fontsize=5, color="#ccc")
            ax.axis("off")

        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")

        fig2.suptitle(
            "Patches  |  Δ = Stage-Verschiebung relativ zu Patch (0,0)",
            color="#e94560", fontsize=9)
        fig2.tight_layout()

        canvas2 = FigureCanvasTkAgg(fig2, master=win)
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        canvas2.draw()

        # Koordinaten-Tabelle als Text
        txt = tk.Text(win, height=8, bg="#0f3460", fg="#7ecfff",
                      font=("Courier", 8), relief="flat")
        txt.pack(fill="x", padx=8, pady=4)
        txt.insert("end", f"{'Patch':<10} {'Δx [px]':>10} {'Δy [px]':>10} "
                           f"{'Δx [µm]':>12} {'Δy [µm]':>12} "
                           f"{'Δx [mm]':>12} {'Δy [mm]':>12}\n")
        txt.insert("end", "─" * 80 + "\n")
        for p in self.patches:
            txt.insert("end",
                f"{str(p.patch_idx):<10} {p.stage_dx_px:>10.1f} {p.stage_dy_px:>10.1f} "
                f"{p.stage_dx_um:>12.2f} {p.stage_dy_um:>12.2f} "
                f"{p.stage_dx_mm:>12.5f} {p.stage_dy_mm:>12.5f}\n")
        txt.config(state="disabled")

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_full(self):
        if self.current_mask is None:
            self._refresh_preview()
        path = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("BMP", "*.bmp"), ("PNG", "*.png"), ("TIFF", "*.tiff"), ("Alle", "*.*")])
        if path:
            Image.fromarray(self.current_mask, mode="L").save(path)
            self.status_var.set(f"Gespeichert: {os.path.basename(path)}")

    def _export_patches(self):
        """
        Speichert jeden Patch als vollständige 1920×1080 Projektorbild:
        Der Patch-Inhalt ist auf den Projektor-Mittelpunkt (Offset) zentriert.
        """
        if not self.patches:
            messagebox.showwarning("Patches", "Bitte zuerst Patches generieren.")
            return
        folder = filedialog.askdirectory(title="Zielordner für Patch-Masken")
        if not folder:
            return
        for p in self.patches:
            # Vollbild (1920×1080), zentriert auf Offset
            img_full = Image.fromarray(p.proj_mask, mode="L")
            fname_full = f"patch_{p.patch_idx[0]:02d}_{p.patch_idx[1]:02d}_projector.bmp"
            img_full.save(os.path.join(folder, fname_full))

            # Optional auch den rohen Ausschnitt speichern
            img_crop = Image.fromarray(p.mask, mode="L")
            fname_crop = f"patch_{p.patch_idx[0]:02d}_{p.patch_idx[1]:02d}_crop.bmp"
            img_crop.save(os.path.join(folder, fname_crop))

        self.status_var.set(
            f"{len(self.patches)} Patches gespeichert "
            f"(je _projector.bmp + _crop.bmp) in {folder}")
        messagebox.showinfo(
            "Export",
            f"{len(self.patches)} Patches exportiert.\n\n"
            f"*_projector.bmp  → 1920×1080, auf Projektor-Mitte zentriert\n"
            f"*_crop.bmp       → reiner Ausschnitt\n\n"
            f"Ordner: {folder}")

    def _export_coords(self):
        if not self.patches:
            messagebox.showwarning("Patches", "Bitte zuerst Patches generieren.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Alle", "*.*")])
        if not path:
            return

        splitter = PatchSplitter(self.proj_cfg)
        coords = splitter.export_stage_coordinates(self.patches)

        # JSON
        export_data = {
            "projector": asdict(self.proj_cfg),
            "scale": {
                "px_per_um": self.proj_cfg.px_per_um,
                "um_per_px": round(1.0 / self.proj_cfg.px_per_um, 6),
                "mm_per_px": round(1.0 / self.proj_cfg.px_per_um / 1000.0, 9),
            },
            "patches": coords,
        }
        with open(path, "w") as f:
            json.dump(export_data, f, indent=2)

        # CSV (gleicher Basisname)
        csv_path = os.path.splitext(path)[0] + ".csv"
        with open(csv_path, "w") as f:
            f.write("patch_col,patch_row,dx_px,dy_px,dx_um,dy_um,dx_mm,dy_mm\n")
            for c in coords:
                f.write(
                    f"{c['patch_idx'][0]},{c['patch_idx'][1]},"
                    f"{c['stage_offset_px'][0]},{c['stage_offset_px'][1]},"
                    f"{c['stage_offset_um'][0]},{c['stage_offset_um'][1]},"
                    f"{c['stage_offset_mm'][0]},{c['stage_offset_mm'][1]}\n")

        self.status_var.set(f"Koordinaten gespeichert: {os.path.basename(path)} + .csv")
        messagebox.showinfo(
            "Export",
            f"JSON + CSV gespeichert:\n{path}\n{csv_path}")

    def _config_dialog(self):
        win = tk.Toplevel(self)
        win.title("Config")
        win.configure(bg="#16213e")
        win.geometry("300x120")
        tk.Button(win, text="Config speichern (JSON)", command=self._save_config,
                  bg="#533483", fg="white", font=("Courier", 9), relief="flat").pack(
                  fill="x", padx=20, pady=10)
        tk.Button(win, text="Config laden (JSON)", command=self._load_config,
                  bg="#2c7873", fg="white", font=("Courier", 9), relief="flat").pack(
                  fill="x", padx=20, pady=4)

    def _save_config(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        self._sync_layer_to_ui()
        data = {
            "projector": asdict(self.proj_cfg),
            "layers": [asdict(l) for l in self.layers],
            "patch": asdict(self.patch_cfg),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        pc = data.get("projector", {})
        self.proj_cfg = ProjectorConfig(**pc)
        self.var_offset_x.set(self.proj_cfg.offset_x)
        self.var_offset_y.set(self.proj_cfg.offset_y)
        self.var_px_um.set(self.proj_cfg.px_per_um)
        self.layers = []
        for ld in data.get("layers", []):
            s = ld["grid"]["shape"]
            layer = MaskLayer(
                name=ld["name"],
                enabled=ld["enabled"],
                intensity=ld["intensity"],
                grid=GridParams(
                    n_cols=ld["grid"]["n_cols"],
                    n_rows=ld["grid"]["n_rows"],
                    gap_x=ld["grid"]["gap_x"],
                    gap_y=ld["grid"]["gap_y"],
                    shape=ShapeParams(**s),
                ),
            )
            self.layers.append(layer)
        self.current_layer_idx = 0
        self._refresh_layer_list()
        self._load_layer_to_ui()
        self._refresh_preview()


if __name__ == "__main__":
    app = App()
    app.mainloop()