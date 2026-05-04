"""
PyQt6 + pyqtgraph live Hamamatsu camera viewer
Links: Kamerabild | Rechts: FFT-Magnitude (log-skaliert)

Steuerung:
    Mausrad           → Zoom
    Linksklick + Drag → Pan
    "Zoom zurücksetzen"-Button → Originalansicht wiederherstellen
    "Auto-Level"-Button → Helligkeitsskalierung neu berechnen

Benötigt:
    pip install pyqt6 pyqtgraph numpy

Für Hamamatsu:
    DCAM Python Bindings installieren.
    Erwartet: from hamamatsu.dcam import dcam, Stream, copy_frame
    Falls nicht vorhanden → Simulationsmodus.
"""

import sys
import logging
import traceback
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QStatusBar,
)
import pyqtgraph as pg


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("hamamatsu_viewer.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("HamamatsuViewer")


# ============================================================
# Kamera-Import – Fallback auf Simulation
# ============================================================

simulation = False

try:
    from hamamatsu.dcam import dcam, Stream, copy_frame
    log.info("Hamamatsu DCAM Bindings erfolgreich geladen.")
except Exception as e:
    simulation = True
    log.warning(f"DCAM Import fehlgeschlagen – Simulationsmodus aktiv. Grund: {e}")


# ============================================================
# Kamera-Abstraktion
# ============================================================

class CameraInterface:

    EXPOSURE_TIME       = 0.005  # 5 ms – an Lichtquellen-Puls anpassen
    TRIGGER_SOURCE      = 2      # 2 = extern (Arduino)
    TRIGGER_ACTIVE      = 2      # 2 = Level (HIGH während Belichtung)
    TRIGGER_POLARITY    = 1      # 1 = positiv (steigende Flanke)
    OUTPUT_TRIGGER_KIND = 4      # 4 = Belichtung als Output-Signal

    def __init__(self):
        if simulation:
            log.info("CameraInterface: Simulationsmodus.")
            self._sim_frame_count = 0
            return

        log.info("Kamera-Initialisierung gestartet...")
        try:
            self.ctx = dcam
            self.ctx.__enter__()

            num_cameras = len(dcam)
            log.info(f"Gefundene Kameras: {num_cameras}")
            if num_cameras == 0:
                raise RuntimeError("Keine Hamamatsu-Kamera gefunden!")

            self.camera = dcam[0]
            self.camera.__enter__()
            log.info(f"Kamera 0 geöffnet: {self.camera.info}")

            self.camera["exposure_time"]       = self.EXPOSURE_TIME
            self.camera["trigger_source"]      = self.TRIGGER_SOURCE
            self.camera["trigger_active"]      = self.TRIGGER_ACTIVE
            self.camera["trigger_polarity"]    = self.TRIGGER_POLARITY
            self.camera["output_trigger_kind"] = self.OUTPUT_TRIGGER_KIND

            # Werte zurücklesen – zeigt ob Kamera sie akzeptiert hat
            for prop in ["exposure_time", "trigger_source", "trigger_active",
                         "trigger_polarity", "output_trigger_kind"]:
                actual = self.camera[prop]
                log.info(f"  {prop}: gesetzt → tatsächlich = {actual}")

            self.stream = Stream(self.camera, 16)
            self.stream.__enter__()
            self.iterator = iter(self.stream)
            self.camera.start()
            log.info("Kamera gestartet – wartet auf Trigger.")

        except Exception as e:
            log.error(f"Kamera-Initialisierung fehlgeschlagen: {e}")
            log.debug(traceback.format_exc())
            raise

    def get_frame(self):
        if simulation:
            return self._sim_frame()
        try:
            frame_buffer = next(self.iterator)
            frame = copy_frame(frame_buffer)
            arr = np.array(frame, dtype=np.uint16)
            log.debug(f"Frame: shape={arr.shape}, min={arr.min()}, max={arr.max()}")
            return arr
        except Exception as e:
            log.error(f"Frame-Fehler: {e}")
            raise

    def close(self):
        if simulation:
            return
        try:
            self.camera.stop()
            self.stream.__exit__(None, None, None)
            self.camera.__exit__(None, None, None)
            self.ctx.__exit__(None, None, None)
            log.info("Kamera-Ressourcen freigegeben.")
        except Exception as e:
            log.error(f"Fehler beim Schließen: {e}")

    def _sim_frame(self):
        self._sim_frame_count += 1
        n = 512
        x = np.linspace(0, 2 * np.pi, n)
        X, Y = np.meshgrid(x, x)
        t = self._sim_frame_count * 0.05
        img = (
            32000
            + 8000 * np.sin(12 * X + 4 * Y + t)
            + 6000 * np.sin(-7 * X + 10 * Y - 0.7 * t)
            + 2000 * np.sin(3 * X - 3 * Y + 0.3 * t)
            + 500  * np.random.randn(n, n)
        )
        return np.clip(img, 0, 65535).astype(np.uint16)


# ============================================================
# Worker-Thread
# ============================================================

class CameraThread(QThread):

    frame_ready    = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str)

    _window = None

    def __init__(self):
        super().__init__()
        self.running = True
        self.camera  = None
        self._frame_count   = 0
        self._last_fps_time = 0.0

    def run(self):
        import time
        log.info("Kamera-Thread gestartet.")

        try:
            self.camera = CameraInterface()
        except Exception as e:
            self.error_occurred.emit(str(e))
            return

        self._last_fps_time = time.monotonic()

        while self.running:
            try:
                frame   = self.camera.get_frame()
                fft_mag = self._compute_fft(frame)
                self.frame_ready.emit(frame, fft_mag)

                self._frame_count += 1
                if self._frame_count % 50 == 0:
                    now = time.monotonic()
                    fps = 50.0 / (now - self._last_fps_time)
                    self._last_fps_time = now
                    log.debug(f"{fps:.1f} fps (Frame #{self._frame_count})")

            except Exception as e:
                log.error(f"Akquisitionsfehler: {e}")
                self.error_occurred.emit(str(e))
                self.msleep(200)

    def _compute_fft(self, frame):
        h, w = frame.shape
        if self._window is None or self._window.shape != (h, w):
            win_y = np.hanning(h).astype(np.float32)
            win_x = np.hanning(w).astype(np.float32)
            self._window = np.outer(win_y, win_x)
            log.debug(f"Hanning-Fenster erstellt: {w}×{h} px")

        f  = frame.astype(np.float32)
        f -= f.mean()
        f *= self._window

        fft = np.fft.fftshift(np.fft.fft2(f))
        return np.log10(np.abs(fft) + 1.0).astype(np.float32)

    def stop(self):
        log.info("Thread wird gestoppt...")
        self.running = False
        self.wait(3000)
        if self.camera:
            self.camera.close()


# ============================================================
# Hilfsfunktion: PlotWidget-basierter Bild-Viewer
# ============================================================

def make_image_widget(title: str, colormap: str = "inferno"):
    """
    Erstellt einen schlanken PlotWidget mit ImageItem.

    Steuerung (pyqtgraph-Standard):
        Mausrad           → Zoom (zentriert auf Mauszeiger)
        Linksklick + Drag → Pan
        Rechtsklick       → Kontextmenü / Zoom-Reset

    Gibt (PlotWidget, ImageItem) zurück.
    """
    plot = pg.PlotWidget()
    plot.setAspectLocked(True)   # Pixel bleiben quadratisch beim Zoom
    plot.invertY(True)           # Y-Achse: oben = 0 (Bildkoordinaten)
    plot.setLabel("top", title)
    plot.hideAxis("left")
    plot.hideAxis("bottom")
    plot.setBackground("k")      # Schwarzer Hintergrund

    img_item = pg.ImageItem()
    plot.addItem(img_item)

    cm = pg.colormap.get(colormap, source="matplotlib")
    img_item.setColorMap(cm)

    return plot, img_item


# ============================================================
# Hauptfenster
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hamamatsu Live View + FFT")
        self.resize(1400, 750)

        pg.setConfigOptions(imageAxisOrder="row-major")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Toolbar ──────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        mode_text = "Simulationsmodus" if simulation else "Hardware"
        toolbar.addWidget(QLabel(f"Modus: {mode_text}"))

        self.fps_label = QLabel("FPS: –")
        toolbar.addWidget(self.fps_label)
        toolbar.addStretch()

        btn_levels = QPushButton("Auto-Level")
        btn_levels.setToolTip("Helligkeitsskalierung einmalig neu berechnen")
        btn_levels.clicked.connect(self._request_reset_levels)
        toolbar.addWidget(btn_levels)

        btn_zoom = QPushButton("Zoom zurücksetzen")
        btn_zoom.setToolTip("Zoom und Pan beider Ansichten zurücksetzen")
        btn_zoom.clicked.connect(self._reset_zoom)
        toolbar.addWidget(btn_zoom)

        main_layout.addLayout(toolbar)

        # ── Bildbereich ───────────────────────────────────────
        views_layout = QHBoxLayout()
        views_layout.setSpacing(4)

        # Kamerabild – "inferno" Colormap (gut für Graustufen-Kameras)
        cam_wrap = QWidget()
        cam_v = QVBoxLayout(cam_wrap)
        cam_v.setContentsMargins(0, 0, 0, 0)
        self.cam_plot, self.cam_img = make_image_widget(
            "Kamerabild", colormap="inferno"
        )
        cam_v.addWidget(self.cam_plot)
        views_layout.addWidget(cam_wrap, 1)

        # FFT – "viridis" Colormap (gut für Frequenzspektrum)
        fft_wrap = QWidget()
        fft_v = QVBoxLayout(fft_wrap)
        fft_v.setContentsMargins(0, 0, 0, 0)
        self.fft_plot, self.fft_img = make_image_widget(
            "FFT-Magnitude (log₁₀, Hanning)", colormap="viridis"
        )
        fft_v.addWidget(self.fft_plot)
        views_layout.addWidget(fft_wrap, 1)

        main_layout.addLayout(views_layout, 1)

        # ── Statusbar ─────────────────────────────────────────
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Kamera wird gestartet...")

        # ── Interner Zustand ──────────────────────────────────
        self._first           = True
        self._do_reset_levels = False
        self._cam_levels      = None   # (lo, hi) uint16
        self._fft_levels      = None   # (lo, hi) float32
        self._fps_count       = 0

        # ── FPS-Timer ─────────────────────────────────────────
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

        # ── Worker ────────────────────────────────────────────
        self.thread = CameraThread()
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.error_occurred.connect(self._on_error)
        self.thread.start()

        log.info("Hauptfenster bereit.")

    # ── Frame-Update ──────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray, fft: np.ndarray):

        if self._first or self._do_reset_levels:
            lo, hi = int(frame.min()), int(frame.max())
            if lo == hi:
                hi = lo + 1
            self._cam_levels = (lo, hi)

            lo_f, hi_f = float(fft.min()), float(fft.max())
            if lo_f == hi_f:
                hi_f = lo_f + 1.0
            self._fft_levels = (lo_f, hi_f)

            log.debug(
                f"Levels neu: Kamera [{lo}, {hi}], FFT [{lo_f:.2f}, {hi_f:.2f}]"
            )
            self._first           = False
            self._do_reset_levels = False

        self.cam_img.setImage(frame, levels=self._cam_levels, autoLevels=False)
        self.fft_img.setImage(fft,   levels=self._fft_levels, autoLevels=False)

        self._fps_count += 1

        h, w = frame.shape
        self.status.showMessage(
            f"Frame: {w}×{h} px  |  "
            f"min={frame.min()}  max={frame.max()}  |  "
            f"{'SIMULATION' if simulation else 'HARDWARE'}"
        )

    def _on_error(self, message: str):
        log.error(f"Thread-Fehler: {message}")
        self.status.showMessage(f"FEHLER: {message}")

    # ── Button-Callbacks ──────────────────────────────────────

    def _request_reset_levels(self):
        self._do_reset_levels = True
        log.info("Auto-Level angefordert.")

    def _reset_zoom(self):
        self.cam_plot.autoRange()
        self.fft_plot.autoRange()
        log.info("Zoom zurückgesetzt.")

    # ── FPS ───────────────────────────────────────────────────

    def _update_fps(self):
        self.fps_label.setText(f"FPS: {self._fps_count}")
        self._fps_count = 0

    # ── Aufräumen ─────────────────────────────────────────────

    def closeEvent(self, event):
        log.info("Fenster schließt – Thread stoppen...")
        self._fps_timer.stop()
        self.thread.stop()
        event.accept()


# ============================================================
# Einstiegspunkt
# ============================================================

def main():
    log.info("=== Hamamatsu Live Viewer startet ===")
    log.info(f"Simulationsmodus: {simulation}")

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
