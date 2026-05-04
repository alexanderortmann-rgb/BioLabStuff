"""
PyQt6 + pyqtgraph live Hamamatsu camera viewer
Links: Kamerabild | Rechts: FFT-Magnitude (log-skaliert)

Benötigt:
    pip install pyqt6 pyqtgraph numpy

Für Hamamatsu:
    DCAM Python Bindings installieren.
    Erwartet: from hamamatsu.dcam import dcam, Stream, copy_frame
    Falls nicht vorhanden → Simulationsmodus.

Trigger-Setup (Arduino):
    trigger_source  = 2  → Extern
    trigger_active  = 2  → Level (aktiv, solange Signal HIGH)
    trigger_polarity = 1 → Positive Flanke (ggf. anpassen)
    1 ->  Kamera löst aus bei steigender Flanke (Signal geht von LOW → HIGH)
    2 -> Kamera löst aus bei fallender Flanke (Signal geht von HIGH → LOW)
    output_trigger_kind = 4 → Belichtungszeit als Output-Signal
"""

import sys
import logging
import traceback
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QStatusBar,
)
import pyqtgraph as pg


# ============================================================
# Logging einrichten
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
# Kamera-Import – mit sauberem Fallback auf Simulation
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
    """
    Kapselt die gesamte Hamamatsu-Kamerasteuerung.
    Im Simulationsmodus werden synthetische Testbilder erzeugt.
    """

    # Kamera-Einstellungen für externes Triggering (Arduino)
    # Dokumentation: DCAM Property IDs – bitte mit SDK-Doku abgleichen!
    EXPOSURE_TIME     = 0.005   # 5 ms – an Belichtungszeit der Lichtquelle anpassen
    TRIGGER_SOURCE    = 2       # 2 = extern (DCAMPROP_TRIGGERSOURCE_EXTERNAL)
    TRIGGER_ACTIVE    = 2       # 2 = Level (HIGH während Belichtung)
                                # 1 = Edge (steigende Flanke) – je nach Arduino-Signal
    TRIGGER_POLARITY  = 1       # 1 = positiv (HIGH auslöst) – ggf. auf 2 (negativ) ändern
    OUTPUT_TRIGGER_KIND = 4     # 4 = Belichtung als Output-Trigger-Signal

    def __init__(self):
        if simulation:
            log.info("CameraInterface: Simulationsmodus – keine Hardware initialisiert.")
            self._sim_frame_count = 0
            return

        log.info("Kamera-Initialisierung gestartet...")
        try:
            self.ctx = dcam
            self.ctx.__enter__()
            log.debug("DCAM-Kontext geöffnet.")

            num_cameras = len(dcam)
            log.info(f"Gefundene Kameras: {num_cameras}")
            if num_cameras == 0:
                raise RuntimeError("Keine Hamamatsu-Kamera gefunden!")

            self.camera = dcam[0]
            self.camera.__enter__()
            log.info(f"Kamera 0 geöffnet: {self.camera.info}")

            # Trigger-Konfiguration
            self.camera["exposure_time"]       = self.EXPOSURE_TIME
            self.camera["trigger_source"]      = self.TRIGGER_SOURCE
            self.camera["trigger_active"]      = self.TRIGGER_ACTIVE
            self.camera["trigger_polarity"]    = self.TRIGGER_POLARITY
            self.camera["output_trigger_kind"] = self.OUTPUT_TRIGGER_KIND

            log.info(
                f"Trigger-Setup: source={self.TRIGGER_SOURCE}, "
                f"active={self.TRIGGER_ACTIVE}, "
                f"polarity={self.TRIGGER_POLARITY}, "
                f"output_kind={self.OUTPUT_TRIGGER_KIND}, "
                f"exposure={self.EXPOSURE_TIME*1000:.1f} ms"
            )

            self.stream = Stream(self.camera, 16)
            self.stream.__enter__()
            log.debug("Frame-Stream mit 16 Puffern geöffnet.")

            self.iterator = iter(self.stream)
            self.camera.start()
            log.info("Kamera gestartet – wartet auf Trigger-Signal.")

        except Exception as e:
            log.error(f"Kamera-Initialisierung fehlgeschlagen: {e}")
            log.debug(traceback.format_exc())
            raise

    def get_frame(self):
        """
        Gibt ein Frame als numpy uint16-Array zurück.
        Blockiert bis zum nächsten Trigger-Signal (externes Triggering).
        """
        if simulation:
            return self._sim_frame()

        try:
            frame_buffer = next(self.iterator)  # Blockiert bis Trigger
            frame = copy_frame(frame_buffer)
            arr = np.array(frame, dtype=np.uint16)

            # Sanity Check – sollte kein reines Rauschen/Grau sein
            if arr.max() == arr.min():
                log.warning(
                    f"Frame hat keinen Kontrast! min=max={arr.max()} "
                    f"– Kamera korrekt konfiguriert?"
                )
            else:
                log.debug(f"Frame OK: shape={arr.shape}, min={arr.min()}, max={arr.max()}")

            return arr

        except StopIteration:
            log.error("Stream erschöpft – zu viele Frames ohne Trigger?")
            raise
        except Exception as e:
            log.error(f"Fehler beim Frame-Lesen: {e}")
            log.debug(traceback.format_exc())
            raise

    def close(self):
        if simulation:
            log.info("Simulation beendet.")
            return
        try:
            self.camera.stop()
            log.info("Kamera gestoppt.")
            self.stream.__exit__(None, None, None)
            self.camera.__exit__(None, None, None)
            self.ctx.__exit__(None, None, None)
            log.info("Alle Ressourcen freigegeben.")
        except Exception as e:
            log.error(f"Fehler beim Schließen der Kamera: {e}")

    # ------------------------------------------------------------------
    # Simulationsmodus: synthetisches Testbild mit bekanntem Muster
    # ------------------------------------------------------------------

    def _sim_frame(self):
        """
        Erzeugt ein synthetisches 512×512 uint16-Bild.
        Enthält definierte Frequenzkomponenten → gut für FFT-Test.
        """
        self._sim_frame_count += 1
        n = 512
        x = np.linspace(0, 2 * np.pi, n)
        X, Y = np.meshgrid(x, x)

        # Leichter Phasen-Drift macht den Livefeed erkennbar "live"
        t = self._sim_frame_count * 0.05

        img = (
            32000
            + 8000 * np.sin(12 * X + 4 * Y + t)
            + 6000 * np.sin(-7 * X + 10 * Y - 0.7 * t)
            + 2000 * np.sin(3 * X - 3 * Y + 0.3 * t)
            + 500  * np.random.randn(n, n)
        )

        img = np.clip(img, 0, 65535).astype(np.uint16)
        return img


# ============================================================
# Worker-Thread (Kamera → Signale an GUI)
# ============================================================

class CameraThread(QThread):
    """
    Läuft in einem eigenen Thread.
    Sendet frame_ready(raw_frame, fft_magnitude) an den Haupt-Thread.
    """

    frame_ready = pyqtSignal(object, object)  # (uint16 frame, float32 FFT)
    error_occurred = pyqtSignal(str)

    # Hanning-Fenster-Cache (wird einmalig berechnet)
    _window = None

    def __init__(self):
        super().__init__()
        self.running = True
        self.camera = None
        self._frame_count = 0
        self._last_fps_time = 0.0

    def run(self):
        import time

        log.info("Kamera-Thread gestartet.")

        try:
            self.camera = CameraInterface()
        except Exception as e:
            msg = f"Kamera nicht initialisierbar: {e}"
            log.error(msg)
            self.error_occurred.emit(msg)
            return

        self._last_fps_time = time.monotonic()

        while self.running:
            try:
                frame = self.camera.get_frame()
                fft_mag = self._compute_fft(frame)

                self.frame_ready.emit(frame, fft_mag)

                # FPS-Logging alle 50 Frames
                self._frame_count += 1
                if self._frame_count % 50 == 0:
                    now = time.monotonic()
                    fps = 50.0 / (now - self._last_fps_time)
                    self._last_fps_time = now
                    log.debug(f"Aktuell {fps:.1f} fps (Frame #{self._frame_count})")

            except Exception as e:
                log.error(f"Akquisitionsfehler: {e}")
                log.debug(traceback.format_exc())
                self.error_occurred.emit(str(e))
                self.msleep(200)

        log.info("Kamera-Thread beendet.")

    def _compute_fft(self, frame):
        """
        FFT mit Hanning-Fenster für bessere Frequenzauflösung.
        Gibt log10(|FFT|) zurück, verschoben auf Null-Frequenz in der Mitte.
        """
        h, w = frame.shape

        # Hanning-Fenster cachen (spart CPU bei gleichbleibender Bildgröße)
        if self._window is None or self._window.shape != (h, w):
            win_y = np.hanning(h).astype(np.float32)
            win_x = np.hanning(w).astype(np.float32)
            self._window = np.outer(win_y, win_x)
            log.debug(f"Hanning-Fenster erstellt: {w}×{h} px")

        f = frame.astype(np.float32)
        f -= f.mean()       # DC-Anteil entfernen (zentriert die Daten)
        f *= self._window   # Fenster anwenden (reduziert Spektrallecken)

        fft = np.fft.fftshift(np.fft.fft2(f))
        mag = np.abs(fft)

        # Kleinen Offset verhindern, damit log10(0) nicht auftaucht
        mag = np.log10(mag + 1.0)

        return mag.astype(np.float32)

    def stop(self):
        log.info("Kamera-Thread wird gestoppt...")
        self.running = False
        self.wait(timeout=3000)
        if self.camera:
            self.camera.close()


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
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(8)

        self.mode_label = QLabel("Modus: Simulationsmodus" if simulation else "Modus: Hardware")
        toolbar_layout.addWidget(self.mode_label)

        self.fps_label = QLabel("FPS: –")
        toolbar_layout.addWidget(self.fps_label)
        toolbar_layout.addStretch()

        reset_levels_btn = QPushButton("Auto-Level")
        reset_levels_btn.setToolTip("Helligkeitsskalierung neu berechnen")
        reset_levels_btn.clicked.connect(self._reset_levels)
        toolbar_layout.addWidget(reset_levels_btn)

        main_layout.addLayout(toolbar_layout)

        # ── Bild-Bereich ─────────────────────────────────────
        views_layout = QHBoxLayout()
        views_layout.setSpacing(4)

        # Kamerabild
        cam_container = QWidget()
        cam_layout = QVBoxLayout(cam_container)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        cam_layout.addWidget(QLabel("Kamerabild (Rohbild)"))
        self.cam_view = pg.ImageView()
        self.cam_view.ui.roiBtn.hide()
        self.cam_view.ui.menuBtn.hide()
        cam_layout.addWidget(self.cam_view)
        views_layout.addWidget(cam_container, 1)

        # FFT-Bild
        fft_container = QWidget()
        fft_layout = QVBoxLayout(fft_container)
        fft_layout.setContentsMargins(0, 0, 0, 0)
        fft_layout.addWidget(QLabel("FFT-Magnitude (log₁₀, Hanning-gefenstert)"))
        self.fft_view = pg.ImageView()
        self.fft_view.ui.roiBtn.hide()
        self.fft_view.ui.menuBtn.hide()
        fft_layout.addWidget(self.fft_view)
        views_layout.addWidget(fft_container, 1)

        main_layout.addLayout(views_layout, 1)

        # ── Status-Bar ────────────────────────────────────────
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Kamera wird gestartet...")

        # ── FPS-Messung ───────────────────────────────────────
        self._fps_timer_count = 0
        self._fps_update_timer = QTimer(self)
        self._fps_update_timer.timeout.connect(self._update_fps_label)
        self._fps_update_timer.start(1000)  # alle 1 s aktualisieren

        # ── Erste-Frame-Flag für Auto-Level ───────────────────
        self._first = True
        self._do_reset_levels = False

        # ── Worker starten ────────────────────────────────────
        self.thread = CameraThread()
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.error_occurred.connect(self._on_error)
        self.thread.start()

        log.info("Hauptfenster initialisiert.")

    # ── Slots ─────────────────────────────────────────────────

    def _on_frame(self, frame, fft):
        """Empfängt frame + FFT und aktualisiert beide Views."""

        auto = self._first or self._do_reset_levels

        # ---- Kamerabild ----
        # uint16 → normalisiert auf [0, 1] für pyqtgraph (vermeidet grau/streifen)
        # Alternativ: einfach uint16 übergeben, aber levels explizit setzen
        if auto:
            lo, hi = int(frame.min()), int(frame.max())
            if lo == hi:
                hi = lo + 1
            self.cam_view.setImage(
                frame,
                autoLevels=False,
                levels=(lo, hi),
            )
            lo_f, hi_f = float(fft.min()), float(fft.max())
            if lo_f == hi_f:
                hi_f = lo_f + 1.0
            self.fft_view.setImage(
                fft,
                autoLevels=False,
                levels=(lo_f, hi_f),
            )
            log.debug(f"Auto-Level: Kamera [{lo}, {hi}], FFT [{lo_f:.2f}, {hi_f:.2f}]")
            self._first = False
            self._do_reset_levels = False
        else:
            self.cam_view.setImage(frame, autoLevels=False)
            self.fft_view.setImage(fft, autoLevels=False)

        self._fps_timer_count += 1

        h, w = frame.shape
        self.status.showMessage(
            f"Frame: {w}×{h} px | "
            f"min={frame.min()} max={frame.max()} | "
            f"{'SIMULATION' if simulation else 'HARDWARE'}"
        )

    def _on_error(self, message):
        log.error(f"Fehler vom Kamera-Thread: {message}")
        self.status.showMessage(f"FEHLER: {message}")

    def _reset_levels(self):
        """Nächster Frame löst Auto-Level aus."""
        self._do_reset_levels = True
        log.info("Auto-Level angefordert.")

    def _update_fps_label(self):
        fps = self._fps_timer_count
        self._fps_timer_count = 0
        self.fps_label.setText(f"FPS: {fps}")

    def closeEvent(self, event):
        log.info("Fenster wird geschlossen – Thread stoppen...")
        self._fps_update_timer.stop()
        self.thread.stop()
        log.info("Programm beendet.")
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
