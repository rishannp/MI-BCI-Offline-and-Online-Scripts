#!/usr/bin/env python3
"""
BCI Experiment Control • Modern UI - First Iteration
=================================
PySide6 desktop GUI for multi‑session BCI experiments.

Key features
------------
* Per‑subject CSV logging + global master log
* Dark/light mode using **qdarktheme** (≥1.3) *or* legacy **pyqtdarktheme**
* Toolbar + tabs: Train → Game → Neuro → Adaptive

This revision adds **robust theme fallback** for older qdarktheme versions that
only expose load_stylesheet() instead of setup_theme().  Upgrade qdarktheme
(pip install --upgrade qdarktheme) for the cleanest experience, but the app
will now work either way.

Run:
    python bci_gui.py
Install:
    pip install pyside6 qdarktheme   # or pyqtdarktheme
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------------
# Dark‑theme shim with graceful fallback
# -----------------------------------------------------------------------------
try:
    import qdarktheme as darktheme  # modern package name
except ModuleNotFoundError:  # pragma: no cover
    try:
        import pyqtdarktheme as darktheme  # legacy name
    except ModuleNotFoundError:
        darktheme = None  # theme disabled

from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize
from PySide6.QtGui import QAction, QIcon, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "BCI Controller"
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
MASTER_FILE = LOG_DIR / "MASTER.csv"
ICONS = {k: QIcon.fromTheme(v) for k, v in {
    "save": "document-save",
    "train": "media-playback-start",
    "game": "applications-games",
    "brain": "face-glasses",
    "adapt": "system-run",
    "exit": "application-exit",
}.items()}

# -----------------------------------------------------------------------------
# Helper: apply dark/light theme depending on qdarktheme API version
# -----------------------------------------------------------------------------

def apply_theme(theme: str = "auto") -> None:
    """Set dark/light palette with any qdarktheme/pyqtdarktheme version."""
    if darktheme is None:
        return  # no theme package installed

    # Newer versions (≥1.3) have setup_theme()
    if hasattr(darktheme, "setup_theme"):
        darktheme.setup_theme(theme, additional_qss="QFrame#card{background:palette(base);}")
    else:
        # Older versions: manually load QSS
        qss = darktheme.load_stylesheet(theme=theme)
        QApplication.instance().setStyleSheet(qss + "\nQFrame#card{background:palette(base);}")

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
class SessionInfo:
    def __init__(self, subj: int, age: int, sex: str, exp: str, notes: str, session: int):
        self.subj, self.age, self.sex, self.exp, self.notes, self.session = subj, age, sex, exp, notes, session
        self.ts = datetime.utcnow().isoformat()

    def row(self) -> str:
        safe_notes = self.notes.replace("\n", " | ")
        return f"{self.ts},{self.subj},{self.age},{self.sex},{self.exp},{self.session},\"{safe_notes}\""

# -----------------------------------------------------------------------------
# Background trainer stub
# -----------------------------------------------------------------------------
class TrainWorker(QThread):
    finished = Signal(str)
    status = Signal(str)

    def __init__(self, info: SessionInfo):
        super().__init__(); self.info = info; self._run = True

    def run(self):
        import time
        self.status.emit("Connecting LSL …")
        for p in range(0, 101, 10):
            if not self._run:
                self.status.emit("Training aborted"); return
            time.sleep(0.5); self.status.emit(f"Training {p}% …")
        model = Path("models") / f"model_subject{self.info.subj:04d}.pkl"
        model.parent.mkdir(parents=True, exist_ok=True); model.write_text("dummy")
        self.finished.emit(str(model))

    def stop(self):
        self._run = False

# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME); self.resize(1000, 650)
        QApplication.instance().setFont(QFont("Inter", 10))
        self._build_toolbar(); self._build_ui()

    # ------------------ toolbar ------------------
    def _build_toolbar(self):
        tb = QToolBar(); tb.setIconSize(QSize(22, 22)); self.addToolBar(tb)
        actions = {
            "save": self._save,
            "train": self._train_clicked,
            "game": self._game_clicked,
            "brain": self._neuro_clicked,
            "adapt": self._adapt_clicked,
            "exit": self.close,
        }
        for key, slot in actions.items():
            if key == "exit": tb.addSeparator()
            act = QAction(ICONS[key], key.capitalize(), self, triggered=slot)
            tb.addAction(act)
        self._dark = QAction("Dark Mode", self, checkable=True, triggered=self._toggle_dark)
        self._dark.setEnabled(bool(darktheme)); tb.addSeparator(); tb.addAction(self._dark)

    # ------------------ central UI ------------------
    def _build_ui(self):
        root = QVBoxLayout(); root.setContentsMargins(16, 12, 16, 12); root.setSpacing(16)
        # Metadata card
        card = QFrame(objectName="card"); card.setStyleSheet("#card{border-radius:12px;padding:16px;}")
        form = QFormLayout(card)
        self.sub = QSpinBox(minimum=1, maximum=9999); self.age = QSpinBox(minimum=6, maximum=120)
        self.sex = QComboBox(); self.sex.addItems(["Male", "Female", "Other", "Prefer not to say"])
        self.exp = QComboBox(); self.exp.addItems(["None", "Some", "Experienced"])
        self.sess = QSpinBox(minimum=1, maximum=100); self.notes = QTextEdit(); self.notes.setFixedHeight(60)
        form.addRow("Subject #", self.sub); form.addRow("Age", self.age); form.addRow("Sex", self.sex)
        form.addRow("BCI Exp", self.exp); form.addRow("Session #", self.sess); form.addRow("Notes", self.notes)
        root.addWidget(card)
        # Tabs
        tabs = QTabWidget(documentMode=True)
        tabs.addTab(self._status_tab("Idle", "Start Training", self._train_clicked, attr=("train_lbl", "train_btn")), "Train")
        tabs.addTab(self._status_tab("Game idle", "Launch Game", self._game_clicked, attr=("game_lbl", None)), "Game")
        tabs.addTab(self._status_tab("Neuro idle", "Start Neuro", self._neuro_clicked, attr=("neuro_lbl", None)), "Neuro")
        tabs.addTab(self._status_tab("Adaptive idle", "Run Adaptive", self._adapt_clicked, attr=("adapt_lbl", None)), "Adaptive")
        root.addWidget(tabs)
        w = QWidget(); w.setLayout(root); self.setCentralWidget(w)

    def _status_tab(self, label: str, btn_text: str, slot, attr=(None, None)):
        w = QWidget(); v = QVBoxLayout(w)
        card = QFrame(objectName="card"); card.setStyleSheet("#card{border-radius:12px;padding:32px;}")
        lay = QVBoxLayout(card)
        lbl = QLabel(label, alignment=Qt.AlignCenter)
        btn = QPushButton(btn_text); btn.clicked.connect(slot); btn.setFixedWidth(200)
        lay.addWidget(lbl); lay.addWidget(btn, alignment=Qt.AlignCenter); lay.addStretch(1); v.addWidget(card, alignment=Qt.AlignCenter)
        if attr[0]: setattr(self, attr[0], lbl)
        if attr[1]: setattr(self, attr[1], btn)
        return w

    # ------------------ helpers ------------------
    def _session_info(self) -> SessionInfo:
        return SessionInfo(self.sub.value(), self.age.value(), self.sex.currentText(), self.exp.currentText(), self.notes.toPlainText().strip(), self.sess.value())

    def _append(self, path: Path, row: str):
        if not path.exists():
            path.write_text("timestamp,subject,age,sex,bci_experience,session,notes\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(row + "\n")

    # ------------------ slots/actions ------------------
    @Slot()
    def _save(self):
        info = self._session_info(); subj_file = LOG_DIR / f"subject_{info.subj:04d}.csv"
        self._append(subj_file, info.row()); self._append(MASTER_FILE, info.row())
        QMessageBox.information(self, "Saved", f"Logged to {subj_file.name}")

    @Slot()
    def _train_clicked(self):
        if not hasattr(self, "worker") or not self.worker.isRunning():
            self.train_btn.setText("Stop Training"); self.train_lbl.setText("Initialising …")
            self.worker = TrainWorker(self._session_info()); self.worker.status.connect(self.train_lbl.setText); self.worker.finished.connect(self._train_done); self.worker.start()
        else:
            self.worker.stop(); self.train_lbl.setText("Stopping …")

    @Slot(str)
    def _train_done(self, path: str):
        self.train_lbl.setText(f"✓ Model ⇒ {path}"); self.train_btn.setText("Start Training")

    @Slot()
    def _game_clicked(self):
        QMessageBox.information(self, "Game", "Game launch not wired yet.")

    @Slot()
    def _neuro_clicked(self):
        QMessageBox.information(self, "Neuro", "Neurofeedback pending implementation.")

    @Slot()
    def _adapt_clicked(self):
        QMessageBox.information(self, "Adaptive", "Adaptive run not implemented.")

    @Slot(bool)
    def _toggle_dark(self, checked: bool):
        if darktheme is None:
            QMessageBox.warning(self, "Theme", "qdarktheme is not installed."); return
        theme = "dark" if checked else "light"
        apply_theme(theme)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    apply_theme("dark")
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()