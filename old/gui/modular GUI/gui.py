#!/usr/bin/env python3
"""
gui.py – now exposes cue‑duration, ITI, and trials/class.
"""
from __future__ import annotations
import sys
from typing import List, Dict, Any
from PySide6.QtCore    import Qt, QThread, Signal, Slot, QSize
from PySide6.QtGui     import QAction, QIcon, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout,
    QListWidget, QListWidgetItem, QComboBox, QLabel, QTabWidget, QToolBar,
    QSpinBox, QDoubleSpinBox, QGroupBox
)

# ------- dark‑theme shim (unchanged) -------
try:
    import qdarktheme as darktheme
except ModuleNotFoundError:
    try:
        import pyqtdarktheme as darktheme
    except ModuleNotFoundError:
        darktheme = None
def apply_theme(theme="auto"):
    if darktheme is None: return
    if hasattr(darktheme, "setup_theme"):
        darktheme.setup_theme(theme, additional_qss="QFrame#card{background:palette(base);}")
    else:
        qss=darktheme.load_stylesheet(theme=theme)
        QApplication.instance().setStyleSheet(qss+"\nQFrame#card{background:palette(base);}")

# ------- registry imports -----------------
from preprocessing      import PREPROC_FUNCS
from feature_extractors import FEAT_FUNCS
from models             import MODEL_FUNCS
from outputs            import OUTPUT_FUNCS
from trainer            import run_pipeline

# ------- containers -----------------------
class Pipeline:
    def __init__(self, pre: List[str], feat: str, model: str, output: str):
        self.pre, self.feat, self.model, self.output = pre, feat, model, output

# ------- widgets --------------------------
class PipeWidget(QWidget):
    changed = Signal(object)
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self.lw_pre=QListWidget()
        for n in PREPROC_FUNCS:
            it=QListWidgetItem(n,self.lw_pre)
            it.setFlags(it.flags()|Qt.ItemIsUserCheckable); it.setCheckState(Qt.Unchecked)
        self.lw_pre.itemChanged.connect(self._emit)
        self.cb_feat=QComboBox(); self.cb_feat.addItems(FEAT_FUNCS); self.cb_feat.currentIndexChanged.connect(self._emit)
        self.cb_model=QComboBox(); self.cb_model.addItems(MODEL_FUNCS); self.cb_model.currentIndexChanged.connect(self._emit)
        self.cb_out=QComboBox(); self.cb_out.addItems(OUTPUT_FUNCS); self.cb_out.currentIndexChanged.connect(self._emit)
        form.addRow("Pre‑processing", self.lw_pre)
        form.addRow("Feature", self.cb_feat)
        form.addRow("Model",   self.cb_model)
        form.addRow("Output",  self.cb_out)
        self._emit()
    def _sel(self): return [self.lw_pre.item(i).text() for i in range(self.lw_pre.count()) if self.lw_pre.item(i).checkState()==Qt.Checked]
    def _emit(self,*_): self.changed.emit(Pipeline(self._sel(), self.cb_feat.currentText(), self.cb_model.currentText(), self.cb_out.currentText()))

# ------------- inside gui.py ----------------
class ParamWidget(QWidget):
    changed = Signal(dict)

    def __init__(self):
        super().__init__()
        box = QGroupBox("Run Parameters")
        f = QFormLayout(box)

        self.sr   = QDoubleSpinBox(decimals=1, minimum=1, maximum=2000, value=256)
        self.cue  = QDoubleSpinBox(decimals=1, minimum=0.5, maximum=30, value=5)
        self.iti  = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=30, value=5)
        self.nper = QSpinBox(minimum=1, maximum=1000, value=10)
        self.bpl  = QDoubleSpinBox(decimals=1, minimum=0.1, maximum=200, value=8)
        self.bph  = QDoubleSpinBox(decimals=1, minimum=0.1, maximum=200, value=30)
        self.ntch = QDoubleSpinBox(decimals=1, minimum=1, maximum=200, value=50)

        for w in (
            self.sr,
            self.cue,
            self.iti,
            self.nper,
            self.bpl,
            self.bph,
            self.ntch,
        ):
            w.valueChanged.connect(self._emit)

        f.addRow("Sampling rate (Hz)", self.sr)
        f.addRow("Cue duration (s)",   self.cue)
        f.addRow("Inter‑trial int. (s)", self.iti)
        f.addRow("Trials / class",     self.nper)
        f.addRow("BPF low (Hz)",       self.bpl)
        f.addRow("BPF high (Hz)",      self.bph)
        f.addRow("Notch freq (Hz)",    self.ntch)

        lay = QVBoxLayout(self)
        lay.addWidget(box)
        lay.addStretch()

        self._emit()  # ensure first emission so sampling_rate key exists

    def _emit(self, *_):
        self.changed.emit(
            dict(
                sampling_rate=self.sr.value(),
                cue_duration=self.cue.value(),
                inter_trial_interval=self.iti.value(),
                num_trials_per_class=self.nper.value(),
                bpf_low=self.bpl.value(),
                bpf_high=self.bph.value(),
                notch_freq=self.ntch.value(),
            )
        )

# ------- worker --------------------------
class Worker(QThread):
    status=Signal(str); done=Signal(str)
    def __init__(self,pipe,params): super().__init__(); self.pipe,self.params=pipe,params
    def run(self):
        self.status.emit("Running …")
        path,_=run_pipeline(self.pipe,self.params)
        self.done.emit(f"✓ Saved {path}")

# ------- main window ---------------------
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Controller – Full Params")
        QApplication.instance().setFont(QFont("Inter",10))
        self.resize(1020,670)
        apply_theme("dark")

        tb=QToolBar(); tb.setIconSize(QSize(22,22)); self.addToolBar(tb)
        act=QAction(QIcon.fromTheme("media-playback-start"),"Run",self); tb.addAction(act)

        tabs=QTabWidget(documentMode=True)
        self.lbl=QLabel("Idle", alignment=Qt.AlignCenter)
        self.pipe=Pipeline([], "PSD", "LDA", "Neurofeedback"); self.params={}

        for title in ("Offline Training","Offline Classification","Neurofeedback","Adaptive Online"):
            tab=QWidget(); v=QVBoxLayout(tab)
            pw=PipeWidget(); pw.changed.connect(self._set_pipe)
            paramw=ParamWidget(); paramw.changed.connect(self._set_params)
            v.addWidget(pw); v.addWidget(paramw)
            if title=="Offline Training": v.addWidget(self.lbl)
            v.addStretch()
            tabs.addTab(tab,title)

        self.setCentralWidget(tabs)
        act.triggered.connect(self._run)

    @Slot(object)
    def _set_pipe(self,p): self.pipe=p; self._refresh()
    @Slot(dict)
    def _set_params(self,d): self.params=d; self._refresh()
    def _refresh(self):
        total = self.params.get("num_trials_per_class",0)*3
        self.lbl.setText(" · ".join(self.pipe.pre or["<none>"]) + f" → {self.pipe.feat}/{self.pipe.model}/{self.pipe.output} | total trials = {total}")

    def _run(self): 
        self.worker=Worker(self.pipe,self.params); self.worker.status.connect(self.lbl.setText); self.worker.done.connect(self.lbl.setText); self.worker.start()

# ------- entry --------------------------
def main():
    app=QApplication(sys.argv); win=Main(); win.show(); sys.exit(app.exec())
if __name__=="__main__": main()
