"""The installers' first-run window: `moth setup` (moth/setup.py) with buttons instead of questions.

The steps are setup.py's own (Setup.weights_step, memory, hooks_step, autostart_step, pet_step), run on a worker
thread; this file only collects the answers up front (checkboxes), shows setup's lines in a log view, a download
progress bar (the bytes landing in the weights folder), and a folder picker for weights already on this computer
(saved in ~/.moth/app.json; moth_app.py passes them on as --model). When the download fails (the model isn't on the
Hub yet, offline) it says so in plain words and points at the picker.

MOTH_APP_SMOKE (CI only): "start" presses Start moth after a few seconds; "download" presses Download.
"""
from __future__ import annotations

import os
import sys
import threading

import moth_app
from moth import hooks, weights

NAMES = {"claude-code": "Claude Code", "codex": "Codex CLI", "hermes": "Hermes", "pi": "Pi", "opencode": "OpenCode",
         "dsh": "dsh"}
CREDIT = ("silkmoth icon: DataBase Center for Life Science (DBCLS), CC BY 4.0 · moth is Apache-2.0 · "
          "<a href='https://github.com/riceharvest/moth'>github.com/riceharvest/moth</a>")
SHORT_FAIL = ("couldn't download the weights: they aren't on Hugging Face yet (they come with the release), or "
              "you're offline. Have the files? Use a folder on this computer…")
NOT_PUBLISHED = ("moth's weights couldn't be downloaded. The model isn't on Hugging Face yet (it comes with the "
                 "release), or this computer is offline. If you have the files (model.safetensors, config.json, "
                 "tokenizer.json), pick their folder with “Use a folder on this computer…”.")


def translocated() -> str | None:
    """macOS runs an app opened straight from Downloads or the disk image from a temporary read-only copy (App
    Translocation, or /Volumes/...): hooks or autostart written from there point at a path that goes away."""
    exe = sys.executable
    if sys.platform == "darwin" and ("/AppTranslocation/" in exe or exe.startswith("/Volumes/")):
        return "Move moth.app into Applications and open it from there to hook in your coding agents and start at login."
    return None


def folder_bytes(folder: str) -> int:
    n = 0
    for d, _, files in os.walk(folder):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return n


def main() -> int:
    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                                   QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget)

    setup = moth_app.patch_setup()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("moth")
    app.setApplicationDisplayName("moth")
    icon_path = os.path.join(moth_app.bundle(), "moth", "assets", "silkmoth_adult_female_icon.svg")
    icon = QIcon(icon_path)
    app.setWindowIcon(icon)

    class Bridge(QObject):
        log = Signal(str)
        done = Signal(str)   # which job finished: "weights" | "start"

    bridge = Bridge()

    class GuiSetup(setup.Setup):
        """setup.Setup, talking to the window: say -> the log view, ask -> the checkboxes."""
        answers: dict = {}
        errors: list = []

        def say(self, text: str = ""):
            bridge.log.emit(text)
            try:
                print(text, flush=True)   # ~/.moth/app.log when opened from the Start menu / Finder
            except (OSError, ValueError):
                pass

        def line(self, mark: str, text: str):
            if mark == "✗":
                self.errors.append(text)
            self.say(f"  {mark} {text}")   # Qt draws ✓ ✗ everywhere (doctor._mark falls back to ASCII for consoles)

        def memory(self):
            """setup's check, except below ~3 GB free off Linux: there the pet starts moth anyway (moth/launch.py
            checks free memory only where /proc/meminfo says it), and macOS's free + inactive estimate is low."""
            super().memory()
            gb = setup.ram_gb()
            if not self.ram_ok and not sys.platform.startswith("linux") and gb is not None and gb >= setup.MIN_RUN_GB:
                self.ram_ok = True
                if self.left and self.left[-1].startswith("free memory"):
                    self.left.pop()
                self.say("    starting anyway: moth itself needs about 1 GB once it's loaded")

        def ask(self, question: str, default: bool) -> bool:
            q = question.strip()
            for name in hooks.HARNESSES:
                if q.endswith(f" {name}?"):
                    a = self.answers.get(name, default)
                    break
            else:
                if q in ("start the pet when you log in?", "point it at this install?"):
                    a = self.answers.get("autostart", default)
                elif q in ("download them now?", "start the pet now?"):
                    a = True
                else:
                    a = default
            self.say(f"    {q} {'yes' if a else 'no'}")
            return a

    s = GuiSetup(timeout=300.0)

    w = QWidget()
    w.setWindowTitle("moth")
    w.setWindowIcon(icon)
    w.setMinimumWidth(560)
    root = QVBoxLayout(w)

    head = QHBoxLayout()
    pic = QLabel()
    pic.setPixmap(icon.pixmap(72, 72))
    head.addWidget(pic, 0, Qt.AlignTop)
    title = QLabel("<h2 style='margin:0'>hi, i'm moth.</h2>"
                   "<p>a tiny companion that flutters around your screen and keeps you company while your coding "
                   "agent works. everything runs on this computer.</p>")
    title.setWordWrap(True)
    head.addWidget(title, 1)
    root.addLayout(head)

    # 1 weights
    g1 = QGroupBox("1 · moth's brain (the model, downloaded once)")
    l1 = QVBoxLayout(g1)
    wstatus = QLabel()
    wstatus.setWordWrap(True)
    wstatus.setTextFormat(Qt.RichText)
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.hide()
    b1 = QHBoxLayout()
    dl = QPushButton(f"Download ({weights.SIZE_HINT})")
    pick = QPushButton("Use a folder on this computer…")
    b1.addWidget(dl)
    b1.addWidget(pick)
    b1.addStretch(1)
    l1.addWidget(wstatus)
    l1.addWidget(bar)
    l1.addLayout(b1)
    root.addWidget(g1)

    # 2 hooks
    g2 = QGroupBox("2 · coding agents moth listens to (one-way: they never hear moth)")
    l2 = QVBoxLayout(g2)
    boxes: dict = {}
    blocked = translocated()
    for name in hooks.HARNESSES:
        st = hooks.status(name)
        if not (st["detected"] or st["state"] != "absent"):
            continue
        done = st["state"] == "installed"
        cb = QCheckBox(f"{NAMES.get(name, name)}: " + ("hooked in already" if done else hooks.WORD[st["state"]])
                       + f"  ({st['detail']})")
        cb.setChecked(st["state"] != "error")
        cb.setEnabled(not done and st["state"] != "error" and not blocked)
        boxes[name] = cb
        l2.addWidget(cb)
    if not boxes:
        l2.addWidget(QLabel("none found on this computer. moth chats fine without one (later: moth hook <agent>)."))
    root.addWidget(g2)

    # 3 autostart
    g3 = QGroupBox("3 · at login")
    l3 = QVBoxLayout(g3)
    auto = QCheckBox("start moth when I log in")
    auto.setChecked(True)
    l3.addWidget(auto)
    if blocked:
        auto.setChecked(False)
        auto.setEnabled(False)
        for cb in boxes.values():
            cb.setChecked(False)
        warn = QLabel(blocked)
        warn.setWordWrap(True)
        l3.addWidget(warn)
    root.addWidget(g3)

    log = QPlainTextEdit()
    log.setReadOnly(True)
    log.setMinimumHeight(110)
    log.setPlaceholderText("what moth setup does shows up here")
    root.addWidget(log)

    foot = QHBoxLayout()
    credit = QLabel(f"<small>{CREDIT}</small>")
    credit.setOpenExternalLinks(True)
    credit.setWordWrap(True)
    foot.addWidget(credit, 1)
    quit_b = QPushButton("Quit")
    start = QPushButton("Start moth")
    start.setDefault(True)
    foot.addWidget(quit_b)
    foot.addWidget(start)
    root.addLayout(foot)

    busy = {"job": None}

    def fit():
        """A word-wrapped label in a layout keeps its old height on macOS: size it for its text, then the window."""
        wstatus.setMinimumHeight(wstatus.heightForWidth(max(wstatus.width(), 400)))
        w.adjustSize()

    def refresh():
        d = moth_app.model_dir()
        if moth_app.ready():
            where = d or weights.cache_dir()
            wstatus.setText(f"✓ ready: <code>{where}</code>")
            dl.setEnabled(False)
        elif d:
            wstatus.setText(f"✗ <code>{d}</code> needs model.safetensors (complete), config.json and tokenizer.json")
            dl.setEnabled(busy["job"] is None)
        else:
            wstatus.setText(f"not downloaded yet: {weights.SIZE_HINT} from huggingface.co/{weights.repo()}, "
                            f"into <code>{weights.cache_dir()}</code>")
            dl.setEnabled(busy["job"] is None)
        pick.setEnabled(busy["job"] is None)
        start.setEnabled(busy["job"] is None and moth_app.ready())

    def run(job: str, fn):
        busy["job"] = job
        refresh()
        start.setEnabled(False)

        def go():
            try:
                fn()
            except Exception as e:  # noqa: BLE001  shown, never a silent dead window
                s.errors.append(f"{type(e).__name__}: {e}")
                bridge.log.emit(f"  ✗ {type(e).__name__}: {e}")
            bridge.done.emit(job)

        threading.Thread(target=go, daemon=True, name=f"moth-setup-{job}").start()

    def download():
        s.errors.clear()
        bar.setValue(0)
        bar.show()
        timer.start(500)
        run("weights", s.weights_step)

    def tick():
        total = 860e6 if weights.repo() == weights.REPO_ID else 0
        if total:
            bar.setValue(min(99, int(100 * folder_bytes(weights.cache_dir()) / total)))
        else:
            bar.setRange(0, 0)

    timer = QTimer()
    timer.timeout.connect(tick)

    def choose():
        d = QFileDialog.getExistingDirectory(w, "The folder with model.safetensors, config.json and tokenizer.json")
        if not d:
            return
        if not moth_app.usable(d):
            wstatus.setText(f"✗ <code>{d}</code> needs model.safetensors (complete), config.json and tokenizer.json")
            return
        moth_app.save_state(model_dir=d)
        os.environ["MOTH_MODEL_DIR"] = d   # the pet setup starts (a child) gets it too
        log.appendPlainText(f"weights: {d}")
        refresh()

    def begin():
        s.answers = {name: cb.isChecked() for name, cb in boxes.items()}
        s.answers["autostart"] = auto.isChecked()
        s.errors.clear()

        def steps():
            kind, _ = setup.health(s.url)
            s.running = kind == "moth"
            s.weights_step()
            s.memory()
            s.hooks_step()
            s.autostart_step()
            s.pet_step()
            s.summary()

        run("start", steps)

    def finished(job: str):
        busy["job"] = None
        timer.stop()
        if job == "weights":
            bar.setRange(0, 100)
            if s.weights_ok and moth_app.ready():
                bar.setValue(100)
            else:
                bar.hide()
                log.appendPlainText(NOT_PUBLISHED)
                refresh()
                wstatus.setText(f"<span style='color:#b3261e'>✗ {SHORT_FAIL}</span>")
                fit()
                return
        refresh()
        if job == "start" and (s.started or "moth running" in s.had):
            moth_app.save_state(first_run_done=True)
            start.setText("moth is awake")
            start.setEnabled(False)
            log.appendPlainText("\nmoth is flying around your screen now (right-click it for the menu). "
                                "this window closes by itself.")
            QTimer.singleShot(4000, app.quit)

    bridge.log.connect(log.appendPlainText)
    bridge.done.connect(finished)
    dl.clicked.connect(download)
    pick.clicked.connect(choose)
    start.clicked.connect(begin)
    quit_b.clicked.connect(app.quit)

    refresh()
    w.show()
    w.raise_()
    w.activateWindow()
    smoke = os.environ.get("MOTH_APP_SMOKE")
    if smoke == "start":
        QTimer.singleShot(6000, begin)
    elif smoke == "download":
        QTimer.singleShot(3000, download)
    rc = app.exec()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)   # skip interpreter teardown: PySide6 can crash destroying widgets after a worker thread ran
