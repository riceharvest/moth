# PyInstaller spec for the installers: one onedir app for this OS (docs/INSTALL.md, "Building the installers").
#   pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/moth.spec
# -> Windows dist/moth/ (moth.exe console + mothw.exe windowed, one _internal/), macOS dist/moth.app, Linux dist/moth/.
# onedir, never onefile: a onefile exe unpacks itself on every start, and a hook runs on every agent turn.
# Icons: packaging/make_icons.py must have written build/icons/ first.
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

HERE = os.path.dirname(os.path.abspath(SPEC))
ROOT = os.path.dirname(HERE)
ICONS = os.path.join(ROOT, "build", "icons")
VERSION = open(os.path.join(ROOT, "moth", "VERSION")).read().strip()

# research / dev-only modules the app never runs (as pyproject.toml leaves them out of the wheel)
DEV = {"add_hermes_pool", "build_sft", "eval", "gen_dialogs", "interview", "judge_mcp", "metrics", "pairwise",
       "probes", "rejudge", "sft"}
mods = sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, "moth")) if f.endswith(".py") and f[:-3] not in DEV)
# subpackages too (moth/backends/ ...), but not integrations/ (data: the hooks run those files from source)
subs = sorted(f"moth.{d}" for d in os.listdir(os.path.join(ROOT, "moth"))
              if os.path.isfile(os.path.join(ROOT, "moth", d, "__init__.py")) and d != "integrations")
hidden = [f"moth.{m}" for m in mods if m != "__init__"] + [m for d in subs for m in collect_submodules(d)] + \
    collect_submodules("tiny_agent") + [
    "moth_firstrun", "moth_hook", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets", "PySide6.QtSvg"]

# data next to the modules, where moth looks for it (paths relative to each module): every non-Python file under
# moth/ but the research data pyproject.toml also leaves out, and integrations/ whole (the hooks run its .py files
# from source; `moth hook` copies the Pi / OpenCode / Hermes adapters from there)
SKIP_DIRS = {"__pycache__", "candidates"}
SKIP_FILES = {"agent_pool.json", "bible_lines_r3.json", "test_integrations.py", ".DS_Store"}
datas = []
for d, dirs, files in os.walk(os.path.join(ROOT, "moth")):
    dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
    code = os.path.relpath(d, ROOT).replace("\\", "/").startswith("moth/integrations")
    for f in files:
        if f in SKIP_FILES or f.endswith(".pyc") or (f.endswith(".py") and not code):
            continue
        datas.append((os.path.join(d, f), os.path.relpath(d, ROOT)))
for f in ("LICENSE", "release/NOTICE", "release/LICENSE-LFM"):
    datas.append((os.path.join(ROOT, f), "licenses"))

a = Analysis(
    [os.path.join(HERE, "moth_app.py")],
    pathex=[ROOT, HERE],
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "scipy", "pandas", "PIL", "cv2", "tensorboard",
              "torchvision", "torchaudio", "triton", "PySide6.QtWebEngineCore", "PySide6.Qt3DCore", "PySide6.QtQuick",
              "PySide6.QtQml", "PySide6.QtPdf"],
    noarchive=False,
)


def keep(dest: str) -> bool:
    """Leave out what the app never needs: torch's C++ tests, test binaries and headers (~140 MB); on Linux the GTK
    platform theme with its whole GTK stack, and the libraries every desktop has, which must come from the host
    (an older bundled libstdc++ / glib / fontconfig breaks the host's GL drivers and input modules). The xcb-util
    family and xkbcommon stay bundled: those are the ones distros often lack (Qt's xcb plugin needs them)."""
    d = dest.replace("\\", "/")
    if d.startswith(("torch/test/", "torch/include/", "torch/share/")) or "torchbind_test" in d:
        return False
    if d.startswith("torch/bin/") and not d.endswith(("torch_shm_manager", "torch_shm_manager.exe")):
        return False   # C++ test binaries and protoc; import torch looks for torch_shm_manager on Linux / macOS
    if not sys.platform.startswith("linux"):
        return True
    if d.endswith("platformthemes/libqgtk3.so"):
        return False
    base = os.path.basename(d)
    return not base.startswith(HOST_LIBS)


HOST_LIBS = (
    # the GTK stack (only the dropped libqgtk3 needs it)
    "libgtk-3", "libgdk-3", "libgdk_pixbuf", "libatk-", "libatspi", "libcairo", "libpango", "libfribidi", "libthai",
    "libdatrie", "libgraphite2", "libpixman", "libepoxy", "libcloudproviders", "libtinysparql", "libjson-glib",
    "libglycin", "liblcms2", "libXcomposite", "libXdamage", "libXinerama", "libXi.", "libXrandr", "libXcursor",
    "libXext", "libXfixes", "libXrender", "libXau", "libharfbuzz",
    # on every desktop, and must match the host's
    "libstdc++", "libgcc_s", "libglib-2", "libgobject-2", "libgio-2", "libgmodule-2", "libgthread-2", "libfontconfig",
    "libfreetype", "libdbus-1", "libsystemd", "libX11.", "libX11-xcb", "libxcb.so", "libxcb-render.so", "libxcb-shm",
    "libxcb-glx", "libcom_err", "libgssapi_krb5", "libk5crypto", "libkrb5", "libkeyutils", "libselinux", "libmount",
    "libblkid", "libpcre2", "libcap", "libseccomp", "libxml2", "libze_loader")
# kept: Python's own (libffi, libz, libbz2, liblzma, libssl, libcrypto: older distros have other sonames), Qt's ICU
a.binaries = [b for b in a.binaries if keep(b[0])]
a.datas = [x for x in a.datas if keep(x[0])]
pyz = PYZ(a.pure)

# moth-hook: the hook runner, stdlib only, with its own small archive (moth_hook.py: a hook runs on every agent turn)
sys.path.insert(0, HERE)
from moth_hook import HOOK_IMPORTS  # noqa: E402
h = Analysis([os.path.join(HERE, "moth_hook.py")], pathex=[HERE], hiddenimports=list(HOOK_IMPORTS),
             excludes=["moth", "tiny_agent", "torch", "numpy", "PySide6", "shiboken6", "tokenizers", "safetensors",
                       "huggingface_hub", "moth_app", "moth_firstrun"])
hpyz = PYZ(h.pure)

win = sys.platform == "win32"
mac = sys.platform == "darwin"
icon = os.path.join(ICONS, "moth.ico" if win else "moth.icns" if mac else "moth-256.png")

exes = [EXE(pyz, a.scripts, [], exclude_binaries=True, name="moth", console=not mac, icon=icon,
            disable_windowed_traceback=False, upx=False)]
if win:   # the Start menu, autostart and the pet: no console window
    exes.append(EXE(pyz, a.scripts, [], exclude_binaries=True, name="mothw", console=False, icon=icon, upx=False))

exes.append(EXE(hpyz, h.scripts, [], exclude_binaries=True, name="moth-hook", console=True, upx=False))

coll = COLLECT(*exes, a.binaries, a.datas, h.binaries, h.datas, name="moth", upx=False)

if mac:
    app = BUNDLE(
        coll,
        name="moth.app",
        icon=icon,
        bundle_identifier="io.github.riceharvest.moth",
        version=VERSION,
        info_plist={
            "CFBundleName": "moth",
            "CFBundleDisplayName": "moth",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSUIElement": True,                 # a desktop pet with a menu bar icon: no Dock icon, no app menu
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "moth: Apache-2.0. Silkmoth icon: DBCLS, CC BY 4.0.",
        },
    )
