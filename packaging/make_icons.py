"""Render moth's silkmoth icon (moth/assets/silkmoth_adult_female_icon.svg, DBCLS, CC BY 4.0) for the installers.

  python packaging/make_icons.py OUT_DIR   ->  OUT_DIR/moth-<size>.png (16..1024), moth.ico (Windows), moth.icns (macOS)

Qt draws the SVG (PySide6's QtSvg, the pet's own renderer), centred on a transparent square; Pillow writes the .ico
and .icns.
"""
import os
import sys

SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)
HERE = os.path.dirname(os.path.abspath(__file__))
SVG = os.path.join(os.path.dirname(HERE), "moth", "assets", "silkmoth_adult_female_icon.svg")


def main(out: str) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PIL import Image

    app = QGuiApplication(sys.argv)  # noqa: F841  (QPainter needs one)
    r = QSvgRenderer(SVG)
    vb = r.viewBoxF()
    os.makedirs(out, exist_ok=True)
    paths = {}
    for n in SIZES:
        img = QImage(n, n, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        pad = n * 0.04
        scale = (n - 2 * pad) / max(vb.width(), vb.height())
        w, h = vb.width() * scale, vb.height() * scale
        r.render(p, QRectF((n - w) / 2, (n - h) / 2, w, h))
        p.end()
        paths[n] = os.path.join(out, f"moth-{n}.png")
        img.save(paths[n])
    big = Image.open(paths[1024]).convert("RGBA")
    big.save(os.path.join(out, "moth.ico"), sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])
    big.save(os.path.join(out, "moth.icns"))
    print("\n".join(sorted(os.listdir(out))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
