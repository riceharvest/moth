#!/usr/bin/env bash
# Linux packages from dist/moth (packaging/moth.spec): an AppImage and a plain tarball of the same folder.
#   packaging/linux/build.sh  ->  dist/moth-<version>-x86_64.AppImage, dist/moth-<version>-linux-x86_64.tar.gz
# appimagetool runs with --appimage-extract-and-run (no FUSE needed to build).
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION=$(cat moth/VERSION)
ARCH=${ARCH:-x86_64}
APPDIR=build/AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps" \
         "$APPDIR/usr/share/icons/hicolor/scalable/apps"
cp -a dist/moth "$APPDIR/usr/lib/moth"
install -m 755 packaging/linux/AppRun "$APPDIR/AppRun"
install -m 644 packaging/linux/moth.desktop "$APPDIR/moth.desktop"
install -m 644 packaging/linux/moth.desktop "$APPDIR/usr/share/applications/moth.desktop"
install -m 644 build/icons/moth-256.png "$APPDIR/moth.png"
install -m 644 build/icons/moth-256.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/moth.png"
install -m 644 moth/assets/silkmoth_adult_female_icon.svg "$APPDIR/usr/share/icons/hicolor/scalable/apps/moth.svg"
ln -sf moth.png "$APPDIR/.DirIcon"
TOOL=build/appimagetool
if [ ! -x "$TOOL" ]; then
    curl -fsSL -o "$TOOL" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"
    chmod +x "$TOOL"
fi
ARCH=$ARCH "$TOOL" --appimage-extract-and-run --no-appstream "$APPDIR" "dist/moth-$VERSION-$ARCH.AppImage"
# the tarball: the same app folder, plus its desktop entry and icon (docs/INSTALL.md)
cp packaging/linux/moth.desktop build/icons/moth-256.png dist/moth/
tar -C dist -czf "dist/moth-$VERSION-linux-$ARCH.tar.gz" moth
rm dist/moth/moth.desktop dist/moth/moth-256.png
ls -la dist/*.AppImage dist/*.tar.gz
