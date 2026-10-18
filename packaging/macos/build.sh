#!/usr/bin/env bash
# The macOS disk image from dist/moth.app (packaging/moth.spec): ad-hoc signed (no Developer ID yet: Gatekeeper warns,
# docs/INSTALL.md), with an Applications link to drag it onto.
#   packaging/macos/build.sh  ->  dist/moth-<version>-macos-<arch>.dmg
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION=$(cat moth/VERSION)
ARCH=$(uname -m)
# an unsigned arm64 binary doesn't run at all, and a bundle whose signature doesn't cover every file is "damaged"
codesign --force --deep --sign - dist/moth.app
codesign --verify --deep --strict --verbose=2 dist/moth.app
STAGE=build/dmg
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -R dist/moth.app "$STAGE/"
ln -s /Applications "$STAGE/Applications"
DMG="dist/moth-$VERSION-macos-$ARCH.dmg"
rm -f "$DMG"
hdiutil create -volname "moth" -srcfolder "$STAGE" -fs HFS+ -format UDZO -imagekey zlib-level=9 -ov "$DMG"
ls -la "$DMG"
