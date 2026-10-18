# Installing moth

Two ways to get moth:

- **The installers** (no Python needed): a Windows setup `.exe`, a macOS `.dmg` and a Linux AppImage (or tarball).
  They come with the release; they aren't published yet.
- **Python**: the Hugging Face download or `pip install moth-companion` (see the [README](../README.md) and the
  [model card](../release/README.md)).

Both are the same moth. The installers bundle Python, CPU-only torch, tokenizers and Qt for the desktop pet. The
model weights (about 860 MB) aren't in the installer: moth downloads them once, on first launch.

## Installers

Each installer is built and tested by `.github/workflows/installers.yml` on GitHub's own runners (Windows, macOS on
Apple silicon, Ubuntu). For now the installers exist only as workflow artifacts. They'll be attached to the release.

| | file | size |
|---|---|---|
| Windows 10/11, x64 | `moth-<version>-windows-x64-setup.exe` | 183 MB (about 620 MB installed) |
| macOS 12+, Apple silicon | `moth-<version>-macos-arm64.dmg` | 192 MB |
| Linux x86_64 (glibc 2.35+) | `moth-<version>-x86_64.AppImage` | 239 MB (runs in place) |
| Linux x86_64, no FUSE | `moth-<version>-linux-x86_64.tar.gz` | 263 MB |

(Sizes for 0.3.0 with torch 2.14.1 CPU, PySide6 6.12.0 and PyInstaller 6.22.3.)

The installers aren't code-signed yet, so Windows and macOS warn the first time you open them. See
[Unsigned: the warnings and how to get past them](#unsigned-the-warnings-and-how-to-get-past-them) below.

There's no Intel Mac build. moth needs torch 2.4 or newer, and torch stopped publishing macOS x86_64 wheels after 2.2.
On an Intel Mac, use Python with a torch you build yourself.

### First launch

The first launch opens a small window, which is `moth setup` with buttons:

1. **moth's brain.** Click **Download (about 860 MB)** to fetch the weights from Hugging Face into
   `~/.cache/moth/<repo>/`, with a progress bar. If you already have them (`model.safetensors`, `config.json`,
   `tokenizer.json` in one folder), click **Use a folder on this computer…** instead. moth remembers that folder in
   `~/.moth/app.json`. Setting `MOTH_MODEL_DIR` to such a folder does the same.
   Until the model is published on Hugging Face, the download fails with a plain message saying so; use the folder
   button or `MOTH_MODEL_DIR` for now.
2. **Coding agents.** One checkbox per agent found on this computer (Claude Code, Codex, Hermes, Pi, OpenCode, dsh).
   moth backs up and merges their configs, as `moth hook` does.
3. **At login.** Start moth when you log in. This is the same entry `moth autostart on` writes.

**Start moth** runs those steps and wakes moth up. The window closes and the moth flies around your screen.
Right-click it for the menu (chat page, mute, sleep, quit). After that, opening moth starts the pet directly. To see
the window again, run `moth --first-run`.

### Windows

1. Run `moth-<version>-windows-x64-setup.exe`. It installs for your user only (no admin) into
   `%LOCALAPPDATA%\Programs\moth`.
2. Options:
   - **Add the moth command to PATH**: on by default.
   - **Start moth when I log in**: off by default. The first-run window asks again.
   - **Desktop shortcut**: off by default.
3. **moth** in the Start menu opens the first-run window, then the pet. The program is `mothw.exe`, which has no
   console window.

**Command line.** `moth.exe` is the console program. In a new terminal, try `moth hooks`, `moth doctor`,
`moth hook --all`, `moth unhook --all` or `moth autostart on|off`.

**Uninstall.** Go to Settings → Apps → moth → Uninstall. It first asks whether to remove moth's hooks from your
coding agents (`moth unhook --all`, which removes only what moth added). It also turns autostart off and stops a
running moth. Your `~/.moth` folder (chat log, token, settings) and the downloaded weights stay. Delete
`%USERPROFILE%\.moth` and `%USERPROFILE%\.cache\moth` to remove them too.

**Silent install.**

```
moth-<version>-windows-x64-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /TASKS=addtopath,autostart
unins000.exe /VERYSILENT /SUPPRESSMSGBOXES
```

The silent uninstall answers yes to removing the hooks.

### macOS

1. Open `moth-<version>-macos-arm64.dmg` and drag **moth** onto **Applications**.
2. Open moth from Applications, not from the disk image or Downloads. From either of those, macOS runs the app from a
   temporary read-only copy. Hooks and autostart written from there would point at a path that disappears, so the
   window doesn't offer them until moth runs from Applications.
3. moth has no Dock icon. It lives in the menu bar (the silkmoth icon), and the moth itself flies on your screen.

**Command line.** Run `/Applications/moth.app/Contents/MacOS/moth hooks` (or `doctor`, `hook --all`,
`unhook --all`, `autostart on|off`). For a short `moth` command:
`ln -s /Applications/moth.app/Contents/MacOS/moth /usr/local/bin/moth`.

**Uninstall.**

```bash
/Applications/moth.app/Contents/MacOS/moth unhook --all     # remove moth's hooks (asks first)
/Applications/moth.app/Contents/MacOS/moth autostart off
```

Then quit moth (menu bar icon → quit) and drag moth.app to the Trash. `~/.moth` and `~/.cache/moth` stay until you
delete them.

### Linux

**AppImage.**

```bash
chmod +x moth-<version>-x86_64.AppImage
./moth-<version>-x86_64.AppImage          # the first-run window, then the pet
```

Keep the AppImage in one place (for example `~/Applications/`) before you hook in agents or turn autostart on. The
hooks and the autostart entry run that file, so they break if it moves. After moving it, run
`./moth-*.AppImage hook --all` and `autostart on` again.

The AppImage needs FUSE 2: `libfuse2` (`libfuse2t64` on Ubuntu 24.04+, `fuse-libs` on Fedora). Without FUSE 2, add
`--appimage-extract-and-run`, or use the tarball.

The AppImage carries its own desktop entry and icon. AppImageLauncher or Gear Lever add them to your menu.

**Tarball.**

```bash
mkdir -p ~/.local/share && tar -C ~/.local/share -xzf moth-<version>-linux-x86_64.tar.gz   # ~/.local/share/moth/
sed "s|^Exec=moth|Exec=$HOME/.local/share/moth/moth|" ~/.local/share/moth/moth.desktop \
  > ~/.local/share/applications/moth.desktop
install -Dm644 ~/.local/share/moth/moth-256.png ~/.local/share/icons/hicolor/256x256/apps/moth.png
```

Then `~/.local/share/moth/moth` runs moth (`moth hooks`, `moth doctor`, … as arguments).

**Uninstall.** Run `moth unhook --all` and `moth autostart off` (through the AppImage or `~/.local/share/moth/moth`),
then delete the AppImage or the folder, the desktop entry and the icon.

Both Linux packages leave the desktop's own libraries to your system: glibc, libstdc++, glib, fontconfig, X11/xcb,
Wayland and GL. They bundle the xcb-util libraries Qt needs that some distros lack.

### Unsigned: the warnings and how to get past them

The installers aren't code-signed yet, so the first launch shows a warning. This is expected for unsigned apps. Get
the files only from this project's own release page.

- **Windows SmartScreen** says "Windows protected your PC". Click **More info**, then **Run anyway**. That's needed
  once, for the setup `.exe`. Some antivirus tools also distrust new unsigned PyInstaller programs; if yours
  quarantines `moth.exe` or `mothw.exe`, restore it and allow it.
- **macOS Gatekeeper** says "moth" can't be opened, or "Apple could not verify 'moth' is free of malware". To get
  past it:
  - On macOS 14 and older: Control-click (right-click) moth in Applications, choose **Open**, then **Open** again.
  - On macOS 15 and newer: try to open moth once. Then go to System Settings → Privacy & Security, scroll down to
    "moth was blocked…" and click **Open Anyway**.
  - Or, in Terminal, remove the download quarantine:
    `xattr -d com.apple.quarantine /Applications/moth.app` (or `xattr -dr` if a file inside still complains).

  If macOS says the app "is damaged", that's the same quarantine check on an unsigned app. The `xattr` command fixes
  it.
- **Linux** has no such check. Just `chmod +x`.

What signing would take (nothing has been bought):

- **macOS:** an Apple Developer Program membership ($99/year) for a "Developer ID Application" certificate. Every
  binary in moth.app has to be signed with the hardened runtime. The PyInstaller bootloader and torch's libraries
  likely need `com.apple.security.cs.allow-unsigned-executable-memory` /
  `disable-library-validation` entitlements; test that before relying on it. Then
  `xcrun notarytool submit --wait` for the .dmg and `xcrun stapler staple`. In CI that means the certificate (.p12)
  and an App Store Connect API key as secrets, and about 10 extra minutes per build. After that, Gatekeeper opens moth
  with no prompt.
- **Windows:** a code-signing certificate, either OV, EV, or Microsoft's Azure Trusted Signing (a monthly
  subscription, for individuals in some countries). `signtool sign /tr <timestamp> /td sha256 /fd sha256` would sign
  `moth.exe`, `mothw.exe`, `moth-hook.exe`, the setup `.exe` and the uninstaller (Inno Setup's `SignTool=`
  directive does the last two). Since 2024 even EV certificates no longer skip SmartScreen right away: reputation
  builds up with downloads of the signed files.

### Coding-agent hooks in the installed app

`moth hook` writes a command that runs the app's own small hook runner, not a Python:

```
"<install folder>/moth-hook" hook-run "<install folder>/_internal/moth/integrations/claude_code.py"
"/Applications/moth.app/Contents/MacOS/moth-hook" hook-run ".../Contents/Frameworks/moth/integrations/claude_code.py"
"/home/you/Applications/moth-0.3.0-x86_64.AppImage" hook-run integrations/claude_code.py
```

`moth-hook` is a separate stdlib-only executable, because a hook runs on every agent turn and Claude Code gives it
3 seconds. It starts in a fraction of the time the full moth program takes (that one carries torch and Qt). Measured
in CI on each OS (`packaging/smoke.py`), as the hook command is written, through the harness's shell:

| | how the agent runs it | first run | typical |
|---|---|---|---|
| Windows | Claude Code through Git Bash | 279 ms | 135 ms |
| Windows | Codex through PowerShell (mostly PowerShell's own start) | 287 ms | 264 ms |
| macOS (Apple silicon) | through bash / sh | 104–259 ms | 88–122 ms |
| Linux tarball | through bash / sh | 66 ms | 66 ms |
| Linux AppImage (a FUSE mount on every run) | through bash / sh | 114 ms | 114 ms |

The full moth program would take 265–375 ms per hook. A hook in a Python install takes about 25 ms.

`moth hooks` reports these hooks as installed. `moth unhook` removes them, and so does the Windows uninstaller. If
you also have a Python install of moth, whichever one ran `moth hook` last owns the hook.

## Building the installers

CI builds them (`.github/workflows/installers.yml`, on demand from the Actions tab or on a `v*` tag; output:
artifacts). To build locally, on the OS you're building for, with Python 3.12:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu    # macOS: pip install torch
pip install -r packaging/requirements.txt
QT_QPA_PLATFORM=offscreen python packaging/make_icons.py build/icons
python -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyi packaging/moth.spec
packaging/linux/build.sh       # Linux: AppImage + tarball
packaging/macos/build.sh       # macOS: ad-hoc signed .dmg
iscc /DAppVersion=$(cat moth/VERSION) /DSourceDir=dist\moth /DIconFile=build\icons\moth.ico /DOutputDir=dist packaging\windows\moth.iss   # Windows
```

The pieces:

- **`packaging/moth.spec`** is one PyInstaller *onedir* app. It isn't onefile, because a onefile exe unpacks itself on
  every start. The spec leaves out torch's C++ tests and headers, and on Linux the GTK theme plugin and the
  libraries the host must provide.
- **`packaging/moth_app.py`** is the entry point. It opens the first-run window or the pet, runs `-m` and `-c` like
  `python` (moth starts its server as `<app> -m moth.serve`), and runs every `moth` command.
- **`packaging/moth_firstrun.py`** is the first-run window: moth/setup.py's steps on a worker thread.
- **`packaging/moth_hook.py`** is the hook runner.
- **`packaging/smoke.py`** is the CI smoke test. It runs in a scratch home on a spare port, and you can run it against
  a local `dist/` too:
  `python packaging/make_fixture.py build/fixture && python packaging/smoke.py --app dist/moth/moth --model build/fixture --out smoke-out`
  On Linux, run it under `xvfb-run`.

Credits: the silkmoth icon is "202206 Silkmoth adult female icon" by DataBase Center for Life Science (DBCLS),
CC BY 4.0, rendered unmodified at icon sizes ([moth/assets/CREDITS.md](../moth/assets/CREDITS.md)). The apps
carry moth's LICENSE, NOTICE and LICENSE-LFM in their `licenses/` folder.
