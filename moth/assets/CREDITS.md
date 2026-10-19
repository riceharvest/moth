# Art and font credits

moth uses one piece of third-party art, two third-party fonts and one third-party sound. None of them is covered by
moth's Apache-2.0 license.

- The art (desktop pet, Petdex sprite sheet, chat page header icon and favicon) is used under CC BY 4.0. See
  "202206 Silkmoth adult female icon" below.
- The fonts (the chat page's Fraunces and JetBrains Mono, embedded in `moth/static/index.html`) are used under the
  SIL Open Font License 1.1. See "Fonts" below.

## 202206 Silkmoth adult female icon

- Title:   202206 Silkmoth adult female icon
- Author:  DataBase Center for Life Science (DBCLS), doi:10.7875/togopic.2022.231
- Source:  https://commons.wikimedia.org/wiki/File:202206_Silkmoth_adult_female_icon.svg
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Files in this folder:

| file | what it is | changes |
|---|---|---|
| `silkmoth_adult_female_icon.svg` | the original SVG from Wikimedia Commons | none |

(`candidates/` holds the other art that was considered. It is not used and is not shipped in the release.)

How moth uses the art. These are modifications, all mechanical. Nothing was drawn new:

- `moth/pet.py` (the desktop pet) renders the SVG with QtSvg and cuts it along the body into three layers: the left
  wing, the right wing and the body (with the antennae). The wings and body share one outline path in the SVG, so
  the cut is a clip polygon that follows the body's outline. Its frames squash each wing toward its root and turn it
  a few degrees, then draw the body on top: a 5-frame wingbeat, a resting pose (wings narrow and swept back) and a
  dozing pose (swept back further). The pet scales the frames and leans them up to 14 degrees toward where it
  flies. Dozing adds "z" glyphs drawn from a font, not art.
- `python -m moth.pet --export-petdex` packs the same frames into the Petdex 8x9 grid (1536x1872), scaled to
  160 px wide and leaned or offset per row. Its `CREDITS.txt` and `pet.json` carry this credit.
- `moth/static/index.html` inlines the SVG with these changes: coordinates rounded to 0.1, the CSS classes
  replaced by fill attributes, and the art drawn three times, clipped to the left wing, the right wing and the body.
  While moth thinks, CSS squashes and turns the wings the same way the pet does. The favicon is a 64x64 PNG rendering
  of the art.

## Fonts

The chat page (`moth/static/index.html`) embeds both fonts as base64 `@font-face` data, so it loads nothing from
the network (it used to load them from Google Fonts).

- Fraunces, by Undercase Type (Phaedra Charles, Flavia Zimbardi). Copyright 2018 The Fraunces Project Authors.
  - Source:  https://github.com/google/fonts/tree/main/ofl/fraunces (`Fraunces[SOFT,WONK,opsz,wght].ttf`), from
    upstream https://github.com/undercasetype/Fraunces at commit d6d385783609ceb11ac0f220f3abd9f1631c8a36
  - License: SIL Open Font License 1.1, `fonts/OFL-Fraunces.txt`
- JetBrains Mono, by JetBrains (Philipp Nurullin, Konstantin Bulenkov). Copyright 2020 The JetBrains Mono Project
  Authors.
  - Source:  https://github.com/google/fonts/tree/main/ofl/jetbrainsmono (`JetBrainsMono[wght].ttf`), from
    upstream https://github.com/JetBrains/JetBrainsMono at commit 19371302b95d218af43299bce79ddbddd0bc364d
  - License: SIL Open Font License 1.1, `fonts/OFL-JetBrainsMono.txt`

Neither license declares a Reserved Font Name, so the modified fonts keep their names.

Changes (mechanical, with fontTools 4.67; nothing was drawn new). The page uses weights 400 and 600, and Fraunces's
optical sizes, so the embedded fonts keep only those ranges:

```sh
fonttools varLib.instancer -o Fraunces-inst.ttf 'Fraunces[SOFT,WONK,opsz,wght].ttf' SOFT=0 WONK=1 opsz=9:144 wght=400:600
fonttools varLib.instancer -o JBM-inst.ttf 'JetBrainsMono[wght].ttf' wght=400:600
U='U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2190-2199,U+2212,U+2215,U+FEFF,U+FFFD'
pyftsubset Fraunces-inst.ttf --unicodes="$U,U+25B8,U+25BE" --flavor=woff2 --no-hinting --desubroutinize --output-file=fraunces-latin.woff2
pyftsubset JBM-inst.ttf --unicodes="$U,U+2500-259F,U+25A0-25FF,U+2713,U+2717" --flavor=woff2 --no-hinting --desubroutinize --output-file=jetbrains-mono-latin.woff2
```

That is the axes Google Fonts serves for this page (SOFT and WONK at their defaults), the Latin subset plus arrows
(and box drawing and shapes for the mono font), no hinting, as woff2: 60 KB and 31 KB, about 123 KB as base64.
Text outside the subset falls back to the page's next font (Georgia/serif, monospace).

Files in `fonts/`:

| file | what it is |
|---|---|
| `OFL-Fraunces.txt` | Fraunces's copyright line and the SIL OFL 1.1, from google/fonts, unmodified |
| `OFL-JetBrainsMono.txt` | JetBrains Mono's copyright line and the SIL OFL 1.1, from google/fonts, unmodified |

The desktop pet asks Qt for "Fraunces", then Georgia, DejaVu Serif, serif. It doesn't bundle or download fonts, so
it uses Fraunces only if it is installed on the system.

## Sound

`sounds/chirp.wav` is the desktop pet's optional "your agent is done" sound (config `chirp`, off by default).

- Title:   `confirmation_001.ogg`, from the "Interface Sounds" pack (1.0, 2020-02-11)
- Author:  Kenney (www.kenney.nl)
- Source:  https://kenney.nl/assets/interface-sounds (`kenney_interface-sounds.zip`, sha256
  f2193d072726d6758a5f7871b2dcc54dcce0d5c35c6f0a62f92549b327c81232)
- License: Creative Commons Zero (CC0 1.0), https://creativecommons.org/publicdomain/zero/1.0/ (the pack's
  `License.txt`). Credit isn't required; it's given anyway.

Changes (mechanical, with ffmpeg; nothing was made new): 12 dB quieter, resampled to 22.05 kHz mono 16-bit WAV
(so Windows' `winsound` and `aplay` can play it without a decoder), metadata dropped:

```sh
ffmpeg -i confirmation_001.ogg -af volume=-12dB -ar 22050 -ac 1 -c:a pcm_s16le -map_metadata -1 -fflags +bitexact chirp.wav
```
