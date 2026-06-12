"""Render a social-media-ready PNG of the predicted 2026 bracket.

Builds a self-contained poster SVG (title band, bracket with embedded flag
images, trophy, champion banner, title-probability panel, and a prediction-date
stamp) and rasterises it to PNG with cairosvg. Flags are baked in as images so
they render on every platform. The poster uses its own large-format geometry
(bigger boxes and fonts than the on-screen widget) so it reads well when shared.

    python -m src.worldcup.poster --date "9 June 2026" --handle "@brighton_nkomo_"

Requires: cairosvg, and flag PNGs in data/raw/flags/ (downloaded from flagcdn).
"""

from __future__ import annotations

import argparse
import base64
import warnings

import cairosvg
import pandas as pd

from . import config as C
from .flags import _CODE
from .run_2026 import _ORDER, _SHORT, _trophy_svg
from .worldcup2026 import R16, QF, SF, FINAL

warnings.filterwarnings("ignore")

# ── Large-format poster geometry ─────────────────────────────────────
PW = 3000
BW, MH = 292, 80
COLX_L = [60, 380, 700, 1020]                      # R32, R16, QF, SF
COLX_R = [PW - x - BW for x in COLX_L]             # mirror
XC = PW / 2
FINAL_X = XC - BW / 2
TOP, BOT = 140, 1700
LEAF = (BOT - TOP) / 8
TH = 230                                           # title-header height
PH = 1760                                          # bracket-area height


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _flag_uris() -> dict:
    uris = {}
    for team, code in _CODE.items():
        path = C.RAW / "flags" / f"{code.lower()}.png"
        if path.exists():
            uris[team] = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    return uris


def build_poster(date_str: str, handle: str = "", bracket_csv=None, sim_csv=None,
                 title_text=None, subtitle_text=None, pill_note="before kickoff") -> str:
    d = pd.read_csv(bracket_csv or C.OUT / "predictions_2026_bracket.csv")
    bracket = {int(r.match): (r.team_a, r.team_b, r.predicted_winner) for r in d.itertuples()}
    sim = pd.read_csv(sim_csv or C.OUT / "simulation_2026.csv")
    uri = _flag_uris()

    def short(n):
        return _SHORT.get(n, n)

    # geometry
    cy, xof, side = {}, {}, {}
    for s in ("L", "R"):
        for i, m in enumerate(_ORDER[s]["R32"]):
            cy[m] = TOP + LEAF * (i + 0.5)
        for par, ch in (("R16", "R32"), ("QF", "R16"), ("SF", "QF")):
            c = _ORDER[s][ch]
            for j, m in enumerate(_ORDER[s][par]):
                cy[m] = (cy[c[2 * j]] + cy[c[2 * j + 1]]) / 2
        for nm, x in zip(("R32", "R16", "QF", "SF"), COLX_L if s == "L" else COLX_R):
            for m in _ORDER[s][nm]:
                xof[m] = x
                side[m] = s
    cy[104], xof[104] = (TOP + BOT) / 2, FINAL_X
    parent = {}
    for dd in (R16, QF, SF, FINAL):
        for p, (a, b) in dd.items():
            parent[a] = parent[b] = p

    def conn(m):
        p = parent[m]
        if side.get(m, "L") == "L":
            x1, x2 = xof[m] + BW, xof[p]
        else:
            x1, x2 = xof[m], xof[p] + BW
        mx = (x1 + x2) / 2
        return (f'<polyline points="{x1:.1f},{cy[m]:.1f} {mx:.1f},{cy[m]:.1f} '
                f'{mx:.1f},{cy[p]:.1f} {x2:.1f},{cy[p]:.1f}" fill="none" '
                f'stroke="#cfd4de" stroke-width="2"/>')

    def trow(name, x, ty, win):
        fill = "#0d1017" if win else "#9aa1ad"
        wt = "700" if win else "400"
        u = uri.get(name)
        img = (f'<image x="{x + 16:.1f}" y="{ty - 13:.1f}" width="40" height="26" '
               f'preserveAspectRatio="xMidYMid slice" href="{u}"/>' if u else "")
        return img + (f'<text x="{x + 66:.1f}" y="{ty:.1f}" font-size="24" font-weight="{wt}" '
                      f'fill="{fill}" dominant-baseline="central">{_esc(short(name))}</text>')

    def box(m):
        a, b, w = bracket[m]
        x, top = xof[m], cy[m] - MH / 2
        stroke, sw = ("#E3A92B", "3") if m == 104 else ("#d7dbe4", "1.6")
        wy = top if a == w else top + MH / 2
        return (
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{BW}" height="{MH}" rx="10" fill="#fff" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
            f'<rect x="{x + 2.5:.1f}" y="{wy + 2.5:.1f}" width="{BW - 5}" height="{MH / 2 - 5:.1f}" '
            f'rx="6" fill="#F6C544" fill-opacity=".20"/>'
            f'<rect x="{x + 2.5:.1f}" y="{wy + 2.5:.1f}" width="5.5" height="{MH / 2 - 5:.1f}" fill="#E3A92B"/>'
            f'<line x1="{x:.1f}" y1="{top + MH / 2:.1f}" x2="{x + BW:.1f}" y2="{top + MH / 2:.1f}" '
            f'stroke="#eef0f4" stroke-width="1.4"/>'
            + trow(a, x, top + MH / 4, a == w) + trow(b, x, top + 3 * MH / 4, b == w))

    # Large round labels, stacked on two lines so they fit inside each column.
    head2 = {"Round of 32": ("ROUND OF", "32"), "Round of 16": ("ROUND OF", "16"),
             "Quarter-finals": ("QUARTER", "FINALS"), "Semi-finals": ("SEMI", "FINALS")}

    def head(cx, lines):
        a = ('font-size="40" font-weight="800" fill="#9098a6" text-anchor="middle" '
             'letter-spacing="1.5"')
        if len(lines) == 1:
            return f'<text x="{cx:.1f}" y="110" {a}>{lines[0]}</text>'
        return (f'<text x="{cx:.1f}" y="86" {a}>{lines[0]}</text>'
                f'<text x="{cx:.1f}" y="128" {a}>{lines[1]}</text>')

    heads = []
    for cols in (COLX_L, COLX_R):
        for nm, x in zip(["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals"], cols):
            heads.append(head(x + BW / 2, head2[nm]))
    heads.append(head(XC, ("FINAL",)))

    conns = "".join(conn(m) for m in xof if m != 104)
    boxes = "".join(box(m) for m in xof)
    final_top = cy[104] - MH / 2
    trophy = _trophy_svg(XC, final_top - 30, 2.8)

    champ = bracket[104][2]
    cyb = cy[104] + MH / 2 + 46
    cu = uri.get(champ)
    cimg = (f'<image x="{XC - 178:.1f}" y="{cyb + 46:.1f}" width="46" height="30" '
            f'preserveAspectRatio="xMidYMid slice" href="{cu}"/>' if cu else "")
    champ_b = (
        f'<rect x="{XC - 230:.1f}" y="{cyb:.1f}" width="460" height="100" rx="18" '
        f'fill="#FBF6E6" stroke="#C9A227" stroke-width="2.4"/>'
        f'<text x="{XC:.1f}" y="{cyb + 32:.1f}" font-size="19" font-weight="700" fill="#A07A1F" '
        f'text-anchor="middle" letter-spacing="2.5">PREDICTED CHAMPION</text>'
        f'{cimg}<text x="{XC + 10:.1f}" y="{cyb + 76:.1f}" font-size="38" font-weight="800" '
        f'fill="#11151f" text-anchor="middle">{_esc(champ)}</text>')

    oy = cyb + 172
    odds = [f'<text x="{XC:.1f}" y="{oy:.1f}" font-size="22" font-weight="700" fill="#9098a6" '
            f'text-anchor="middle" letter-spacing="1.8">TITLE PROBABILITY (MONTE CARLO)</text>']
    for k, r in enumerate(sim.head(8).itertuples(index=False)):
        ry, bx = oy + 50 + k * 52, XC - 40
        u = uri.get(r.team)
        fimg = (f'<image x="{XC - 340:.1f}" y="{ry - 16:.1f}" width="40" height="26" '
                f'preserveAspectRatio="xMidYMid slice" href="{u}"/>' if u else "")
        odds.append(
            fimg
            + f'<text x="{XC - 288:.1f}" y="{ry:.1f}" font-size="24" font-weight="600" '
              f'fill="#11151f" dominant-baseline="central">{_esc(short(r.team))}</text>'
            + f'<rect x="{bx:.1f}" y="{ry - 10:.1f}" width="320" height="20" rx="10" fill="#eef0f4"/>'
            + f'<rect x="{bx:.1f}" y="{ry - 10:.1f}" width="{r.Champion * 320 / 0.16:.0f}" height="20" '
              f'rx="10" fill="#E9B83F"/>'
            + f'<text x="{bx + 338:.1f}" y="{ry:.1f}" font-size="22" font-weight="700" fill="#3a4150" '
              f'dominant-baseline="central">{r.Champion * 100:.1f}%</text>')

    body = f'<g transform="translate(0,{TH})">{"".join(heads)}{conns}{boxes}{trophy}{champ_b}{"".join(odds)}</g>'

    # centred title (unchanged size) + larger subtitle + green prediction pill
    _note = f"  {pill_note}" if pill_note else ""
    visual = f"PREDICTED {date_str}{_note}"
    tw = len(visual) * 15.0
    sx = XC - tw / 2
    pill_w = tw + 110
    _note_tspan = f'<tspan fill="#ffffff"> &#183; {_esc(pill_note)}</tspan>' if pill_note else ""
    stamp = (
        f'<rect x="{XC - pill_w / 2:.0f}" y="158" width="{pill_w:.0f}" height="54" rx="27" fill="#15683f"/>'
        f'<text x="{sx:.0f}" y="193" font-size="24" font-weight="700">'
        f'<tspan fill="#E9B83F">PREDICTED </tspan>'
        f'<tspan fill="#ffffff">{_esc(date_str)}</tspan>{_note_tspan}</text>')
    title = (
        f'<rect x="0" y="0" width="{PW}" height="12" fill="url(#band)"/>'
        f'<text x="{XC:.0f}" y="84" font-size="58" font-weight="800" fill="#11151f" text-anchor="middle">'
        f'{_esc(title_text or "FIFA World Cup 2026: Predicted Knockout Bracket")}</text>'
        f'<text x="{XC:.0f}" y="130" font-size="32" fill="#737a88" text-anchor="middle">'
        f'{_esc(subtitle_text or "A machine-learning forecast, simulated through the official 48-team knockout bracket.")}</text>{stamp}')
    foot = (f'<text x="{PW - 70:.0f}" y="{PH + TH - 36:.0f}" font-size="30" font-weight="700" '
            f'fill="#C9A227" text-anchor="end">{_esc(handle)}</text>' if handle else "")

    return (
        f'<svg viewBox="0 0 {PW} {PH + TH}" xmlns="http://www.w3.org/2000/svg" '
        'font-family="DejaVu Sans, Arial, sans-serif">'
        '<defs>'
        '<linearGradient id="band" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#E6007E"/><stop offset=".2" stop-color="#F5821F"/>'
        '<stop offset=".4" stop-color="#FFC400"/><stop offset=".6" stop-color="#1FB82E"/>'
        '<stop offset=".8" stop-color="#1A86FF"/><stop offset="1" stop-color="#7B2FF7"/></linearGradient>'
        '<linearGradient id="gold" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FCE7A8"/>'
        '<stop offset=".45" stop-color="#E9B83F"/><stop offset="1" stop-color="#B07E1C"/></linearGradient>'
        '<linearGradient id="green" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2f7d52"/>'
        '<stop offset="1" stop-color="#1c5236"/></linearGradient></defs>'
        f'<rect x="0" y="0" width="{PW}" height="{PH + TH}" fill="#ffffff"/>'
        f'{title}{body}{foot}</svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="9 June 2026")
    ap.add_argument("--handle", default="")
    ap.add_argument("--width", type=int, default=2800)
    args = ap.parse_args()
    svg = build_poster(args.date, args.handle)
    out = C.OUT / "bracket_2026_share.png"
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(out), output_width=args.width)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
