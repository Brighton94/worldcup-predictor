"""Build the 2026 World Cup forecast and render the predicted bracket tree."""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from . import config as C
from .worldcup2026 import build_2026, R32, R16, QF, SF, FINAL
from .flags import flag_emoji

warnings.filterwarnings("ignore")

# Vertical (top-to-bottom) ordering of matches so the columns read as a tree.
R32_ORDER = [74, 77, 73, 75, 83, 84, 81, 82, 76, 78, 79, 80, 86, 88, 85, 87]
R16_ORDER = [89, 90, 93, 94, 91, 92, 95, 96]
QF_ORDER = [97, 98, 99, 100]
SF_ORDER = [101, 102]
ROUND_OF = {**{m: "Round of 32" for m in R32}, **{m: "Round of 16" for m in R16},
            **{m: "Quarter-final" for m in QF}, **{m: "Semi-final" for m in SF},
            104: "Final"}


def _bracket_rows(bracket: dict) -> pd.DataFrame:
    rows = []
    for m in list(R32) + list(R16) + list(QF) + list(SF) + [104]:
        a, b, w = bracket[m]
        rows.append({"round": ROUND_OF[m], "match": m, "team_a": a, "team_b": b,
                     "predicted_winner": w})
    return pd.DataFrame(rows)


# HTML rendering

# SVG bracket geometry (symmetric, with connector lines)
_W, _H = 1980, 1560
_BW, _MH = 192, 48                       # match box width / height (two rows)
_XC = _W / 2                             # centre
_COLX_L = [40, 248, 456, 664]           # left edges: R32, R16, QF, SF
_COLX_R = [_W - 40 - _BW, _W - 248 - _BW, _W - 456 - _BW, _W - 664 - _BW]
_FINAL_X = _XC - _BW / 2
_TOP, _BOT = 86, _H - 60
_LEAF = (_BOT - _TOP) / 8

_ORDER = {
    "L": {"R32": [74, 77, 73, 75, 83, 84, 81, 82], "R16": [89, 90, 93, 94],
          "QF": [97, 98], "SF": [101]},
    "R": {"R32": [76, 78, 79, 80, 86, 88, 85, 87], "R16": [91, 92, 95, 96],
          "QF": [99, 100], "SF": [102]},
}
_SHORT = {"Bosnia and Herzegovina": "Bosnia & Herz.", "United States": "USA",
          "South Korea": "S. Korea", "South Africa": "S. Africa",
          "New Zealand": "N. Zealand", "Saudi Arabia": "Saudi Arabia"}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _label(name: str) -> str:
    e = flag_emoji(name)
    return _esc((f"{e} " if e else "") + _SHORT.get(name, name))


def _trophy_svg(cx: float, bottom: float, scale: float) -> str:
    """Original detailed gold-trophy vector: globe on a twisting body with the"""
    tx, ty = cx - 50 * scale, bottom - 150 * scale
    return (
        f'<g transform="translate({tx:.1f},{ty:.1f}) scale({scale:.3f})">'
        '<circle cx="50" cy="30" r="24" fill="url(#gold)"/>'
        '<ellipse cx="50" cy="30" rx="9" ry="24" fill="none" stroke="#9c7a22" stroke-width="1" opacity=".4"/>'
        '<ellipse cx="50" cy="30" rx="24" ry="9" fill="none" stroke="#9c7a22" stroke-width="1" opacity=".4"/>'
        '<ellipse cx="42" cy="22" rx="7" ry="10" fill="#ffffff" opacity=".18"/>'
        '<path d="M30 50 C 20 66 40 72 38 98 C 37 114 30 118 45 122 L 55 122 '
        'C 70 118 63 114 62 98 C 60 72 80 66 70 50 Z" fill="url(#gold)"/>'
        '<path d="M50 52 C 43 72 57 94 50 120" fill="none" stroke="#a9821f" stroke-width="2" opacity=".5"/>'
        '<path d="M37 55 C 31 71 41 92 41 116" fill="none" stroke="#ffffff" stroke-width="2" opacity=".15"/>'
        '<polygon points="36,122 64,122 67,130 33,130" fill="url(#gold)"/>'
        '<polygon points="33,130 67,130 70,144 30,144" fill="url(#green)"/>'
        '<polygon points="30,144 70,144 72,150 28,150" fill="url(#gold)"/>'
        '<line x1="32" y1="137" x2="68" y2="137" stroke="#d9b24a" stroke-width="1" opacity=".6"/>'
        '</g>')


def render_svg(bracket: dict) -> str:
    # match-id -> vertical centre and left-edge x
    cy, xof, side_of = {}, {}, {}
    for side in ("L", "R"):
        r32 = _ORDER[side]["R32"]
        for i, m in enumerate(r32):
            cy[m] = _TOP + _LEAF * (i + 0.5)
        for parent, child in (("R16", "R32"), ("QF", "R16"), ("SF", "QF")):
            ch = _ORDER[side][child]
            for j, m in enumerate(_ORDER[side][parent]):
                cy[m] = (cy[ch[2 * j]] + cy[ch[2 * j + 1]]) / 2
        cols = _COLX_L if side == "L" else _COLX_R
        for name, x in zip(("R32", "R16", "QF", "SF"), cols):
            for m in _ORDER[side][name]:
                xof[m] = x
                side_of[m] = side
    cy[104] = _H / 2
    xof[104] = _FINAL_X
    left_ids = {m for n in _ORDER["L"].values() for m in n}

    parent = {}
    for d in (R16, QF, SF, FINAL):
        for p, (a, b) in d.items():
            parent[a] = p
            parent[b] = p

    def connector(m):
        p = parent[m]
        yc, yp = cy[m], cy[p]
        if side_of.get(m, "L") == "L":
            x1, x2 = xof[m] + _BW, xof[p]
        else:
            x1, x2 = xof[m], xof[p] + _BW
        mx = (x1 + x2) / 2
        return (f'<polyline points="{x1:.1f},{yc:.1f} {mx:.1f},{yc:.1f} '
                f'{mx:.1f},{yp:.1f} {x2:.1f},{yp:.1f}" fill="none" '
                f'stroke="#cfd4de" stroke-width="1.6"/>')

    def row(name, x, ty, win):
        fill = "#0d1017" if win else "#9aa1ad"
        wt = "700" if win else "400"
        return (f'<text x="{x + 12:.1f}" y="{ty:.1f}" font-size="14" font-weight="{wt}" '
                f'fill="{fill}" dominant-baseline="central">{_label(name)}</text>')

    def box(m):
        a, b, w = bracket[m]
        x, top = xof[m], cy[m] - _MH / 2
        stroke, sw = ("#E3A92B", "2") if m == 104 else ("#d7dbe4", "1.2")
        wy = top if a == w else top + _MH / 2
        return (
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{_BW}" height="{_MH}" rx="7" '
            f'fill="#ffffff" stroke="{stroke}" stroke-width="{sw}"/>'
            f'<rect x="{x + 1.5:.1f}" y="{wy + 1.5:.1f}" width="{_BW - 3}" height="{_MH / 2 - 3:.1f}" '
            f'rx="4" fill="#F6C544" fill-opacity=".20"/>'
            f'<rect x="{x + 1.5:.1f}" y="{wy + 1.5:.1f}" width="3.5" height="{_MH / 2 - 3:.1f}" fill="#E3A92B"/>'
            f'<line x1="{x:.1f}" y1="{top + _MH / 2:.1f}" x2="{x + _BW:.1f}" y2="{top + _MH / 2:.1f}" '
            f'stroke="#eef0f4" stroke-width="1"/>'
            + row(a, x, top + _MH / 4, a == w) + row(b, x, top + 3 * _MH / 4, b == w))

    # column headers
    heads = []
    names = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals"]
    for cols in (_COLX_L, _COLX_R):
        for nm, x in zip(names, cols):
            heads.append(f'<text x="{x + _BW / 2:.1f}" y="46" font-size="13" font-weight="700" '
                         f'fill="#9098a6" text-anchor="middle" letter-spacing="1">{nm.upper()}</text>')
    heads.append(f'<text x="{_XC:.1f}" y="46" font-size="13" font-weight="700" '
                 f'fill="#9098a6" text-anchor="middle" letter-spacing="1">FINAL</text>')

    conns = "".join(connector(m) for m in xof if m != 104)
    boxes = "".join(box(m) for m in xof)
    trophy = _trophy_svg(_XC, _H / 2 - _MH / 2 - 26, 1.45)

    champ = bracket[104][2]
    cyb = _H / 2 + _MH / 2 + 30
    champ_banner = (
        f'<rect x="{_XC - 150:.1f}" y="{cyb:.1f}" width="300" height="64" rx="12" '
        f'fill="#FBF6E6" stroke="#C9A227" stroke-width="1.6"/>'
        f'<text x="{_XC:.1f}" y="{cyb + 21:.1f}" font-size="12" font-weight="700" fill="#A07A1F" '
        f'text-anchor="middle" letter-spacing="1.5">PREDICTED CHAMPION</text>'
        f'<text x="{_XC:.1f}" y="{cyb + 45:.1f}" font-size="22" font-weight="800" fill="#11151f" '
        f'text-anchor="middle">{_label(champ)}</text>')

    return (
        f'<svg viewBox="0 0 {_W} {_H}" width="100%" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="-apple-system,Segoe UI,Roboto,\'Apple Color Emoji\',\'Segoe UI Emoji\',Helvetica,Arial,sans-serif">'
        '<defs>'
        '<linearGradient id="gold" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#FCE7A8"/><stop offset=".45" stop-color="#E9B83F"/>'
        '<stop offset="1" stop-color="#B07E1C"/></linearGradient>'
        '<linearGradient id="green" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#2f7d52"/><stop offset="1" stop-color="#1c5236"/></linearGradient>'
        '</defs>'
        f'{"".join(heads)}{conns}{boxes}{trophy}{champ_banner}</svg>')


# Light theme, white background, 2026 tournament colour band.
_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#fff;margin:0;padding:24px;color:#11151f}
.wc{background:#fff;border:1px solid #e6e8ee;border-radius:16px;overflow:hidden;max-width:1320px;margin:0 auto}
.band{height:6px;background:linear-gradient(90deg,#E6007E,#F5821F,#FFC400,#1FB82E,#1A86FF,#7B2FF7)}
.hd{padding:16px 22px 4px}
.hd .ti{font-size:19px;font-weight:800;color:#11151f}
.hd .su{font-size:12px;color:#737a88;margin-top:2px}
.svgwrap{padding:4px 16px}
.odds{padding:2px 22px 20px}
.odds .oh{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#9098a6;font-weight:700;margin-bottom:6px}
.orow{display:flex;align-items:center;gap:10px;margin:3px 0;font-size:12px;color:#3a4150}
.orow .on{width:150px}.orow .ov{width:48px;text-align:right;color:#737a88}
.obar{flex:1;height:8px;max-width:280px;background:#eef0f4;border-radius:4px;overflow:hidden}
.obar i{display:block;height:100%;background:linear-gradient(90deg,#F5821F,#FFC400)}
"""


def _figure(result: dict) -> str:
    """Inner HTML (panel + band + title + SVG bracket + odds)."""
    b = result["bracket"]
    sim = result["sim"].head(8)
    odds = "".join(
        f'<div class="orow"><span class="on">{_label(r.team)}</span>'
        f'<span class="obar"><i style="width:{r.Champion * 100 * 2.6:.0f}px"></i></span>'
        f'<span class="ov">{r.Champion * 100:.1f}%</span></div>'
        for r in sim.itertuples(index=False)
    )
    return (
        '<div class="wc"><div class="band"></div>'
        '<div class="hd"><div class="ti">FIFA World Cup 2026: Predicted Knockout Bracket</div>'
        '<div class="su">Forecast from team Elo (current to June 2026) and EA FC 26 squad ratings, '
        'simulated through FIFA&rsquo;s official 48-team knockout bracket.</div></div>'
        f'<div class="svgwrap">{render_svg(b)}</div>'
        f'<div class="odds"><div class="oh">Title probability (Monte-Carlo, top 8)</div>{odds}</div></div>')


def render_html(result: dict) -> str:
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>FIFA World Cup 2026 - predicted bracket</title>'
            f'<style>{_CSS}</style></head><body>{_figure(result)}</body></html>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    args = ap.parse_args()

    res = build_2026(n_sims=args.sims)
    out = C.OUT

    _bracket_rows(res["bracket"]).to_csv(out / "predictions_2026_bracket.csv", index=False)
    res["sim"].to_csv(out / "simulation_2026.csv", index=False)
    gt = pd.DataFrame({g: order for g, order in res["group_table"].items()},
                      index=["1st", "2nd", "3rd", "4th"]).T
    gt.index.name = "group"
    gt.to_csv(out / "groups_2026.csv")
    (out / "bracket_2026.html").write_text(render_html(res), encoding="utf-8")

    b = res["bracket"]
    print(f"training matches: {res['n_train']}")
    print(f"Predicted champion: {b[104][2]}")
    print(f"Final: {b[101][2]} vs {b[102][2]}")
    print("\nTitle probabilities (top 8):")
    print(res["sim"].head(8).to_string(index=False))
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
