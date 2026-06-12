"""Confirmed-squad strength from the official 26-man World Cup squads."""

from __future__ import annotations

import json
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

from . import config as C
from .features import _STRENGTH, edition_table_filled
from .worldcup2026 import EDITION_2026

# football-data.org team-name spellings -> the model's canonical names
_FD_ALIAS = {"Bosnia-Herzegovina": "Bosnia and Herzegovina", "Cape Verde Islands": "Cape Verde"}


def canon(name: str) -> str:
    return _FD_ALIAS.get(name, name)


WC_TEAMS_FILE = C.RAW / "footballdata_api" / "WC_teams.json"
MIN_MATCHED = 14          # below this, fall back to the nationality-pool proxy
FUZZY_CUTOFF = 80

# canonical team -> EA FC 26 "Nation" spelling (only where it differs)
_EA_NATION = {
    "Netherlands": "Holland", "Ivory Coast": "Cote d'Ivoire",
    "Cape Verde": "Cape Verde Islands", "Czechia": "Czech Republic",
    "South Korea": "Korea Republic",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace(".", " ").replace("-", " ").split())


def _group(position: str) -> str:
    p = position.lower()
    if "keeper" in p or p == "gk":
        return "GK"
    if "back" in p or "defen" in p or "cb" in p:
        return "DEF"
    if "midfield" in p or p in ("cm", "cdm", "cam", "dm"):
        return "MID"
    return "ATT"


def load_squads() -> dict[str, list[tuple[str, str]]]:
    """{canonical team: [(player_name, position), ...]} from the confirmed squads."""
    teams = json.load(open(WC_TEAMS_FILE))["teams"]
    out = {}
    for t in teams:
        sq = t.get("squad", [])
        if sq:
            out[canon(t["name"])] = [(p["name"], p.get("position", "")) for p in sq]
    return out


def _strength_from_players(players: list[tuple[float, str]]) -> dict:
    """Squad-strength columns from a list of (overall, position-group)."""
    ov = sorted((o for o, _ in players), reverse=True)

    def grp(g, n):
        vals = sorted((o for o, gg in players if gg == g), reverse=True)[:n]
        return sum(vals) / len(vals) if vals else float("nan")

    def best_xi(n=11, max_gk=1):
        """Top-n by overall, but no more than ``max_gk`` keepers (a real XI"""
        chosen, gk = [], 0
        for o, g in sorted(players, key=lambda x: -x[0]):
            if g == "GK":
                if gk >= max_gk:
                    continue
                gk += 1
            chosen.append(o)
            if len(chosen) == n:
                break
        return sum(chosen) / len(chosen) if chosen else float("nan")

    gks = [o for o, gg in players if gg == "GK"]
    return {
        "ovr_top23": sum(ov[:23]) / len(ov[:23]),
        "ovr_top11": best_xi(11, max_gk=1),
        "ovr_top3": sum(ov[:3]) / len(ov[:3]),
        "gk": max(gks) if gks else float("nan"),
        "def_top": grp("DEF", 5), "mid_top": grp("MID", 4), "att_top": grp("ATT", 3),
        "depth80": sum(1 for o in ov if o >= 80),
    }


def confirmed_strength_table():
    """Squad-strength table for the 48 WC teams (confirmed squads, proxy fallback)."""
    squads = load_squads()
    fc = pd.read_csv(C.RAW / "fifa" / "players_26.csv", low_memory=False)
    fc["_n"] = fc["Name"].map(_norm)
    fc["_nat"] = fc["Nation"].map(_norm)
    proxy = edition_table_filled(EDITION_2026)

    rows, report = {}, []
    for team, players in squads.items():
        nat = _norm(_EA_NATION.get(team, team))
        cand = fc[fc["_nat"] == nat]
        matched = []
        if len(cand) >= 5:
            names = cand["_n"].tolist()
            ovr = cand["OVR"].tolist()
            surn = [n.split()[-1] if n else "" for n in names]   # EA surnames
            for name, pos in players:
                pn = _norm(name)
                if not pn:
                    continue
                # exact surname match, disambiguated by best full-name similarity.
                sn = pn.split()[-1]
                same = [i for i, s in enumerate(surn) if s == sn]
                if same:
                    i = max(same, key=lambda i: fuzz.token_set_ratio(pn, names[i]))
                    matched.append((float(ovr[i]), _group(pos)))
                    continue
                # 2) token-set fuzzy fallback (subset-tolerant)
                hit = process.extractOne(pn, names, scorer=fuzz.token_set_ratio)
                if hit and hit[1] >= FUZZY_CUTOFF:
                    matched.append((float(ovr[hit[2]]), _group(pos)))
        if len(matched) >= MIN_MATCHED:
            s = _strength_from_players(matched)
            # backfill any NaN position columns from the proxy row
            pr = proxy.loc[team] if team in proxy.index else None
            for col in _STRENGTH:
                if pd.isna(s.get(col)) and pr is not None:
                    s[col] = float(pr[col])
            rows[team] = s
            report.append((team, len(matched), len(players), "confirmed"))
        else:
            src = proxy.loc[team] if team in proxy.index else None
            rows[team] = {c: float(src[c]) for c in _STRENGTH} if src is not None else None
            report.append((team, len(matched), len(players), "proxy"))

    tbl = pd.DataFrame.from_dict({k: v for k, v in rows.items() if v}, orient="index")[_STRENGTH]
    tbl.index.name = "team"
    rep = pd.DataFrame(report, columns=["team", "matched", "squad", "source"]).sort_values(
        ["source", "matched"])
    return tbl, rep


def main():
    tbl, rep = confirmed_strength_table()
    n_conf = (rep["source"] == "confirmed").sum()
    print(f"confirmed-squad strength: {n_conf}/{len(rep)} teams from real 26-man squads, "
          f"{len(rep) - n_conf} via nationality-pool proxy")
    print("\nConfirmed (sample, highest squad strength):")
    top = tbl.sort_values("ovr_top23", ascending=False).head(8)
    for t in top.index:
        r = rep[rep.team == t].iloc[0]
        print(f"  {t:14} ovr_top23 {tbl.loc[t, 'ovr_top23']:.1f}  ({r.matched}/{r.squad} matched)")
    print("\nFell back to proxy (EA FC lacks the players):")
    for r in rep[rep.source == "proxy"].itertuples(index=False):
        print(f"  {r.team:14} {r.matched}/{r.squad} matched")


if __name__ == "__main__":
    main()
