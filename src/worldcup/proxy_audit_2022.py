"""Audit: does the nationality-pool proxy over-rate deep squads (Brazil 2022)?"""

from __future__ import annotations

import warnings

import pandas as pd
from rapidfuzz import fuzz, process

from . import config as C
from .data import load_fifa_players, load_intl_results, load_tournaments
from .elo import compute_elo
from .features import edition_table_filled, FEATURES, _STRENGTH
from .model import train_eval
from .squads import _norm, _group, _strength_from_players

warnings.filterwarnings("ignore")

# FIFA 23 (released Sept 2022) is the edition active before the Nov 2022 World Cup.
EDITION = 2023

# FIFA short-name nicknames the fuzzy matcher would otherwise miss.
_ALIAS = {"vinicius junior": "vini jr"}

# Confirmed 26-man Qatar 2022 squads (position groups: Goalkeeper/Defence/Midfield/Offence).
SQUADS_2022 = {
    "Brazil": [
        ("Alisson", "Goalkeeper"), ("Ederson", "Goalkeeper"), ("Weverton", "Goalkeeper"),
        ("Dani Alves", "Defence"), ("Danilo", "Defence"), ("Alex Sandro", "Defence"),
        ("Alex Telles", "Defence"), ("Bremer", "Defence"), ("Eder Militao", "Defence"),
        ("Marquinhos", "Defence"), ("Thiago Silva", "Defence"),
        ("Casemiro", "Midfield"), ("Fred", "Midfield"), ("Fabinho", "Midfield"),
        ("Bruno Guimaraes", "Midfield"), ("Lucas Paqueta", "Midfield"), ("Everton Ribeiro", "Midfield"),
        ("Neymar", "Offence"), ("Vinicius Junior", "Offence"), ("Richarlison", "Offence"),
        ("Raphinha", "Offence"), ("Antony", "Offence"), ("Rodrygo", "Offence"),
        ("Gabriel Jesus", "Offence"), ("Gabriel Martinelli", "Offence"), ("Pedro", "Offence"),
    ],
    "Argentina": [
        ("Emiliano Martinez", "Goalkeeper"), ("Franco Armani", "Goalkeeper"), ("Geronimo Rulli", "Goalkeeper"),
        ("Nahuel Molina", "Defence"), ("Gonzalo Montiel", "Defence"), ("Cristian Romero", "Defence"),
        ("German Pezzella", "Defence"), ("Nicolas Otamendi", "Defence"), ("Lisandro Martinez", "Defence"),
        ("Marcos Acuna", "Defence"), ("Nicolas Tagliafico", "Defence"), ("Juan Foyth", "Defence"),
        ("Rodrigo De Paul", "Midfield"), ("Leandro Paredes", "Midfield"), ("Guido Rodriguez", "Midfield"),
        ("Alexis Mac Allister", "Midfield"), ("Enzo Fernandez", "Midfield"), ("Exequiel Palacios", "Midfield"),
        ("Alejandro Gomez", "Midfield"),
        ("Lionel Messi", "Offence"), ("Angel Di Maria", "Offence"), ("Lautaro Martinez", "Offence"),
        ("Julian Alvarez", "Offence"), ("Nicolas Gonzalez", "Offence"), ("Joaquin Correa", "Offence"),
        ("Paulo Dybala", "Offence"),
    ],
}


def _pool():
    p = load_fifa_players()
    p = p[p["edition"] == EDITION].copy()
    p["_n"] = p["short_name"].map(_norm)
    return p


def confirmed_strength(team: str, pool: pd.DataFrame) -> tuple[dict, int, list[str]]:
    """Strength of a team's confirmed squad, matched to FIFA short names by surname then fuzzy."""
    cand = pool[pool["nationality"] == team]
    names = cand["_n"].tolist()
    ovr = cand["overall"].astype(float).tolist()
    surn = [n.split()[-1] if n else "" for n in names]
    matched, miss = [], []
    for name, pos in SQUADS_2022[team]:
        pn = _ALIAS.get(_norm(name), _norm(name))
        sn = pn.split()[-1]
        same = [i for i, s in enumerate(surn) if s == sn]
        if same:
            i = max(same, key=lambda i: fuzz.token_set_ratio(pn, names[i]))
            matched.append((ovr[i], _group(pos)))
            continue
        hit = process.extractOne(pn, names, scorer=fuzz.token_set_ratio)
        if hit and hit[1] >= 86:
            matched.append((ovr[hit[2]], _group(pos)))
        else:
            miss.append(name)
    return _strength_from_players(matched), len(matched), miss


def _pre_tournament_elo() -> pd.Series:
    elo = compute_elo(load_intl_results())
    start = load_tournaments().set_index("tournament_id").loc["WC-2022", "start_date"]
    rows = []
    for _, r in elo[elo["date"] < start].iterrows():
        rows.append((r["date"], r["team1"], r["elo1_pre"]))
        rows.append((r["date"], r["team2"], r["elo2_pre"]))
    return pd.DataFrame(rows, columns=["d", "t", "e"]).sort_values("d").groupby("t").last()["e"]


def main():
    pool = _pool()
    proxy = edition_table_filled(EDITION)
    conf = {}
    for t in ("Brazil", "Argentina"):
        s, n, miss = confirmed_strength(t, pool)
        conf[t] = s
        print(f"{t}: matched {n}/26   unmatched: {miss}")

    gap_p = proxy.loc["Brazil", "ovr_top23"] - proxy.loc["Argentina", "ovr_top23"]
    gap_c = conf["Brazil"]["ovr_top23"] - conf["Argentina"]["ovr_top23"]
    print("\nsquad ovr_top23 (proxy -> confirmed):")
    for t in ("Brazil", "Argentina"):
        print(f"  {t:10} {proxy.loc[t, 'ovr_top23']:.1f} -> {conf[t]['ovr_top23']:.1f}")
    print(f"Brazil minus Argentina gap: proxy {gap_p:+.2f} -> confirmed {gap_c:+.2f}")

    elo = _pre_tournament_elo()
    d_elo = elo["Brazil"] - elo["Argentina"]
    model = train_eval(2022)["model"]

    def hh(src_b, src_a):
        r = {"d_elo": d_elo, "home_field": 0.0}
        for c in _STRENGTH:
            r[f"d_{c}"] = float(src_b[c]) - float(src_a[c])
        return model.predict_proba(pd.DataFrame([r])[FEATURES])[0]

    pp = hh(proxy.loc["Brazil"], proxy.loc["Argentina"])
    pc = hh(conf["Brazil"], conf["Argentina"])
    print(f"\npre-2022 Elo: Brazil {elo['Brazil']:.0f}, Argentina {elo['Argentina']:.0f} (d_elo {d_elo:+.0f})")
    print("Brazil vs Argentina (neutral) P(Brazil / draw / Argentina):")
    print(f"  proxy squads     : {pp[0]*100:4.1f} / {pp[1]*100:4.1f} / {pp[2]*100:4.1f}")
    print(f"  confirmed squads : {pc[0]*100:4.1f} / {pc[1]*100:4.1f} / {pc[2]*100:4.1f}")


if __name__ == "__main__":
    main()
