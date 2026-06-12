"""Monte-Carlo bracket simulation for round-advancement and title odds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import load_tournaments
from .elo import compute_elo, ratings_as_of
from .data import load_intl_results
from .features import _edition_table, _STRENGTH, FEATURES, strength_vector

# Standard 32-team R16 pairings by (group, rank). rank 0 = winner, 1 = runner-up
R16_PAIRS = [
    (("A", 0), ("B", 1)), (("C", 0), ("D", 1)),
    (("E", 0), ("F", 1)), (("G", 0), ("H", 1)),
    (("B", 0), ("A", 1)), (("D", 0), ("C", 1)),
    (("F", 0), ("E", 1)), (("H", 0), ("G", 1)),
]


def _group_membership(wc_year: int) -> dict[str, list[str]]:
    """The group draw (membership only -- not final standings) per group."""
    gs = pd.read_csv(C.WC_RAW / "group_standings.csv")
    gs = gs[gs["tournament_id"] == f"WC-{wc_year}"]
    out: dict[str, list[str]] = {}
    for grp, g in gs.groupby("group_name"):
        out[grp.replace("Group ", "")] = sorted(g["team_name"].unique())
    return out


def _pairwise_probs(model, teams, tbl, elo_at_start, host) -> dict:
    """{(t1, t2): (p1_win, p_draw, p2_win)} for all ordered team pairs."""
    rows, keys = [], []
    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                continue
            s1, s2 = strength_vector(tbl, t1), strength_vector(tbl, t2)
            hf = 1.0 if t1 == host else (-1.0 if t2 == host else 0.0)
            feat = {"d_elo": elo_at_start.get(t1, 1500.0) - elo_at_start.get(t2, 1500.0),
                    "home_field": hf}
            for c in _STRENGTH:
                feat[f"d_{c}"] = float(s1[c] - s2[c])
            rows.append(feat)
            keys.append((t1, t2))
    P = model.predict_proba(pd.DataFrame(rows)[FEATURES])
    return {k: P[i] for i, k in enumerate(keys)}


def simulate(wc_year: int, model, n_sims: int = 20000, seed: int = 7,
             strength_table: pd.DataFrame | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tid = f"WC-{wc_year}"
    tours = load_tournaments().set_index("tournament_id")
    start, host = tours.loc[tid, "start_date"], str(tours.loc[tid, "host_country"])
    tbl = strength_table if strength_table is not None else _edition_table(C.active_edition(start))
    elo_at_start = ratings_as_of(compute_elo(load_intl_results()), start)

    groups = _group_membership(wc_year)
    teams = [t for g in groups.values() for t in g]
    pw = _pairwise_probs(model, teams, tbl, elo_at_start, host)
    elo = {t: elo_at_start.get(t, 1500.0) for t in teams}

    rounds = ["R16", "QF", "SF", "Final", "Champion"]
    tally = {t: dict.fromkeys(rounds, 0) for t in teams}

    def adv_prob(t1, t2):  # knockout: draw -> coin flip
        p = pw[(t1, t2)]
        return p[0] + 0.5 * p[1]

    for _ in range(n_sims):
        # group stage
        winners: dict[str, list[str]] = {}
        for grp, members in groups.items():
            pts = dict.fromkeys(members, 0.0)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    o = rng.choice(3, p=pw[(a, b)])
                    if o == 0:
                        pts[a] += 3
                    elif o == 1:
                        pts[a] += 1; pts[b] += 1
                    else:
                        pts[b] += 3
            ranked = sorted(members, key=lambda t: (pts[t], elo[t] + rng.normal(0, 30)),
                            reverse=True)
            winners[grp] = ranked[:2]
            for t in ranked[:2]:
                tally[t]["R16"] += 1

        # knockout
        r16 = [(winners[g1][r1], winners[g2][r2]) for (g1, r1), (g2, r2) in R16_PAIRS]
        qf = []
        for a, b in r16:
            w = a if rng.random() < adv_prob(a, b) else b
            tally[w]["QF"] += 1
            qf.append(w)
        sf_in = [(qf[0], qf[1]), (qf[2], qf[3]), (qf[4], qf[5]), (qf[6], qf[7])]
        sf = []
        for a, b in sf_in:
            w = a if rng.random() < adv_prob(a, b) else b
            tally[w]["SF"] += 1
            sf.append(w)
        f_in = [(sf[0], sf[1]), (sf[2], sf[3])]
        finalists = []
        for a, b in f_in:
            w = a if rng.random() < adv_prob(a, b) else b
            tally[w]["Final"] += 1
            finalists.append(w)
        champ = finalists[0] if rng.random() < adv_prob(*finalists) else finalists[1]
        tally[champ]["Champion"] += 1

    df = pd.DataFrame(tally).T
    df = (df / n_sims).round(4)
    df.index.name = "team"
    return df.sort_values("Champion", ascending=False).reset_index()
