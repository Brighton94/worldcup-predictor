"""Predict the 2026 World Cup (first 48-team edition)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from .data import load_intl_results
from .elo import compute_elo, ratings_as_of
from .features import _edition_table, edition_table_filled, _STRENGTH, FEATURES, strength_vector
from .model import _fit

WC2026_START = pd.Timestamp("2026-06-11")
HOSTS_2026 = {"United States", "Canada", "Mexico"}
EDITION_2026 = C.active_edition(WC2026_START)  # latest available edition (FIFA 23)

# Extra aliases so fixture-list team names match the FIFA strength table.
_TEAM_ALIASES = {"DR Congo": "Congo DR"}


def _canon(name: str) -> str:
    name = C.canon_intl(name)
    return _TEAM_ALIASES.get(name, name)


# Official 2026 knockout bracket (FIFA): fixed R32 slots; thirds resolved per Annex C.
R32 = {
    73: (("2", "A"), ("2", "B")),
    74: (("1", "E"), ("3", set("ABCDF"))),
    75: (("1", "F"), ("2", "C")),
    76: (("1", "C"), ("2", "F")),
    77: (("1", "I"), ("3", set("CDFGH"))),
    78: (("2", "E"), ("2", "I")),
    79: (("1", "A"), ("3", set("CEFHI"))),
    80: (("1", "L"), ("3", set("EHIJK"))),
    81: (("1", "D"), ("3", set("BEFIJ"))),
    82: (("1", "G"), ("3", set("AEHIJ"))),
    83: (("2", "K"), ("2", "L")),
    84: (("1", "H"), ("2", "J")),
    85: (("1", "B"), ("3", set("EFGIJ"))),
    86: (("1", "J"), ("2", "H")),
    87: (("1", "K"), ("3", set("DEIJL"))),
    88: (("2", "D"), ("2", "G")),
}
# Later rounds reference winners of earlier matches.
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL = {104: (101, 102)}

THIRD_SLOTS = [m for m, (a, b) in R32.items() if b[0] == "3"]

# Annex C assignment-column order -> the R32 match each fills (1A,1B,1D,1E,1G,1I,1K,1L).
_ANNEX_SLOT_MATCH = [79, 85, 81, 74, 82, 77, 87, 80]


@lru_cache(maxsize=1)
def _annex_c() -> dict:
    """FIFA Annex C: frozenset(8 third-placed groups) -> {match_id: group_letter}."""
    table = {}
    for ln in open(C.WC_RAW / "annex_c_raw.txt"):
        toks = [x.strip().lstrip("3") for x in ln.strip().split(";") if x.strip()]
        if len(toks) != 16:
            continue
        present, assign = frozenset(toks[:8]), toks[8:]
        table[present] = {m: g for m, g in zip(_ANNEX_SLOT_MATCH, assign)}
    return table


# Official 2026 group lettering (canonical names), labelled to match the FIFA draw, not alphabetically.
OFFICIAL_GROUPS_2026 = {
    "A": {"Mexico", "South Africa", "South Korea", "Czechia"},
    "B": {"Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"},
    "C": {"Brazil", "Morocco", "Haiti", "Scotland"},
    "D": {"United States", "Paraguay", "Australia", "Turkey"},
    "E": {"Germany", "Curaçao", "Ivory Coast", "Ecuador"},
    "F": {"Netherlands", "Japan", "Sweden", "Tunisia"},
    "G": {"Belgium", "Egypt", "Iran", "New Zealand"},
    "H": {"Spain", "Cape Verde", "Saudi Arabia", "Uruguay"},
    "I": {"France", "Senegal", "Iraq", "Norway"},
    "J": {"Argentina", "Algeria", "Austria", "Jordan"},
    "K": {"Portugal", "Congo DR", "Uzbekistan", "Colombia"},
    "L": {"England", "Croatia", "Ghana", "Panama"},
}


def load_groups_2026() -> dict[str, list[str]]:
    """Reconstruct the 12 groups from the 72 group-stage fixtures."""
    results = pd.read_csv(C.INTL_RAW / "results.csv")
    results["date"] = pd.to_datetime(results["date"])
    wc = results[(results["tournament"] == "FIFA World Cup")
                 & (results["date"] >= "2026-06-01") & (results["date"] < "2026-06-28")]
    adj: dict[str, set[str]] = {}
    for r in wc.itertuples(index=False):
        a, b = _canon(r.home_team), _canon(r.away_team)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for t in adj:
        if t in seen:
            continue
        stack, comp = [t], set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n); comp.add(n)
            stack += [m for m in adj[n] if m not in seen]
        comps.append(sorted(comp))

    # label each reconstructed group with its official FIFA letter by membership
    labelled = {}
    for comp in comps:
        cs = set(comp)
        letter = next((L for L, s in OFFICIAL_GROUPS_2026.items() if s == cs), None)
        if letter is not None:
            labelled[letter] = comp
    if len(labelled) == 12:
        return labelled
    # fallback (data drift): alphabetical, with a clear warning
    import warnings
    warnings.warn("2026 groups did not match the official draw; using alphabetical letters")
    comps.sort(key=lambda c: c[0])
    return {chr(65 + i): c for i, c in enumerate(comps)}


# Pairwise probabilities from the fitted model


def pairwise_probs(model, teams, tbl, elo, hosts) -> dict:
    rows, keys = [], []
    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                continue
            s1, s2 = strength_vector(tbl, t1), strength_vector(tbl, t2)
            hf = (1.0 if t1 in hosts and t2 not in hosts
                  else -1.0 if t2 in hosts and t1 not in hosts else 0.0)
            feat = {"d_elo": elo.get(t1, 1500.0) - elo.get(t2, 1500.0), "home_field": hf}
            for c in _STRENGTH:
                feat[f"d_{c}"] = float(s1[c] - s2[c])
            rows.append(feat); keys.append((t1, t2))
    P = model.predict_proba(pd.DataFrame(rows)[FEATURES])
    return {k: P[i] for i, k in enumerate(keys)}


def _assign_thirds(third_by_group: dict[str, str]) -> dict[int, str]:
    """Assign the 8 qualifying thirds to the 8 third-slots via FIFA's Annex C."""
    annex = _annex_c().get(frozenset(third_by_group))
    if annex is not None:
        return {m: third_by_group[g] for m, g in annex.items()}

    # fallback: backtracking perfect matching on slot eligibility
    groups, slots = list(third_by_group), THIRD_SLOTS

    def bt(i, used):
        if i == len(slots):
            return {}
        for g in groups:
            if g in used or g not in R32[slots[i]][1][1]:
                continue
            rest = bt(i + 1, used | {g})
            if rest is not None:
                return {slots[i]: third_by_group[g], **rest}
        return None

    return bt(0, set()) or {}


# Deterministic "most likely" bracket


def predict_bracket(model, groups, tbl, elo, hosts, played=None) -> tuple[dict, dict, list]:
    """Chalk bracket: expected group order, then higher win-prob advances."""
    teams = [t for g in groups.values() for t in g]
    pw = pairwise_probs(model, teams, tbl, elo, hosts)

    def adv(a, b):  # knockout: draw -> split by relative win prob
        p = pw[(a, b)]
        return p[0] + 0.5 * p[1]

    def pts_vs(t, o):  # real points if the game is played, else expected points
        if played and frozenset({t, o}) in played:
            return played[frozenset({t, o})][t]
        return 3 * pw[(t, o)][0] + pw[(t, o)][1]

    # group stage: rank by (real where played, else expected) points, tiebreak Elo
    standings, thirds, group_table = {}, [], {}
    for g, members in groups.items():
        epts = {t: sum(pts_vs(t, o) for o in members if o != t) for t in members}
        order = sorted(members, key=lambda t: (epts[t], elo.get(t, 1500)), reverse=True)
        group_table[g] = order
        standings[(g, "1")], standings[(g, "2")] = order[0], order[1]
        thirds.append((g, order[2], epts[order[2]], elo.get(order[2], 1500)))

    # eight best third-placed teams
    thirds.sort(key=lambda x: (x[2], x[3]), reverse=True)
    best = thirds[:8]
    slot_third = _assign_thirds({g: t for g, t, _, _ in best})

    bracket = {}
    for m, (sa, sb) in R32.items():
        a = standings[(sa[1], sa[0])]
        b = slot_third[m] if sb[0] == "3" else standings[(sb[1], sb[0])]
        bracket[m] = (a, b, a if adv(a, b) >= 0.5 else b)
    for rnd in (R16, QF, SF, FINAL):
        for m, (x, y) in rnd.items():
            a, b = bracket[x][2], bracket[y][2]
            bracket[m] = (a, b, a if adv(a, b) >= 0.5 else b)
    return bracket, group_table, best


# Monte-Carlo title / round probabilities


def simulate(model, groups, tbl, elo, hosts, n_sims=20000, seed=7, played=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [t for g in groups.values() for t in g]
    pw = pairwise_probs(model, teams, tbl, elo, hosts)
    rounds = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    tally = {t: dict.fromkeys(rounds, 0) for t in teams}

    def adv(a, b):
        p = pw[(a, b)]
        return p[0] + 0.5 * p[1]

    for _ in range(n_sims):
        standings, thirds = {}, []
        for g, members in groups.items():
            pts = dict.fromkeys(members, 0.0)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if played and frozenset({a, b}) in played:  # use the real result
                        res = played[frozenset({a, b})]
                        pts[a] += res[a]; pts[b] += res[b]
                        continue
                    o = rng.choice(3, p=pw[(a, b)])
                    if o == 0:
                        pts[a] += 3
                    elif o == 1:
                        pts[a] += 1; pts[b] += 1
                    else:
                        pts[b] += 3
            order = sorted(members, key=lambda t: (pts[t], elo.get(t, 1500) + rng.normal(0, 25)),
                           reverse=True)
            standings[(g, "1")], standings[(g, "2")] = order[0], order[1]
            thirds.append((g, order[2], pts[order[2]], elo.get(order[2], 1500) + rng.normal(0, 25)))
        for t in [standings[(g, "1")] for g in groups] + [standings[(g, "2")] for g in groups]:
            tally[t]["R32"] += 1
        thirds.sort(key=lambda x: (x[2], x[3]), reverse=True)
        best = thirds[:8]
        for _, t, _, _ in best:
            tally[t]["R32"] += 1
        slot_third = _assign_thirds({g: t for g, t, _, _ in best})

        winners = {}
        for m, (sa, sb) in R32.items():
            a = standings[(sa[1], sa[0])]
            b = slot_third[m] if sb[0] == "3" else standings[(sb[1], sb[0])]
            winners[m] = a if rng.random() < adv(a, b) else b
        for rnd, label in ((R16, "R16"), (QF, "QF"), (SF, "SF"), (FINAL, "Final")):
            for m, (x, y) in rnd.items():
                a, b = winners[x], winners[y]
                w = a if rng.random() < adv(a, b) else b
                winners[m] = w
                tally[w][label] += 1
        tally[winners[104]]["Champion"] += 1

    df = pd.DataFrame(tally).T
    df = (df / n_sims).round(4)
    df.index.name = "team"
    return df.sort_values("Champion", ascending=False).reset_index()


# Orchestration


def build_2026(n_sims: int = 20000, squad_table=None, results_elo=None,
               as_of_date=None, played=None):
    """Build the 2026 forecast."""
    if results_elo is None:
        results_elo = compute_elo(load_intl_results())
    groups = load_groups_2026()
    tbl = squad_table if squad_table is not None else edition_table_filled(EDITION_2026)
    elo = ratings_as_of(results_elo, as_of_date or WC2026_START)

    from .features import build_backbone
    train = build_backbone(WC2026_START, results_elo)  # training stays pre-tournament
    model = _fit(train, FEATURES, calibrate=True)

    bracket, group_table, thirds = predict_bracket(model, groups, tbl, elo, HOSTS_2026, played=played)
    sim = simulate(model, groups, tbl, elo, HOSTS_2026, n_sims=n_sims, played=played)
    return {"groups": groups, "model": model, "elo": elo, "bracket": bracket,
            "group_table": group_table, "thirds": thirds, "sim": sim,
            "n_train": len(train)}
