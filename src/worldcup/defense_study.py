"""Backtest: does a strong defence (GK + defenders) win tournaments?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .data import load_intl_results
from .features import edition_table_filled

MAJORS = ["FIFA World Cup", "UEFA Euro", "Copa América",
          "African Cup of Nations", "AFC Asian Cup", "Gold Cup"]
MIN_YEAR = 2017          # first FIFA edition with national coverage
GAP_DAYS = 60            # date gap that separates two tournament instances
DEF_N = 5               # defenders averaged with the GK (1 GK + DEF_N defenders)


def _instances(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each match with a tournament-instance id (handles cross-year events)."""
    df = df.sort_values("date").copy()
    out = []
    for tour, g in df.groupby("tournament"):
        g = g.sort_values("date")
        gap = g["date"].diff() > pd.Timedelta(days=GAP_DAYS)
        g = g.assign(inst=gap.cumsum())
        g["instance"] = tour + "|" + g["inst"].astype(str)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def tournament_outcomes() -> pd.DataFrame:
    """One row per (tournament instance, team): games, PPG, finalist, champion."""
    res = load_intl_results()
    res = res[(res["tournament"].isin(MAJORS)) & (res["date"].dt.year >= MIN_YEAR)].copy()
    res["home_team"] = res["home_team"].map(C.canon_intl)
    res["away_team"] = res["away_team"].map(C.canon_intl)
    res = _instances(res)

    rows, meta = {}, {}
    for inst, g in res.groupby("instance"):
        start = g["date"].min()
        tour = g["tournament"].iloc[0]
        final = g.loc[g["date"].idxmax()]
        # finalists / champion from the last match of the instance
        fin = {final["home_team"], final["away_team"]}
        if final["home_score"] == final["away_score"]:
            champ = None                       # decided on penalties: unknown from score
        elif final["home_score"] > final["away_score"]:
            champ = final["home_team"]
        else:
            champ = final["away_team"]
        meta[inst] = {"tournament": tour, "start": start, "year": start.year,
                      "finalists": fin, "champion": champ}
        # points per team across the instance
        for r in g.itertuples(index=False):
            for team, gf, ga in ((r.home_team, r.home_score, r.away_score),
                                 (r.away_team, r.away_score, r.home_score)):
                d = rows.setdefault((inst, team), {"games": 0, "points": 0})
                d["games"] += 1
                d["points"] += 3 if gf > ga else (1 if gf == ga else 0)

    out = []
    for (inst, team), d in rows.items():
        m = meta[inst]
        out.append({
            "instance": inst, "tournament": m["tournament"], "year": m["year"],
            "start": m["start"], "team": team,
            "games": d["games"], "ppg": d["points"] / d["games"],
            "finalist": team in m["finalists"],
            "champion": (m["champion"] is not None and team == m["champion"]),
        })
    return pd.DataFrame(out)


def attach_ratings(panel: pd.DataFrame) -> pd.DataFrame:
    """Join the active-edition defensive-line, overall and tilt ratings."""
    cache: dict[int, pd.DataFrame] = {}

    def table(edition: int) -> pd.DataFrame:
        if edition not in cache:
            cache[edition] = edition_table_filled(edition)
        return cache[edition]

    recs = []
    for r in panel.itertuples(index=False):
        ed = C.active_edition(pd.Timestamp(r.start))
        if ed is None:
            recs.append({})
            continue
        t = table(ed)
        if r.team not in t.index:
            recs.append({"edition": ed})
            continue
        row = t.loc[r.team]
        def_line = (row["gk"] + DEF_N * row["def_top"]) / (1 + DEF_N)
        recs.append({
            "edition": ed, "overall": row["ovr_top23"], "gk": row["gk"],
            "def_top": row["def_top"], "att_top": row["att_top"],
            "def_line": def_line,
            "def_tilt": def_line - row["ovr_top23"],
            "att_tilt": row["att_top"] - row["ovr_top23"],
        })
    return pd.concat([panel.reset_index(drop=True), pd.DataFrame(recs)], axis=1)


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def analyse(df: pd.DataFrame) -> dict:
    """Confound-controlled tests of defence -> tournament success."""
    d = df.dropna(subset=["overall", "def_line", "ppg"]).copy()
    # z-score within each tournament instance to strip out scale / format
    for col in ["ppg", "overall", "def_line", "def_tilt", "att_tilt", "gk"]:
        d[col + "_z"] = d.groupby("instance")[col].transform(_z)

    def corr(a, b):
        r, p = stats.pearsonr(d[a], d[b])
        return r, p

    # incremental OLS: ppg_z ~ overall_z + def_line_z  (does defence add signal?)
    X = np.column_stack([np.ones(len(d)), d["overall_z"], d["def_line_z"]])
    beta, *_ = np.linalg.lstsq(X, d["ppg_z"].to_numpy(), rcond=None)
    resid = d["ppg_z"].to_numpy() - X @ beta
    dof = len(d) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (resid @ resid) / dof)
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)

    champs = d[d["champion"]]
    return {
        "n_rows": len(d), "n_instances": d["instance"].nunique(),
        "n_champions": int(d["champion"].sum()),
        "corr_overall_ppg": corr("overall_z", "ppg_z"),
        "corr_defline_ppg": corr("def_line_z", "ppg_z"),
        "corr_gk_ppg": corr("gk_z", "ppg_z"),
        "corr_deftilt_ppg": corr("def_tilt_z", "ppg_z"),
        "corr_atttilt_ppg": corr("att_tilt_z", "ppg_z"),
        "ols_beta": {"overall_z": (beta[1], pval[1]), "def_line_z": (beta[2], pval[2])},
        "champ_mean_def_tilt": champs["def_tilt"].mean(),
        "champ_mean_att_tilt": champs["att_tilt"].mean(),
        "field_mean_def_tilt": d["def_tilt"].mean(),
        "field_mean_att_tilt": d["att_tilt"].mean(),
        "data": d,
    }


def main():
    panel = attach_ratings(tournament_outcomes())
    cov = panel["overall"].notna().mean()
    print(f"tournament-team rows: {len(panel)}  ({panel['instance'].nunique()} instances, "
          f"{panel['year'].min()}-{panel['year'].max()})")
    print(f"rating coverage: {cov:.0%} of rows joined to an edition table\n")

    a = analyse(panel)
    print(f"analysed: {a['n_rows']} rows, {a['n_instances']} tournaments, "
          f"{a['n_champions']} champions\n")
    print("Within-tournament correlations with points-per-game (r, p):")
    print(f"  overall (ovr_top23)        {a['corr_overall_ppg'][0]:+.3f}  p={a['corr_overall_ppg'][1]:.3f}")
    print(f"  defensive line (GK+5 def)  {a['corr_defline_ppg'][0]:+.3f}  p={a['corr_defline_ppg'][1]:.3f}")
    print(f"  goalkeeper alone           {a['corr_gk_ppg'][0]:+.3f}  p={a['corr_gk_ppg'][1]:.3f}")
    print(f"  defence TILT (def-overall) {a['corr_deftilt_ppg'][0]:+.3f}  p={a['corr_deftilt_ppg'][1]:.3f}")
    print(f"  attack  TILT (att-overall) {a['corr_atttilt_ppg'][0]:+.3f}  p={a['corr_atttilt_ppg'][1]:.3f}")
    print("\nIncremental OLS  ppg_z ~ overall_z + def_line_z  (beta, p):")
    print(f"  overall_z   {a['ols_beta']['overall_z'][0]:+.3f}  p={a['ols_beta']['overall_z'][1]:.3f}")
    print(f"  def_line_z  {a['ols_beta']['def_line_z'][0]:+.3f}  p={a['ols_beta']['def_line_z'][1]:.3f}")
    print("\nChampions vs field (raw rating points):")
    print(f"  defence tilt  champions {a['champ_mean_def_tilt']:+.2f}  vs field {a['field_mean_def_tilt']:+.2f}")
    print(f"  attack  tilt  champions {a['champ_mean_att_tilt']:+.2f}  vs field {a['field_mean_att_tilt']:+.2f}")

    out = panel.sort_values(["year", "tournament", "ppg"], ascending=[True, True, False])
    out.to_csv(C.OUT / "defense_study_panel.csv", index=False)
    print(f"\nwrote panel -> {C.OUT / 'defense_study_panel.csv'}")


if __name__ == "__main__":
    main()
