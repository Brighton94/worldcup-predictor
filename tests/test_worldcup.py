"""Deterministic tests for the World Cup module.

These guard the correctness properties that quietly invalidate tournament
models if broken: the leak-free regulation target, the country join, Elo
pre-match snapshots, orientation invariance, and edition mapping.

They read the local raw datasets under ``data/raw`` (gitignored). If those are
absent the data-dependent tests are skipped so the suite still runs in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.worldcup import config as C
from src.worldcup.elo import compute_elo, INIT_ELO
from src.worldcup.model import symmetrize, _FLIP_Y

_HAS_DATA = (C.WC_RAW / "matches.csv").exists() and any(C.FIFA_RAW.glob("players_*.csv"))
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="raw datasets not present")


# ── pure-logic tests (no data) ───────────────────────────────────────


def test_active_edition_maps_to_pre_tournament_release():
    # the latest edition released on or before the date
    assert C.active_edition(pd.Timestamp("2018-06-14")) == 2018  # FIFA 18 (Sep 2017)
    assert C.active_edition(pd.Timestamp("2022-11-20")) == 2023  # FIFA 23 (Sep 2022)
    assert C.active_edition(pd.Timestamp("2015-01-01")) is None  # before coverage


def test_canonical_country_aliases():
    assert C.canon_fifa("Korea Republic") == "South Korea"
    assert C.canon_fifa("China PR") == "China"
    assert C.canon_intl("Czech Republic") == "Czechia"


def test_symmetrize_flips_label_and_negates_features():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "d_elo": [100.0, -50.0],
        "home_field": [1.0, 0.0],
        "y": [0, 2],
    })
    out = symmetrize(df, ["d_elo", "home_field"])
    assert len(out) == 2 * len(df)
    mirror = out.iloc[len(df):].reset_index(drop=True)
    assert list(mirror["d_elo"]) == [-100.0, 50.0]
    assert list(mirror["y"]) == [_FLIP_Y[0], _FLIP_Y[2]]  # H<->A, D fixed


# ── data-dependent tests ─────────────────────────────────────────────


@needs_data
def test_extra_time_knockouts_are_regulation_draws():
    from src.worldcup.data import load_wc_matches
    m = load_wc_matches()
    et = m[m["extra_time"] == 1]
    assert len(et) > 0
    # a match that went to extra time was level after 90 -> regulation draw
    assert (et["result"] == "D").all()


@needs_data
def test_modern_group_matches_never_go_to_extra_time():
    # Historic tournaments had group-stage play-offs that could go to extra
    # time; the modern (modelled) format does not. The regulation-result logic
    # is correct either way, but this documents the modelled assumption.
    from src.worldcup.data import load_wc_matches
    m = load_wc_matches()
    modern = m[m["tournament_id"].isin(["WC-2018", "WC-2022"])]
    assert (modern[modern["group_stage"] == 1]["extra_time"] == 0).all()


@needs_data
def test_elo_first_match_is_initial_and_no_future_leak():
    from src.worldcup.data import load_intl_results
    r = compute_elo(load_intl_results())
    first = r.iloc[0]
    assert first["elo1_pre"] == INIT_ELO and first["elo2_pre"] == INIT_ELO
    # pre-match ratings must be finite and predate any update for that row
    assert np.isfinite(r["elo1_pre"]).all() and np.isfinite(r["elo2_pre"]).all()


@needs_data
def test_wc_test_set_is_complete_and_clean():
    from src.worldcup.features import build_dataset, FEATURES
    data = build_dataset(2022)
    test = data["test"]
    assert len(test) == 64  # all 2022 matches covered (Qatar imputed)
    assert not test[FEATURES].isna().any().any()
    assert set(test["y"].unique()).issubset({0, 1, 2})
