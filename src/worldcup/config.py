"""Paths, constants, and country-name normalisation for the World Cup module.

Everything is repo-relative via ``pathlib`` (no hardcoded absolute paths).
Country names are normalised to a single canonical form (the jfjelstul World
Cup ``team_name`` spelling) so the three data sources join cleanly.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────

# repo root = three parents up from this file (src/worldcup/config.py)
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
WC_RAW = RAW / "worldcup"
FIFA_RAW = RAW / "fifa"
INTL_RAW = RAW / "international"
OUT = ROOT / "_processed_outputs" / "worldcup"
OUT.mkdir(parents=True, exist_ok=True)

# ── Modelling cycle -> FIFA editions ─────────────────────────────────
# Each World Cup is described by the FIFA editions released *before* kickoff.
# Only editions present in data/raw/fifa are usable; 14-16 can be dropped in
# later and added here without touching anything else. The edition closest to
# the tournament is the most predictive of squad strength, so editions are
# recency-weighted in team_strength (weight grows with edition year).
CYCLE_EDITIONS: dict[int, list[int]] = {
    2018: [2017, 2018],              # 2018 WC (Jun 2018); FIFA18 = Sep 2017
    2022: [2019, 2020, 2021, 2022],  # 2022 WC (Nov 2022); FIFA22 = Oct 2021
}

# Tournaments modelled (men's only by default). Women's WCs exist in the data
# but FIFA-rating coverage is for the men's game, so they are excluded.
MODELLED_TOURNAMENTS = [f"WC-{y}" for y in CYCLE_EDITIONS]

# FIFA edition release dates (approximate, late September of the prior year).
# Used to attach the edition "active" at any match date so every international
# match in the covered window can be enriched with squad-strength features.
import pandas as _pd  # noqa: E402

FIFA_RELEASE = {
    2017: _pd.Timestamp("2016-09-27"),
    2018: _pd.Timestamp("2017-09-29"),
    2019: _pd.Timestamp("2018-09-28"),
    2020: _pd.Timestamp("2019-09-27"),
    2021: _pd.Timestamp("2020-10-09"),
    2022: _pd.Timestamp("2021-10-01"),
    2023: _pd.Timestamp("2022-09-30"),  # last FIFA-branded edition
    # EA FC editions are Kaggle-gated; release dates are registered so that, once
    # the matching players_24..26.csv (sofifa schema) are dropped into
    # data/raw/fifa, they are used automatically. Until then they are ignored
    # because active_edition only considers editions actually present.
    2024: _pd.Timestamp("2023-09-29"),  # EA FC 24
    2025: _pd.Timestamp("2024-09-27"),  # EA FC 25
    2026: _pd.Timestamp("2025-09-26"),  # EA FC 26
}


def active_edition(date: "_pd.Timestamp") -> int | None:
    """Latest *available* edition released on or before ``date`` (else None).

    Considers only editions that are actually present with full national
    coverage, so a registered-but-not-yet-downloaded edition (e.g. EA FC 26) is
    skipped until its data file exists.
    """
    from .data import available_editions  # lazy: avoids data<->config import cycle

    have = set(available_editions())
    eligible = [e for e, rel in FIFA_RELEASE.items() if rel <= date and e in have]
    return max(eligible) if eligible else None

# ── Continental tournaments for cross-tournament features ────────────
# Labels as they appear in the martj42 international-results ``tournament``
# column. Used both for the correlation study and for form features.
CONTINENTAL_TOURNAMENTS = [
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "FIFA Confederations Cup",
]

# ── Country canonicalisation ─────────────────────────────────────────
# Map every source's spelling onto the canonical (World Cup) spelling.
# Only entries that actually differ need listing; unmatched names are
# reported at build time so this map can be extended deliberately.

# FIFA ``nationality`` -> canonical
FIFA_COUNTRY_ALIASES: dict[str, str] = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Republic of Ireland": "Republic of Ireland",
    "Czech Republic": "Czechia",
    "Cape Verde Islands": "Cape Verde",
    "DR Congo": "Congo DR",
    "Holland": "Netherlands",  # EA FC 25/26 spelling
}

# martj42 results -> canonical
INTL_COUNTRY_ALIASES: dict[str, str] = {
    "United States": "United States",
    "South Korea": "South Korea",
    "Ivory Coast": "Ivory Coast",
    "Czech Republic": "Czechia",
    "China PR": "China",
    "Cape Verde": "Cape Verde",
    "Democratic Republic of the Congo": "Congo DR",
    "DR Congo": "Congo DR",  # martj42 spelling; align with squad-strength / groups
    "Republic of Ireland": "Republic of Ireland",
}


def canon_fifa(name: str) -> str:
    """Canonicalise a FIFA nationality string."""
    return FIFA_COUNTRY_ALIASES.get(name, name)


def canon_intl(name: str) -> str:
    """Canonicalise an international-results team string."""
    return INTL_COUNTRY_ALIASES.get(name, name)
