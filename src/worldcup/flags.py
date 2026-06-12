"""Country flag emojis for national teams."""

from __future__ import annotations

try:
    import flag as _flag
    _HAVE = True
except ImportError:  # pragma: no cover
    _HAVE = False

# canonical team name -> ISO 3166-1 alpha-2 (or GB subdivision)
_CODE = {
    "Algeria": "DZ", "Argentina": "AR", "Australia": "AU", "Austria": "AT",
    "Belgium": "BE", "Bosnia and Herzegovina": "BA", "Brazil": "BR",
    "Canada": "CA", "Cape Verde": "CV", "Colombia": "CO", "Croatia": "HR",
    "Curaçao": "CW", "Czechia": "CZ", "Congo DR": "CD", "Ecuador": "EC",
    "Egypt": "EG", "England": "GB-ENG", "France": "FR", "Germany": "DE",
    "Ghana": "GH", "Haiti": "HT", "Iran": "IR", "Iraq": "IQ",
    "Ivory Coast": "CI", "Japan": "JP", "Jordan": "JO", "Mexico": "MX",
    "Morocco": "MA", "Netherlands": "NL", "New Zealand": "NZ", "Norway": "NO",
    "Panama": "PA", "Paraguay": "PY", "Portugal": "PT", "Qatar": "QA",
    "Saudi Arabia": "SA", "Scotland": "GB-SCT", "Senegal": "SN",
    "South Africa": "ZA", "South Korea": "KR", "Spain": "ES", "Sweden": "SE",
    "Switzerland": "CH", "Tunisia": "TN", "Turkey": "TR",
    "United States": "US", "Uruguay": "UY", "Uzbekistan": "UZ", "Wales": "GB-WLS",
}


def flag_emoji(team: str) -> str:
    """Flag emoji for a team's canonical name (empty string if unknown)."""
    code = _CODE.get(team)
    if not code or not _HAVE:
        return ""
    try:
        return _flag.flag(code)
    except Exception:
        return ""


def label(team: str) -> str:
    """``"<flag> Team"`` when a flag is available, else just the team name."""
    e = flag_emoji(team)
    return f"{e} {team}" if e else team
