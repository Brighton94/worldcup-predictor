"""World Cup prediction module.

A self-contained pipeline for international-tournament prediction that lives
alongside the club-league engine but does not depend on it (it only reuses
``src.analysis.evaluation`` for scoring rules).

Pipeline
--------
1. ``data``           -- load World Cup, FIFA-rating, and international-match
                         sources with a single canonical country naming.
2. ``elo``            -- forward-in-time World-Football Elo from all
                         international matches (leak-free pre-match snapshot).
3. ``team_strength``  -- aggregate FIFA player ratings into per-nation squad
                         strength for each tournament cycle.
4. ``cross_tournament`` -- continental-tournament form features + the
                         standalone "does Euro/Copa form predict the WC?"
                         correlation study.
5. ``features``       -- assemble the match-level training matrix.
6. ``model``          -- temporal-split training, calibration, baselines.
7. ``simulate``       -- Monte-Carlo bracket simulation for round-advancement
                         and title probabilities.

Class encoding follows the repo convention: H = 0, D = 1, A = 2, where
"home"/"away" are the two orientations stored in the source match row. World
Cup matches are at neutral venues, so a ``host`` flag carries the real
home-advantage signal rather than the nominal home designation.
"""

RANDOM_STATE = 7
