"""Why the model said what it said.

The brief asks each projection to carry its reasons -- "+ Increased projected
minutes", "- Tough opponent defense". This module turns exact TreeSHAP
contributions into that.

**Themes, not raw features.** A player-game has 157 features and a SHAP value for
each. Handing a reader ``ewma_points = +1.83`` and ``last20_fga_mean = +0.41``
is not an explanation, it is a data dump; and maintaining 157 hand-written
labels means every feature-set change silently degrades the copy. So
contributions are summed into a handful of themes that mean something to a
person -- minutes, form, role, schedule, opponent, team, teammates, experience --
and the largest signed movers are phrased in plain language.

Summing SHAP values across a group is legitimate in a way that summing most
model diagnostics is not: SHAP contributions are additive by construction, so a
theme's total is exactly that theme's net effect on the prediction. The raw
per-feature values are kept alongside the prose, because the readable version is
for the product and the raw version is for whoever has to debug it.

**On honesty.** These explain the *model*, not basketball. A theme total says
this model moved this prediction this far for these inputs. It does not
establish that tough defences cause fewer points. Everything here is worded to
describe the projection rather than to assert a cause.
"""

from __future__ import annotations

from typing import Any

#: Ordered rules mapping a raw feature name to a theme. First match wins, so the
#: specific rules must precede the general ones -- ``last5_minutes_mean`` is a
#: minutes feature, not a generic form feature, and the minutes rules come first
#: for exactly that reason.
#:
#: Prefix/suffix rules rather than an exhaustive list: a new rolling window or a
#: new metric lands in the right theme without touching this file, and anything
#: genuinely unrecognised falls to ``other`` and shows up in the raw values
#: rather than being silently dropped.
_THEME_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # (theme, substrings, exact names)
    #
    # Order is load-bearing. Two traps caught here:
    #
    # - ``team_win_rate`` and ``opp_win_rate`` end in "_rate", which is also how
    #   the role features are recognised. Team and opponent therefore have to be
    #   tested before role, or a defensive rating gets reported as a change in
    #   the player's role.
    # - ``teammate_*`` is tested before ``team_*``. They do not actually collide
    #   ("teammate_" has no underscore after "team"), but the two are one typo
    #   apart and the ordering makes the intent explicit.
    (
        "minutes",
        ("_minutes_mean", "ewma_minutes", "std10_minutes", "trend10_minutes"),
        ("previous_game_minutes", "minutes_last_7d"),
    ),
    (
        "opponent",
        ("opp_", "matchup_"),
        (),
    ),
    (
        "teammates",
        ("teammate", "rotation_"),
        ("player_missed_recent_games", "player_played_last_game", "squad_games_observed"),
    ),
    (
        "team",
        ("team_",),
        ("is_home",),
    ),
    (
        "schedule",
        (),
        ("days_rest", "is_short_rest", "games_last_7d", "games_last_14d"),
    ),
    (
        "experience",
        ("games_played_",),
        ("career_window_is_current_season",),
    ),
    (
        "role",
        ("starter_", "_per40", "_rate"),
        (),
    ),
    (
        "form",
        ("season_", "last3_", "last5_", "last10_", "last20_", "ewma_", "std10_", "trend10_"),
        (),
    ),
)

#: Plain-language phrasing per theme and direction. The two examples from the
#: brief -- increased projected minutes, tough opponent defence -- are the
#: minutes-positive and opponent-negative entries.
_PHRASES: dict[str, tuple[str, str]] = {
    # theme: (positive, negative)
    "minutes": ("Increased projected minutes", "Reduced projected minutes"),
    "form": ("Strong recent form", "Weak recent form"),
    "role": ("Larger role in the rotation", "Smaller role in the rotation"),
    "schedule": ("Well rested", "Short rest or congested schedule"),
    "opponent": ("Favourable matchup", "Tough opponent defense"),
    "team": ("Strong team context", "Weak team context"),
    "teammates": ("Teammates missing, more opportunity", "Squad at full strength"),
    "experience": ("Established EuroLeague record", "Little EuroLeague history"),
    "other": ("Other factors", "Other factors"),
}

#: How many drivers to publish. Three is what fits next to a projection without
#: turning it into an essay, and beyond the third the contributions are usually
#: too small to be worth a sentence.
TOP_DRIVERS = 3

#: Contributions smaller than this are noise for every target we serve. Without
#: a floor, a player whose prediction is entirely explained by minutes still gets
#: two more bullets manufactured from rounding dust.
MIN_IMPACT = 0.05


def theme_for(feature_name: str) -> str:
    """Theme a raw feature belongs to, or ``"other"`` if nothing matches."""
    for theme, substrings, exact in _THEME_RULES:
        if feature_name in exact:
            return theme
        if any(fragment in feature_name for fragment in substrings):
            return theme
    return "other"


def phrase_for(theme: str, impact: float) -> str:
    """Readable label for a signed theme contribution."""
    positive, negative = _PHRASES.get(theme, _PHRASES["other"])
    return positive if impact >= 0 else negative


def theme_totals(
    feature_names: list[str], contributions: list[float]
) -> dict[str, float]:
    """Sum SHAP contributions per theme.

    Valid because SHAP values are additive: the sum over a group is exactly that
    group's net effect on this prediction.
    """
    if len(feature_names) != len(contributions):
        raise ValueError(
            f"feature/contribution mismatch: {len(feature_names)} names, "
            f"{len(contributions)} contributions"
        )

    totals: dict[str, float] = {}
    for name, value in zip(feature_names, contributions, strict=True):
        theme = theme_for(name)
        totals[theme] = totals.get(theme, 0.0) + float(value)
    return totals


def explain_prediction(
    feature_names: list[str],
    contributions: list[float],
    *,
    base_value: float,
    calibration_offset: float = 0.0,
    top_n: int = TOP_DRIVERS,
    top_features: int = 8,
) -> dict[str, Any]:
    """Build the stored explanation for one prediction.

    Args:
        feature_names: Pinned column order, matching ``contributions``.
        contributions: Per-feature SHAP values from ``pred_contrib``, with the
            trailing base value already removed.
        base_value: The model's expected output, SHAP's base term.
        calibration_offset: Recorded so the arithmetic reconciles. Base plus all
            contributions plus the offset is the served prediction, before the
            clip at zero -- without it the numbers appear not to add up.

    Returns:
        A JSON-serialisable dict with readable drivers, every theme total, and
        the largest raw contributions for debugging.
    """
    totals = theme_totals(feature_names, contributions)

    ranked = sorted(totals.items(), key=lambda item: -abs(item[1]))
    drivers = [
        {
            "theme": theme,
            "direction": "+" if impact >= 0 else "-",
            "label": phrase_for(theme, impact),
            "impact": round(impact, 3),
        }
        for theme, impact in ranked[:top_n]
        if abs(impact) >= MIN_IMPACT
    ]

    raw = sorted(
        zip(feature_names, contributions, strict=True),
        key=lambda item: -abs(item[1]),
    )

    return {
        "base_value": round(float(base_value), 3),
        "calibration_offset": round(float(calibration_offset), 3),
        "drivers": drivers,
        "themes": {theme: round(value, 3) for theme, value in ranked},
        # Kept deliberately: the prose is for the product, these are for whoever
        # has to work out why the prose says something surprising.
        "top_features": [[name, round(float(value), 3)] for name, value in raw[:top_features]],
    }


def render_drivers(explanation: dict[str, Any] | None) -> str:
    """One-line rendering, e.g. ``"+ Increased projected minutes, - Tough opponent defense"``."""
    if not explanation:
        return ""
    return ", ".join(
        f"{driver['direction']} {driver['label']}"
        for driver in explanation.get("drivers", [])
    )
