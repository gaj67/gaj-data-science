"""
This module implements code and constants to help extract
information from match records.
"""

from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

from match_parser import DATETIME_FORMAT


TIMESTAMP = "timestamp"
HASH = "hash"


###############################################
# Tools to obtain historical matches

def add_timestamp(df_matches):
    """
    Adds a comparable timestamp to all matches,
    and sorts the matches in chronological order.

    Input:
        - df_matches (DataFrame): The historical matches.
    """
    date_fn = lambda s: datetime.strptime(s, DATETIME_FORMAT)
    df_matches[TIMESTAMP] = df_matches.datetime.apply(date_fn)
    df_matches.sort_values(TIMESTAMP, inplace=True)


def add_hash(df_matches):
    """
    Adds a comparable hash to all matches for each team.

    Input:
        - df_matches (DataFrame): The historical matches or match features.
    """
    if "team" in df_matches.columns:
        # Match features
        hash_fn = lambda m: hash(m.datetime + m.team)
        df_matches[HASH] = df_matches.apply(hash_fn, axis=1)
    else:
        # Matches
        for prefix in ["for_", "against_"]:
            hash_fn = lambda m: hash(m.datetime + m[prefix + "team"])
            df_matches[prefix + HASH] = df_matches.apply(hash_fn, axis=1)


def get_prior_matches(df_matches, timestamp):
    """
    Obtains all matches prior to the given date-time.

    Inputs:
        - df_matches (DataFrame): The historical matches.
        - timestamp (datetime): The current timestamp.
    Returns:
        - (DataFrame): The selected matches.
    """
    return df_matches[df_matches[TIMESTAMP] < timestamp]


def get_season_matches(df_matches, season):
    """
    Obtains a single season of matches.

    Inputs:
        - df_matches (DataFrame): The historical matches.
        - season (int): The year of the season.
    Returns:
        - (DataFrame): The season matches.
    """
    return df_matches[df_matches.season == season]


def get_team_matches(df_matches, team, season=None, timestamp=None):
    """
    Obtains matches played by a given team.

    If a season is specified, then only matches within that season
    are found.

    If a timestamp is specified, then only matches prior to that
    timestamp are found. Assumes that add_timestamp() has already
    been called.

    Inputs:
        - df_matches (DataFrame): The historical matches.
        - team (str): The name of the team.
        - season (int): The optional year of the season.
        - timestamp (datetime): The optional timestamp.
    Returns:
        - (DataFrame): The team matches.
    """
    ind = (df_matches.for_team == team) | (df_matches.against_team == team)
    if season is not None:
        ind &= df_matches.season == season
    if timestamp is not None:
        ind &= df_matches[TIMESTAMP] < timestamp
    return df_matches[ind]


def get_previous_matches(df_matches, match, is_for, within_season=True):
    """
    Obtains all matches prior to the specified match that
    were played by a given team within a given season.

    Assumes that add_timestamp() has already been called.

    Inputs:
        - df_matches (DataFrame): The historical matches.
        - match (Pandas): The current match.
        - is_for (bool): Indicates whether to extract matches
            for the 'for' team (True) or the 'against' team (False).
        - within_season (bool): Indicates whether to extract only matches
            within the cuurrent season (True) or all previous matches (False).
    Returns:
        - (DataFrame): The previous team matches within the season.
    """
    team = get_match_team(match, is_for)
    ts = getattr(match, TIMESTAMP)
    season = match.season if within_season else None
    return get_team_matches(df_matches, team, season, ts)


def get_previous_match(df_matches, match, is_for):
    """
    Obtains the match immediately prior to the specified match
    that was played by a given team within a given season.

    Assumes that add_timestamp() has already been called.

    Inputs:
        - df_matches (DataFrame): The historical matches.
        - match (Pandas): The current match.
        - is_for (bool): Indicates whether to extract the match
            for the 'for' team (True) or the 'against' team (False).
    Returns:
        - (Pandas): The previous team match within the season,
            or a value of None if there is no previous match.
    """
    df = get_previous_matches(df_matches, match, is_for)
    if len(df) == 0:
        return None
    return next(df.iloc[-1:, :].itertuples())


def get_minor_rounds(df_matches):
    """
    Extracts the minor-round matches.

    Inputs:
        - df_matches (DataFrame): The historical matches.
    Returns:
        - df_minor (DataFrame): The minor-round matches.
    """
    ind = df_matches['round'].str.startswith('R')
    return df_matches[ind]


###############################################
# Tools to examine a single match

def get_match_team(match, is_for):
    """
    Obtains the specified team for the match.

    Inputs:
        - match (Pandas): The current match.
        - is_for (bool): Indicates whether to extract the
            'for' team (True) or the 'against' team (False).
    Returns:
        - team (str): The team name."""
    return match.for_team if is_for else match.against_team


def get_match_result(match, is_for: bool) -> int:
    """
    Encodes the match result for the specified team as
    +1 for a win, -1 for a loss, and 0 for a draw.

    Inputs:
        - match (Pandas): The current match.
        - is_for (bool): Indicates whether to extract the result
            for the 'for' team (True) or the 'against' team (False).
    Returns:
        - res (int): The encoded result.
    """
    res = 0 if match.for_is_draw else 1 if match.for_is_win else -1
    if not is_for:
        res = -res
    return res


def get_match_proportion(match, is_for: bool) -> float:
    """
    Encodes the match result for the specified team as
    1 for a win, 0 for a loss, and 0.5 for a draw.

    Inputs:
        - match (Pandas): The current match.
        - is_for (bool): Indicates whether to extract the result
            for the 'for' team (True) or the 'against' team (False).
    Returns:
        - res (float): The encoded result.
    """
    res = 0.5 if match.for_is_draw else 1.0 if match.for_is_win else 0.0
    if not is_for:
        res = 1 - res
    return res


def get_match_home(match, is_for):
    """
    Determines whether or not the specified team played the
    given match at their home ground. Encodes the resulting
    Boolean value as an integer, i.e. True -> 1, False -> 0.

    Inputs:
        - match (Pandas): The current match.
        - is_for (bool): Indicates whether to extract the result
            for the 'for' team (True) or the 'against' team (False).
    Returns:
        - home (int): The binary indicator.
    """
    if is_for:
        return int(match.for_is_home)
    else:
        return int(match.against_is_home)


def get_match_score(match, is_for):
    """
    Obtains the total number of points scored in the given match
    by the specified team.

    Inputs:
        - match (Pandas): The current match.
        - is_for (bool): Indicates whether to extract the score
            for the 'for' team (True) or the 'against' team (False).
    Returns:
        - score (int): The team score.
    """

    if is_for:
        return match.for_total_score
    else:
        return match.against_total_score


###############################################
# Tools to extract match features

def get_match_features(df_features, match, is_for):
    """
    Obtains the features, pre-computed  prior to the
    specified match, that were played by a given team.

    Assumes that add_hash() has already been called on both
    the features and the matches.

    Inputs:
      - df_features (DataFrame): The pre-computed match features.
      - match (Pandas): The current match.
      - is_for (bool): Indicates whether to extract features
          for the 'for' team (True) or the 'against' team (False).
    Returns:
      - (Pandas): The team features, or a value of None if there
          are no features avaialble.
    """
    prefix = "for_" if is_for else "against_"
    df = df_features[df_features[HASH] == getattr(match, prefix + HASH)]
    if len(df) == 0:
        return None
    return next(df.itertuples())


def iter_match_features(df_matches, df_features):
    """
    Iterates over each match, giving the pre-computed
    features, prior to the match, for both teams.

    Assumes that add_hash() has already been called on both
    the features and the matches.

    Inputs:
      - df_matches (DataFrame): The historical matches.
      - df_features (DataFrame): The pre-computed match features.
    Returns:
        - (iter of tuple): An iterator over tuples of results,
            giving the match, the 'for' team features (or None),
            and the 'against' team features (or None).
    """
    for match in df_matches.itertuples():
        for_features = get_match_features(df_features, match, is_for=True)
        against_features = get_match_features(df_features, match, is_for=False)
        yield match, for_features, against_features


def get_seasonal_features(df_seasonal, match, is_for, is_prev = True):
    """
    Obtains the end-of-season features for the given team.

    Inputs:
      - df_seasonal (DataFrame): The pre-computed seasonal features.
      - match (Pandas): The current match.
      - is_for (bool): Indicates whether to extract features
          for the 'for' team (True) or the 'against' team (False).
      - is_prev (bool): Indicates whether to obtain the features
          from the previous season (True) or the current season (False).
          By default, the previous season is assumed.
    Returns:
      - (Pandas): The team features, or a value of None if there
          are no features avaialble.
    """
    team = get_match_team(match, is_for)
    res = df_seasonal[
        (df_seasonal.team == team)
        & (df_seasonal.season == match.season - int(is_prev))
    ]
    if len(res) == 0:
        return None
    return next(res.itertuples())


def iter_seasonal_features(df_matches, df_seasonal, is_prev = True):
    """
    Iterates over each match, giving the pre-computed
    features, prior to the match, for both teams.

    Inputs:
      - df_matches (DataFrame): The historical matches.
      - df_seasonal (DataFrame): The pre-computed seasonal features.
      - is_prev (bool): Indicates whether to obtain the features
          from the previous season (True) or the current season (False).
          By default, the previous season is assumed.
    Returns:
        - (iter of tuple): An iterator over tuples of results,
            giving the match, the 'for' team features (or None),
            and the 'against' team features (or None).
    """
    for match in df_matches.itertuples():
        for_features = get_seasonal_features(
            df_seasonal, match, True, is_prev
        )
        against_features = get_seasonal_features(
          df_seasonal, match, False, is_prev
        )
        yield match, for_features, against_features
