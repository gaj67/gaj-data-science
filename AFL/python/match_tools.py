"""
This module implements code and constants to help analyse match records.
"""
from collections import defaultdict

import pandas as pd
import numpy as np

from match_parser import (
    DATETIME_FORMAT,
    ET_WIN, ET_DRAW, ET_LOSS
)

from graph_analysis import (
    flow_prestige, adjusted_scores
)


def init_team_features(df_matches):
    """
    Initialises the team features for one or more
    seasons of matches, or a subset of such matches.
    If multiple seasons are given, then all teams
    will be included as if all matches occurred 
    within the last season.
    
    Input:
        - df_matches (DataFrame): The selected matches.
    Returns:
        - df_features (DataFrame): The initialised data structure.
    """
    if len(df_matches) == 0:
        season = 0
        teams = []
    else:
        season = max(df_matches.season)
        teams = sorted(
            set(df_matches.for_team) | set(df_matches.against_team)
        )
    num_teams = len(teams)
    return pd.DataFrame({
        'season': [season] * num_teams,
        'team': teams,
        'teams': [num_teams] * num_teams
    })


def add_wins_features(df_features, df_matches):
    """
    Updates the features with the total wins, 
    draws and losses for each team.
    
    Inputs:
        - df_features (DataFrame): The pre-initialised features.
        - df_matches (DataFrame): The selected matches.        
    """
    # Count team -> win, draw, loss
    WIN, DRAW, LOSS = range(3)
    data = defaultdict(lambda : [0, 0, 0])
    for match in df_matches.itertuples():
        # Check outcome of 'for' team vs 'against' team
        outcome = match.edge_type
        if outcome == ET_WIN:
            data[match.for_team][WIN] += 1
            data[match.against_team][LOSS] += 1
        elif outcome == ET_LOSS:
            data[match.for_team][LOSS] += 1
            data[match.against_team][WIN] += 1
        else:
            data[match.for_team][DRAW] += 1
            data[match.against_team][DRAW] += 1
    # Add features
    df_features['games'] = [sum(data[t]) for t in df_features.team]
    df_features['wins'] = [data[t][WIN] for t in df_features.team]
    df_features['draws'] = [data[t][DRAW] for t in df_features.team]
    df_features['losses'] = [data[t][LOSS] for t in df_features.team]
    adj_wins = df_features.wins + 0.5 * df_features.draws
    adj_losses = df_features.losses + 0.5 * df_features.draws
    df_features['wins_ratio'] = adj_wins / (adj_wins + adj_losses)


def add_points_features(df_features, df_matches):
    """
    Updates the features with the total number of 
    points scored by each team and against each team.
    
    Inputs:
        - df_features (DataFrame): The pre-initialised features.
        - df_matches (DataFrame): The selected matches.        
    """
    # Count team -> points_for, points_against
    TEAM_SCORED, OPPONENT_SCORED = range(2)
    data = defaultdict(lambda : [0, 0])
    for match in df_matches.itertuples():
        for_team = match.for_team
        for_score = match.for_total_score
        against_team = match.against_team
        against_score = match.against_total_score
        data[for_team][TEAM_SCORED] += for_score
        data[for_team][OPPONENT_SCORED] += against_score
        data[against_team][TEAM_SCORED] += against_score
        data[against_team][OPPONENT_SCORED] += for_score
    # Add features
    df_features['points_for'] = [data[t][TEAM_SCORED] for t in df_features.team]
    df_features['points_against'] = [data[t][OPPONENT_SCORED] for t in df_features.team]
    _wins = df_features.points_for
    _losses = df_features.points_against
    df_features['points_ratio'] = _wins / (_wins + _losses)


def add_scores_features(df_features, df_matches):
    """
    Updates the features with the total number of 
    goals and behinds scored by each team and against 
    each team.
    
    Inputs:
        - df_features (DataFrame): The pre-initialised features.
        - df_matches (DataFrame): The selected matches.        
    """
    # Count team -> points_for, points_against
    GOALS_FOR, BEHINDS_FOR, GOALS_AGAINST, BEHINDS_AGAINST = range(4)
    data = defaultdict(lambda : [0, 0, 0, 0])
    for match in df_matches.itertuples():
        for_goals = sum([getattr(match, 'for_goals' + str(i)) for i in range(1, 5)])
        for_behinds = sum([getattr(match, 'for_behinds' + str(i)) for i in range(1, 5)])
        against_goals = sum([getattr(match, 'against_goals' + str(i)) for i in range(1, 5)])
        against_behinds = sum([getattr(match, 'against_behinds' + str(i)) for i in range(1, 5)])
        for_team = match.for_team
        data[for_team][GOALS_FOR] += for_goals
        data[for_team][BEHINDS_FOR] += for_behinds
        data[for_team][GOALS_AGAINST] += against_goals
        data[for_team][BEHINDS_AGAINST] += against_behinds
        against_team = match.against_team
        data[against_team][GOALS_FOR] += against_goals
        data[against_team][BEHINDS_FOR] += against_behinds
        data[against_team][GOALS_AGAINST] += for_goals
        data[against_team][BEHINDS_AGAINST] += for_behinds
    # Add features
    df_features['goals_for'] = [data[t][GOALS_FOR] for t in df_features.team]
    df_features['behinds_for'] = [data[t][BEHINDS_FOR] for t in df_features.team]
    gf = df_features.goals_for.values
    bf = df_features.behinds_for.values
    df_features['accuracy_for'] = gf / (gf + bf)
    df_features['goals_against'] = [data[t][GOALS_AGAINST] for t in df_features.team]
    df_features['behinds_against'] = [data[t][BEHINDS_AGAINST] for t in df_features.team]
    ga = df_features.goals_against.values
    ba = df_features.behinds_against.values
    df_features['accuracy_against'] = ga / (ga + ba)
    df_features['goals_ratio'] = gf / (gf + ga)
    df_features['behinds_ratio'] = bf / (bf + ba)


def add_rank_features(df_features):
    """
    Updates the features with the rankings for each team.
    Requires both 'add_wins_features()' and
    'add_points_features()' to have been called first.
    
    Inputs:
        - df_features (DataFrame): The pre-initialised features.
    """
    points = 4 * df_features.wins + 2 * df_features.draws
    per = 100 * df_features.points_for / df_features.points_against
    df = pd.concat([points, per], axis=1, ignore_index=True)
    df.columns = ['match_points', 'percentage']
    df.sort_values(['match_points', 'percentage'], ascending=False, inplace=True)
    df['rank'] = range(1, len(df) + 1)
    df.sort_index(inplace=True)
    df_features['rank'] = ranks = df['rank'].values
    scale = -1.0 / (len(df) - 1)
    df_features['rank_score'] = scale * (ranks - 1.0) + 1.0


def compute_loss_rate_graph(teams, df_matches, for_var, against_var):
    """
    Computes the weighted adjacency matrix of the loss-rate
    graph of all matches. Each edge v_i -> v_j is directed
    from the losing team (vertex) v_i to the winning team 
    (vertex) v_j. The edge weight A_ij is the average (per-match)
    score lost by v_i and gained by v_j.
    
    Inputs:
        - teams (list): The ordered list of vertices.
        - df_matches (DataFrame): The selected matches.
        - for_var (str): The name of feature indicating
            the score won by the 'for' team.
        - against_var (str): The name of the feature
            indicating the score won by the 'against' team.
    Returns:
        - A (ndarray): The adjacency matrix.
    """
    # Compute the adjacency matrix A of the loss graph,
    # i.e. A_ij = points scored by team j against team i.
    team_map = { t: i for i, t in enumerate(teams) }
    num_teams = len(teams)
    A = np.zeros((num_teams, num_teams), dtype=float)
    # Also keep track of the matrix N of games played,
    # i.e. N_ij = #games played between teams i and j.
    N = np.zeros((num_teams, num_teams), dtype=int)
    for match in df_matches.itertuples():
        i = team_map.get(match.for_team, -1)
        j = team_map.get(match.against_team, -1)
        if i < 0 or j < 0:
            # Either or both teams not in graph - ignore match
            continue
        # Compute total loss along edge v_i -> v_j
        A[i, j] += getattr(match, against_var)
        N[i, j] += 1
        # Compute total loss along edge v_j -> v_i
        A[j, i] += getattr(match, for_var)
        N[j, i] += 1
    # Compute the average (per-match) scores, i.e. the loss rate
    ind = N > 0
    A[ind] /= N[ind]
    return A


def add_prestige_features(df_features, df_matches):
    """
    Updates the features with the prestige scores for each team,
    computed from the loss-rate graph of all matches.
    
    Inputs:
        - df_features (DataFrame): The pre-initialised features.
        - df_matches (DataFrame): The selected matches.        
    """
    A = compute_loss_rate_graph(
        df_features.team, df_matches,
        'for_is_win', 'against_is_win'
    )
    df_features['wins_prestige'] = p = flow_prestige(A)
    df_features['adj_wins_prestige'] = adjusted_scores(p)

    A = compute_loss_rate_graph(
        df_features.team, df_matches,
        'for_total_score', 'against_total_score'
    )
    df_features['points_prestige'] = p = flow_prestige(A)
    df_features['adj_points_prestige'] = adjusted_scores(p)


def compute_features(df_matches):
    """
    Computes all features for the given collection of matches.
    
    Inputs:
        - df_features (DataFrame): The pre-initialised features.
        - df_matches (DataFrame): The selected matches.        
    """
    df_features = init_team_features(df_matches)
    add_wins_features(df_features, df_matches)
    add_points_features(df_features, df_matches)
    add_scores_features(df_features, df_matches)
    add_rank_features(df_features)
    add_prestige_features(df_features, df_matches)
    return df_features
