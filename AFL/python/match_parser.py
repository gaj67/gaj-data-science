"""
This module implements a web-page scraper to obtain match records.
The web pages are assumed to have already been downloaded from:

    https://afltables.com/afl/afl_index.html

See notebooks/1_introduction.ipynb and 2_data_extraction.ipynb 
for more details about the match data files.

Currently, the defined fields are:
  - Rnd: The round of the season
  - T: The home and away flag; H (home), A (away)
  - Opponent: The opposing team
  - Scoring3: The cumulative quarterly scores of the team
  - F: The total points scored by the team
  - Scoring5: The cumulative quarterly scores of the opposing team
  - A: The total points scored by the opposing team
  - R: The match result; W (win), D (draw), L (loss)
  - M
  - W-D-L: The wins/draws/losses to date (ignored)
  - Venue: The name of the match ground
  - Crowd: The estimated size of the crowd (ignored)
  - Date: The date-time of the match.
"""

import os

import pandas as pd
from bs4 import BeautifulSoup


# Define local timestamps of matches
DATETIME_FORMAT = "%a %d-%b-%Y %I:%M %p"

# Define result/edge types from 'for' team to 'against' team
ET_WIN = "defeated"
ET_DRAW = "drew-with"
ET_LOSS = "lost-to"


def get_team_files(dir_path):
    """
    Obtains the list of paths of team files within
    the given directory.

    Input:
        - dir_path (str): The path to the directory.
    Returns:
        - team_files (list of str): The list of team file paths.
    """
    team_files = []
    for f in os.listdir(dir_path):
        if f.startswith("."):
            continue
        team_file = os.path.join(dir_path, f)
        if os.path.isfile(team_file):
            team_files.append(team_file)
    return team_files


def parse_team_name(team_file):
    """
    Parses the team name from a team file path.

    Input:
        - team_file (str): The path to the file of matches for a given team,
            in the format "<path>/<team>.<ext>".
    Returns:
        - team_name (str): The name of the team.
    """
    return os.path.splitext(os.path.basename(team_file))[0]


def _dedup(fields):
    fields_set = set(fields)
    if len(fields) == len(fields_set):
        return fields
    dup_fields = list(fields)
    for f in fields_set:
        dup_fields.remove(f)
    dup_fields = set(dup_fields)
    new_fields = []
    for i, f in enumerate(fields):
        if f in dup_fields:
            new_fields.append(f + str(i))
        else:
            new_fields.append(f)
    return new_fields


def parse_team_seasons(team_file, min_season=0, max_season=9999):
    """
    Parses the given file to find all match records per season.

    Input:
        - team_file (str): The path to the file of matches for a given team.
    Returns:
        - matches (dict: int -> DataFrame): A season -> matches mapping.
    """
    with open(team_file, "rt") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    matches = {}
    for table in soup.find_all("table"):
        th = table.find("th")
        season = int(th.text)
        if season < min_season or season > max_season:
            continue
        for i, row in enumerate(table.find_all("tr")):
            if i == 0:
                fields = _dedup([th.text.strip() for th in row.find_all("th")])
                df_matches = pd.DataFrame(columns=fields)
            else:
                data = [td.text.strip() for td in row.find_all("td")]
                if len(data) == len(fields):
                    df_matches.loc[i - 1] = data
        matches[season] = df_matches
    return matches


def split_score(score_str):
    goals, behinds = score_str.split(".")
    return int(goals), int(behinds)


def parse_quarter_scores(scores_str):
    scores = []
    prev_goals = 0
    prev_behinds = 0
    for score_str in scores_str.split(" "):
        goals, behinds = split_score(score_str)
        scores.append(goals - prev_goals)
        scores.append(behinds - prev_behinds)
        prev_goals = goals
        prev_behinds = behinds
    return scores


def old_team_name(team, season):
    """
    The team name is extracted from the file name, but
    this does not reflect changes over time.
    """
    if team == "Western Bulldogs" and season < 1997:
        return "Footscray"
    if team == "North Melbourne" and 1999 <= season <= 2007:
        return "Kangaroos"
    if team == "Sydney" and season < 1982:
        return "South Melbourne"
    if team == "Brisbane Lions" and season < 1997:
        raise ValueError("Cannot rename due to merger of teams")
    return team


def new_team_name(team):
    if team == "Footscray":
        return "Western Bulldogs"
    if team == "Kangaroos":
        return "North Melbourne"
    if team == "South Melbourne":
        return "Sydney"
    if team == "Fitzroy":
        raise ValueError("Cannot rename due to merger of teams")
    if team == "Brisbane Bears":
        raise ValueError("Cannot rename due to merger of teams")
    return team


def precedes(team1, team2):
    """
    Specifies the canonical ordering of teams.
    """
    return team1 < team2


def extract_match_data(matches, use_old_names=True):
    """
    Interprets the scraped match data.

    Input:
        - matches (dict: str -> int -> DataFrame): The raw match data, keyed by
            team -> season -> rounds.
    Output:
        - (DataFrame): The data-frame of all matches.
    """
    # Define new fields for the data-frame
    env_fields = ["season", "round", "datetime", "venue"]
    quarter_score_fields = [
        f + str(i) for i in range(1, 5) for f in ["goals", "behinds"]
    ]
    team_fields = (
        ["team", "is_home"]
        + quarter_score_fields
        + ["total_score", "match_points"]
        + ["is_win", "is_draw", "is_loss"]
    )
    for_team_fields = ["for_" + f for f in team_fields]
    against_team_fields = ["against_" + f for f in team_fields]
    result_fields = ["edge_type"]
    edge_fields = env_fields + for_team_fields + against_team_fields + result_fields

    df_edges = pd.DataFrame(columns=edge_fields)
    num_accepted = 0
    num_rejected = 0
    for team_name, team_matches in matches.items():
        for season, df_matches in team_matches.items():
            for_team = old_team_name(team_name, season) if use_old_names else team_name
            for match in df_matches.itertuples():
                against_team = (
                    match.Opponent if use_old_names else new_team_name(match.Opponent)
                )
                if not precedes(for_team, against_team):
                    # Ignore this edge; it will be extracted for the opposing team.
                    num_rejected += 1
                    continue
                num_accepted += 1
                env_info = [season, match.Rnd, match.Date, match.Venue]
                for_match_points = 4 if match.R == "W" else 2 if match.R == "D" else 0
                against_match_points = 4 - for_match_points
                for_match_scores = parse_quarter_scores(match.Scoring3)
                against_match_scores = parse_quarter_scores(match.Scoring5)
                for_info = (
                    [for_team, match.T == "H"]
                    + for_match_scores
                    + [int(match.F), for_match_points]
                    + [match.R == "W", match.R == "D", match.R == "L"]
                )
                against_info = (
                    [against_team, match.T == "A"]
                    + against_match_scores
                    + [int(match.A), against_match_points]
                    + [match.R == "L", match.R == "D", match.R == "W"]
                )
                result = (
                    ET_WIN if match.R == "W" else ET_DRAW if match.R == "D" else ET_LOSS
                )
                edge_info = env_info + for_info + against_info + [result]
                df_edges.loc[len(df_edges), :] = edge_info
    assert num_accepted == num_rejected
    return df_edges
