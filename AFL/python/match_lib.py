"""
Provides a library of tools for parsing and manipulating match records.
"""

import os

import pandas as pd
from bs4 import BeautifulSoup


def get_team_files(dir_path):
    """
    Obtains the list of paths of team files within
    the given directory.

    Input:
        - dir_path: The string path to the directory.
    Returns:
        - team_files: The list of string team file paths.
    """
    team_files = []
    for f in os.listdir(dir_path):
        team_file = os.path.join(dir_path, f)
        if os.path.isfile(team_file):
            team_files.append(team_file)
    return team_files


def parse_team_name(team_file):
    """
    Parses the team name from a team file path.

    Input:
        - team_file: The string path to the file of matches for a given team,
            in the format "<path>/<team>.<ext>".
    Returns:
        - team_name: The name of the team.
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
        - team_file: The string path to the file of matches for a given team.
    Returns:
        - matches: A dictionary of match data-frames keyed by season.
    """
    with open(team_file, 'r') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    matches = {}
    for table in soup.find_all('table'):
        th = table.find('th')
        season = int(th.text)
        if season < min_season or season > max_season:
            continue
        for i, row in enumerate(table.find_all('tr')):
            if i == 0:
                fields = _dedup([th.text.strip() for th in row.find_all('th')])
                df_matches = pd.DataFrame(columns=fields)
            else:
                data = [td.text.strip() for td in row.find_all('td')]
                if len(data) == len(fields):
                    df_matches.loc[i - 1] = data
        matches[season] = df_matches
    return matches
