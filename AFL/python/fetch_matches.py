import os
import requests


URL_ROOT = "https://afltables.com/afl/teams/"
URL_STUB = "/allgames.html"
HERE_PATH = os.path.dirname(__file__)
FILE_ROOT = os.path.join(HERE_PATH, "..", "matches")
FILE_STUB = ".html" if os.path.sep == "/" else ".htm"
TEAMS = {
    "adelaide": "Adelaide",
    "brisbaneb": "Brisbane Bears",
    "brisbanel": "Brisbane Lions",
    "carlton": "Carlton",
    "collingwood": "Collingwood",
    "essendon": "Essendon",
    "fitzroy": "Fitzroy",
    "fremantle": "Fremantle",
    "geelong": "Geelong",
    "goldcoast": "Gold Coast",
    "gws": "Greater Western Sydney",
    "hawthorn": "Hawthorn",
    "melbourne": "Melbourne",
    "kangaroos": "North Melbourne",
    "padelaide": "Port Adelaide",
    "richmond": "Richmond",
    "stkilda": "St Kilda",
    "swans": "Sydney",
    "westcoast": "West Coast",
    "bullldogs": "Western Bulldogs",  # Typo in URL
}


def fetch(url, file_path):
    r = requests.get(url)
    if r.status_code == requests.status_codes.codes.ok:
        with open(file_path, "wb") as f:
            f.write(r.content)
    return r.status_code


if __name__ == "__main__":
    for uri, team in TEAMS.items():
        print(f"Fetching {uri} -> {team}...")
        url = URL_ROOT + uri + URL_STUB
        file_path = os.path.join(FILE_ROOT, team + FILE_STUB)
        ret = fetch(url, file_path)
        print(f" - Status: {ret}")
