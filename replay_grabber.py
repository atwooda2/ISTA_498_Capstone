import requests
from typing import List, Dict, Optional

BASE_URL = "https://ballchasing.com/api/replays"


def grab_replays(
    token: str,
    playlist: str,
    player_name: str,
    rank: str,
    total_needed: int = 125,
) -> List[Dict]:
    """
    Fetch exactly `total_needed` replays for a specific rank.

    Args:
        token: Ballchasing API token
        playlist: ranked-duels | ranked-doubles | ranked-standard
        player_name: Player to filter
        rank: Exact rank slug (e.g., diamond-3)
        total_needed: Number of replays to collect

    Returns:
        List of replay metadata dictionaries
    """

    headers = {"Authorization": token}
    params = {
        "playlist": playlist,
        "player-name": player_name,
        "min-rank": rank,
        "max-rank": rank,  # critical for exact rank isolation
        "count": min(total_needed, 200),
        "sort-by": "replay-date",
        "sort-dir": "desc",
    }

    replays = []
    url = BASE_URL

    try:
        while url and len(replays) < total_needed:
            response = requests.get(url, headers=headers, params=params if url == BASE_URL else None)
            response.raise_for_status()
            data = response.json()

            replays.extend(data.get("list", []))

            # pagination
            url = data.get("next")

            # after first request, params must not be re-sent if using full "next" URL
            params = None

        return replays[:total_needed]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching replays: {e}")
        return []


if __name__ == "__main__":
    TOKEN = "PASTE_YOUR_API_TOKEN"

    playlists = ["ranked-duels", "ranked-doubles", "ranked-standard"]

    ranks = [
        "bronze",
        "silver",
        "gold",
        "platinum",
        "diamond",
        "champion",
        "grand-champion",
        "supersonic-legend"
    ]

    for playlist in playlists:
        for rank in ranks:
            print(f"\nFetching {playlist} - {rank}")
            results = grab_replays(TOKEN, playlist, "Firstkiller", rank)
            print(f"Collected {len(results)} replays")
