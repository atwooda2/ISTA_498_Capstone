import time
import requests
from typing import List, Dict, Optional

BASE_URL = "https://ballchasing.com/api/replays"


def fetch_replay_ids(
    token: str,
    playlist: str,
    min_rank: str,
    max_rank: Optional[str] = None,
    count_needed: int = 125,
    player_name: Optional[str] = None,
    sort_by: str = "replay-date",
    sort_dir: str = "desc",
    per_page: int = 200,
    sleep_s: float = 0.15,
) -> List[str]:
    """
    Fetch replay IDs from ballchasing with pagination until count_needed is reached.

    - If max_rank is provided (and equal to min_rank), results are rank-isolated.
    - Returns a list of replay UUIDs as strings.
    """
    headers = {"Authorization": token}

    params: Dict[str, str | int] = {
        "playlist": playlist,
        "min-rank": min_rank,
        "count": min(per_page, 200),
        "sort-by": sort_by,
        "sort-dir": sort_dir,
    }
    if max_rank:
        params["max-rank"] = max_rank
    if player_name:
        params["player-name"] = player_name

    ids: List[str] = []
    url: Optional[str] = BASE_URL
    first_request = True

    while url and len(ids) < count_needed:
        # For the first request, use params. For subsequent pages, use the full "next" URL.
        if first_request:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            first_request = False
        else:
            resp = requests.get(url, headers=headers, timeout=30)

        # Helpful error detail if something goes wrong
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from None

        data = resp.json()

        for item in data.get("list", []):
            rid = item.get("id")
            if rid:
                ids.append(rid)
            if len(ids) >= count_needed:
                break

        url = data.get("next")

        # small delay to be nice to the API / avoid spikes
        if url and sleep_s:
            time.sleep(sleep_s)

    return ids[:count_needed]


def main():
    TOKEN = "7IDKb2v3rWgFyn4bmd2KAmPzoj4nSapzdx5qeUTc"

    playlist = "ranked-doubles"

    # If you want the Firstkiller filter like your example, set this:
    player_name = None  # e.g., "Firstkiller"

    ranks = [
        "bronze-1", "bronze-2", "bronze-3",
        "silver-1", "silver-2", "silver-3",
        "gold-1", "gold-2", "gold-3",
        "platinum-1", "platinum-2", "platinum-3",
        "diamond-1", "diamond-2", "diamond-3",
        "champion-1", "champion-2", "champion-3",
        "grand-champion",
        "supersonic-legend",
    ]

    for rank in ranks:
        # Rank-isolated: min = max
        replay_ids = fetch_replay_ids(
            token=TOKEN,
            playlist=playlist,
            min_rank=rank,
            max_rank=rank,
            count_needed=125,
            player_name=player_name,
        )

        print(f"\n{playlist} | {rank} | {len(replay_ids)} IDs")
        # PowerShell-style output: just print the IDs
        for rid in replay_ids:
            print(rid)


if __name__ == "__main__":
    main()
