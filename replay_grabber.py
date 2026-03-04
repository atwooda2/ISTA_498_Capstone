import csv
import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import os
load_dotenv()

import requests

BASE_URL = "https://ballchasing.com/api/replays"
DOWNLOADS_DIR = Path("replay_csvs")
MASTER_CSV_PATH = DOWNLOADS_DIR / "master_replays.csv"


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
        if first_request:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            first_request = False
        else:
            resp = requests.get(url, headers=headers, timeout=30)

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

        if url and sleep_s:
            time.sleep(sleep_s)

    return ids[:count_needed]


def download_replay_csv(
    session: requests.Session,
    token: str,
    replay_id: str,
    output_path: Path,
) -> str:
    """
    Download one replay file from ballchasing and save it locally.

    Returns the CSV text so it can be appended into the master file immediately.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://ballchasing.com/dl/stats/players/{replay_id}/{replay_id}-players.csv"

    # Retry up to 3 times on rate limiting (429)
    for attempt in range(3):
        resp = session.get(url, headers={"Authorization": token}, timeout=60)
        if resp.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"    Rate limited — retrying in {wait}s (attempt {attempt + 1}/3)...")
            time.sleep(wait)
            continue
        break
    else:
        raise RuntimeError(f"Replay {replay_id} failed after 3 retries due to rate limiting.")

    try:
        resp.raise_for_status()
    except requests.HTTPError:
        raise RuntimeError(f"Replay {replay_id} download failed: HTTP {resp.status_code}") from None

    csv_text = resp.text

    # Detect HTML error pages (Cloudflare walls, 403 pages, etc.)
    stripped = csv_text.lstrip()
    if stripped.startswith("<!") or stripped.lower().startswith("<html"):
        raise RuntimeError(
            f"Replay {replay_id} returned an HTML page instead of CSV "
            f"(HTTP {resp.status_code}). It may be private, deleted, or rate limited."
        )

    if not csv_text.strip():
        raise RuntimeError(f"Replay {replay_id} returned empty content.")

    output_path.write_text(csv_text, encoding="utf-8", newline="")
    return csv_text


def append_replay_to_master(
    csv_text: str,
    replay_id: str,
    playlist: str,
    rank: str,
    master_csv_path: Path,
    expected_header: Optional[List[str]],
) -> Tuple[List[str], bool]:
    """
    Append one replay CSV into the master CSV while preserving the original columns.

    Extra metadata columns are added so each row can be traced back to the source replay.
    """
    # The csv module expects newline="" so embedded newlines inside fields are handled correctly.
    reader = csv.reader(io.StringIO(csv_text, newline=""))
    rows = list(reader)
    if not rows:
        raise RuntimeError(f"Replay {replay_id} returned an empty CSV.")

    header = rows[0]
    if not header:
        raise RuntimeError(f"Replay {replay_id} returned a CSV with no header.")

    data_rows = [row for row in rows[1:] if any(cell.strip() for cell in row)]

    header_expanded = False
    if expected_header is None:
        expected_header = header.copy()
        header_expanded = True
    else:
        for column in header:
            if column not in expected_header:
                expected_header.append(column)
                header_expanded = True

    row_maps: List[Dict[str, str]] = []
    for row in data_rows:
        padded_row = row + [""] * (len(header) - len(row))
        row_maps.append(dict(zip(header, padded_row[: len(header)])))

    master_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not master_csv_path.exists() or master_csv_path.stat().st_size == 0

    with master_csv_path.open("a", newline="", encoding="utf-8") as master_file:
        writer = csv.writer(master_file)
        if write_header:
            writer.writerow(["replay_id", "playlist", "rank", *expected_header])

        for row_map in row_maps:
            writer.writerow(
                [replay_id, playlist, rank, *[row_map.get(column, "") for column in expected_header]]
            )

    return expected_header, header_expanded


def rebuild_master_csv(replay_root: Path, master_csv_path: Path, playlist: str) -> List[str]:
    """
    Rebuild the master CSV from the downloaded replay files using the union of all columns.
    """
    csv_files = sorted(path for path in replay_root.rglob("*.csv") if path != master_csv_path)
    if not csv_files:
        raise RuntimeError("No replay CSV files were found to rebuild the master CSV.")

    master_rows: List[Tuple[str, str, str, Dict[str, str]]] = []
    combined_header: List[str] = []

    for csv_file in csv_files:
        rank = csv_file.parent.name
        replay_id = csv_file.stem

        with csv_file.open("r", encoding="utf-8", newline="") as replay_file:
            reader = csv.reader(replay_file)
            rows = list(reader)

        if not rows or not rows[0]:
            continue

        header = rows[0]
        for column in header:
            if column not in combined_header:
                combined_header.append(column)

        for row in rows[1:]:
            if not any(cell.strip() for cell in row):
                continue
            padded_row = row + [""] * (len(header) - len(row))
            master_rows.append((replay_id, playlist, rank, dict(zip(header, padded_row[: len(header)]))))

    with master_csv_path.open("w", newline="", encoding="utf-8") as master_file:
        writer = csv.writer(master_file)
        writer.writerow(["replay_id", "playlist", "rank", *combined_header])
        for replay_id, playlist, rank, row_map in master_rows:
            writer.writerow([replay_id, playlist, rank, *[row_map.get(column, "") for column in combined_header]])

    return combined_header
def main():
    token = os.getenv("BALLCHASING_TOKEN")
    if not token:
        raise RuntimeError("Please set the BALLCHASING_TOKEN environment variable with your API token.")
    playlist = "ranked-doubles"
    player_name = None
    count_per_rank = 12
    sleep_s = 0.15

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

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if MASTER_CSV_PATH.exists():
        MASTER_CSV_PATH.unlink()

    session = requests.Session()
    expected_header: Optional[List[str]] = None

    for rank in ranks:
        replay_ids = fetch_replay_ids(
            token=token,
            playlist=playlist,
            min_rank=rank,
            max_rank=rank,
            count_needed=count_per_rank,
            player_name=player_name,
        )

        print(f"\n{playlist} | {rank} | {len(replay_ids)} IDs")

        rank_dir = DOWNLOADS_DIR / rank

        for index, replay_id in enumerate(replay_ids, start=1):
            replay_csv_path = rank_dir / f"{replay_id}.csv"
            try:
                csv_text = download_replay_csv(
                    session=session,
                    token=token,
                    replay_id=replay_id,
                    output_path=replay_csv_path,
                )
                expected_header, header_expanded = append_replay_to_master(
                    csv_text=csv_text,
                    replay_id=replay_id,
                    playlist=playlist,
                    rank=rank,
                    master_csv_path=MASTER_CSV_PATH,
                    expected_header=expected_header,
                )
                if header_expanded and MASTER_CSV_PATH.exists() and MASTER_CSV_PATH.stat().st_size > 0:
                    expected_header = rebuild_master_csv(
                        replay_root=DOWNLOADS_DIR,
                        master_csv_path=MASTER_CSV_PATH,
                        playlist=playlist,
                    )
                print(f"[{rank}] {index}/{len(replay_ids)} saved {replay_csv_path}")
            except Exception as exc:
                print(f"[{rank}] {index}/{len(replay_ids)} failed for {replay_id}: {exc}")

            if sleep_s:
                time.sleep(sleep_s)

    print(f"\nMaster CSV written to {MASTER_CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()