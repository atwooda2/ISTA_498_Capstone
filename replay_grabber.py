import csv
import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file

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
 
    # Retry up to 3x on rate limiting (429)
    for attempt in range(3):
        resp = session.get(url, headers={"Authorization": token}, timeout=60)
        if resp.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"    Rate limited — retrying in {wait}s (attempt {attempt + 1}/3)...")
            time.sleep(wait)
            continue
        break
    else:
        raise RuntimeError(f"Replay {replay_id} failed after 3 retries (rate limited).")
 
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        raise RuntimeError(f"Replay {replay_id} download failed: HTTP {resp.status_code}") from None
 
    csv_text = resp.text
 
    # Detect HTML error pages (Cloudflare, 403 splash pages, etc.)
    stripped = csv_text.lstrip()
    if stripped.startswith("<!") or stripped.lower().startswith("<html"):
        raise RuntimeError(
            f"Replay {replay_id} returned an HTML page instead of CSV "
            f"(HTTP {resp.status_code}). It may be private or deleted."
        )
 
    # Detect raw binary .replay files
    if stripped.startswith("\x00") or not csv_text.isprintable() and ";" not in csv_text[:200]:
        raise RuntimeError(
            f"Replay {replay_id} returned binary data instead of CSV. "
            "Check that the download URL is correct."
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
    token       = os.environ.get("BALLCHASING_TOKEN", "")
    group_input = "https://ballchasing.com/group/potential-smurfs-enaka01dhi"
    sleep_s     = 0.15

    if not token:
        raise RuntimeError("No API token found. Set BALLCHASING_TOKEN in your .env file.")

    group_id  = group_input.rstrip("/").split("/group/")[-1]  # → "potential-smurfs-enaka01dhi"
    group_dir = DOWNLOADS_DIR / group_id                       # → replay_csvs/potential-smurfs-enaka01dhi/

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    session  = requests.Session()
    expected_header: Optional[List[str]] = None

    # Rebuild header state from any previously downloaded files
    if MASTER_CSV_PATH.exists():
        print("Existing master CSV found — rebuilding header state from downloaded files...")
        expected_header = rebuild_master_csv(
            replay_root=DOWNLOADS_DIR,
            master_csv_path=MASTER_CSV_PATH,
            playlist="ranked-doubles",
        )

    # Fetch all replay IDs in the group
    print(f"Fetching replays from group: {group_id}")
    params   = {"group": group_id, "count": 200}
    url      = BASE_URL
    replays  = []
    first    = True
    while url:
        resp = session.get(url, headers={"Authorization": token},
                           params=params if first else {}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        replays.extend(data.get("list", []))
        url   = data.get("next")
        first = False
        if url and sleep_s:
            time.sleep(sleep_s)

    if not replays:
        print("No replays found in this group.")
        return

    print(f"\nGroup | {group_id} | {len(replays)} replays")

    for index, replay in enumerate(replays, start=1):
        replay_id       = replay["id"]
        playlist        = replay.get("playlist_name", "unknown")
        replay_csv_path = group_dir / f"{replay_id}.csv"

        # Skip if already downloaded
        if replay_csv_path.exists():
            print(f"[{index}/{len(replays)}] skipped (already exists) {replay_csv_path}")
            if sleep_s:
                time.sleep(sleep_s)
            continue

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
                rank=group_id,
                master_csv_path=MASTER_CSV_PATH,
                expected_header=expected_header,
            )
            if header_expanded and MASTER_CSV_PATH.exists() and MASTER_CSV_PATH.stat().st_size > 0:
                expected_header = rebuild_master_csv(
                    replay_root=DOWNLOADS_DIR,
                    master_csv_path=MASTER_CSV_PATH,
                    playlist=playlist,
                )
            print(f"[{index}/{len(replays)}] saved {replay_csv_path}")
        except Exception as exc:
            print(f"[{index}/{len(replays)}] failed for {replay_id}: {exc}")

        if sleep_s:
            time.sleep(sleep_s)

    print(f"\nMaster CSV written to {MASTER_CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()