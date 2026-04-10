"""
fix_master_csv.py
─────────────────
Fixes master_replays.csv which has a mixed-delimiter format:
  - First 3 columns (replay_id, playlist, rank) are comma-separated
  - Remaining columns (color, player name, ...) are semicolon-separated

Outputs a clean, fully comma-delimited CSV with all columns separated.

Usage:
    python fix_master_csv.py
    python fix_master_csv.py --input my_master.csv --output my_master_fixed.csv
"""

import csv
import argparse
from pathlib import Path


def fix_master_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open(encoding="utf-8") as infile:
        lines = infile.readlines()

    if not lines:
        raise RuntimeError("Input file is empty.")

    # ── Parse header ─────────────────────────────────────────────────────────
    # Format: "replay_id,playlist,rank,color;team name;player name;..."
    # Split on comma first (max 3 splits), then split the 4th chunk on semicolons
    header_raw  = lines[0].rstrip("\n")
    h_meta      = header_raw.split(",", 3)         # [replay_id, playlist, rank, "color;..."]
    h_data_cols = h_meta[3].split(";")             # [color, team name, player name, ...]
    full_header  = h_meta[:3] + h_data_cols        # all columns as a flat list

    print(f"Columns detected: {len(full_header)}")
    print(f"First 6: {full_header[:6]}")
    print(f"Last 3:  {full_header[-3:]}")
    print(f"Rows to process: {len(lines) - 1}\n")

    # ── Write fixed file ──────────────────────────────────────────────────────
    skipped = 0
    written = 0

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(full_header)

        for i, line in enumerate(lines[1:], start=2):
            line = line.rstrip("\n")
            if not line.strip():
                skipped += 1
                continue

            r_parts = line.split(",", 3)

            if len(r_parts) < 4:
                print(f"  Warning: row {i} has fewer than 4 comma-separated parts — skipping.")
                skipped += 1
                continue

            r_meta      = r_parts[:3]               # replay_id, playlist, rank
            r_data_cols = r_parts[3].split(";")     # color, team name, ...

            row = r_meta + r_data_cols

            # Pad or trim to match header length so CSV stays rectangular
            if len(row) < len(full_header):
                row += [""] * (len(full_header) - len(row))
            elif len(row) > len(full_header):
                row = row[:len(full_header)]

            writer.writerow(row)
            written += 1

    print(f"Done — {written} rows written, {skipped} skipped.")
    print(f"Fixed CSV saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Fix mixed-delimiter master_replays.csv")
    parser.add_argument("--input",  default="/Users/sw33t0404/Desktop/ISTA 498/ISTA_498_Capstone/replay_csvs/master_replays.csv",       help="Input CSV path")
    parser.add_argument("--output", default="/Users/sw33t0404/Desktop/ISTA 498/ISTA_498_Capstone/replay_csvs/master_replays_fixed.csv", help="Output CSV path")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fix_master_csv(input_path, output_path)


if __name__ == "__main__":
    main()