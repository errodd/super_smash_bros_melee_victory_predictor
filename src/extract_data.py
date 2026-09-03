"""
Script to decompress, unpack, and flatten Super Smash Bros Melee match frame data.

Data Format:
- Compressed msgpack archives (.msgpack.zst) in data/archive/train/ and data/archive/test/
- Labels in data/archive/training_labels.csv

This script extracts nested player states and game metadata from the frame records,
reconstructs match identifiers (match_id) from contiguous frame sequences, merges
the corresponding match outcome label, and converts the data into a tabular format
(Pandas DataFrame, Parquet, or CSV).
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
import zstandard as zstd
import msgpack
import pandas as pd


def load_labels_subset(labels_csv_path: str, target_file_ids: Set[str]) -> Dict[str, int]:
    """
    Streams the training labels CSV file line-by-line and loads outcomes only
    for the specified file IDs to avoid loading the entire 880+ MB file into memory.

    Args:
        labels_csv_path: Path to `training_labels.csv`.
        target_file_ids: Set of file prefixes to extract (e.g., {'0000', '0001'}).

    Returns:
        Dictionary mapping row_id (e.g. '0000_123') -> outcome (1 or 2).
    """
    labels = {}
    if not os.path.exists(labels_csv_path):
        print(f"[Warning] Labels file not found at: {labels_csv_path}")
        return labels

    print(f"Loading labels for file prefixes: {target_file_ids}...")
    max_prefix = max(target_file_ids) if target_file_ids else ""
    with open(labels_csv_path, "r", encoding="utf-8") as f:
        _header = f.readline()
        for line in f:
            line_clean = line.strip()
            if not line_clean:
                continue
            parts = line_clean.split(",")
            if len(parts) == 2:
                row_id, outcome = parts
                prefix = row_id.split("_")[0]
                if prefix in target_file_ids:
                    labels[row_id] = int(outcome)
                elif max_prefix and prefix > max_prefix:
                    # Labels are sorted by prefix; stop once we exceed the highest target prefix
                    break
    print(f"Loaded {len(labels):,} label records.")
    return labels


def assign_match_ids(frames_list: list, file_id: str) -> Dict[int, str]:
    """
    Reconstructs distinct matches from a shuffled list of frames.
    
    Frames are grouped by stage, player costumes, and characters, then sorted
    chronologically by frame index. Any sequence where consecutive frames increment
    by 1 belongs to the exact same match.
    
    Args:
        frames_list: Unpacked list of frame dictionaries from a single archive.
        file_id: The partition identifier (e.g. "0000").
        
    Returns:
        Dictionary mapping original frame index in the archive -> match_id.
    """
    groups = defaultdict(list)
    for idx, fr in enumerate(frames_list):
        p1 = fr.get("players", {}).get(1, {})
        p2 = fr.get("players", {}).get(2, {})
        key = (
            fr.get("stage"),
            p1.get("costume"),
            p2.get("costume"),
            p1.get("character"),
            p2.get("character")
        )
        groups[key].append((fr.get("frame"), idx))

    match_id_by_idx = {}
    match_counter = 0
    for _key, items in groups.items():
        items.sort(key=lambda x: x[0])
        current_match = [items[0]]
        for i in range(1, len(items)):
            if items[i][0] == items[i - 1][0] + 1:
                current_match.append(items[i])
            else:
                match_counter += 1
                mid = f"{file_id}_m{match_counter:02d}"
                for _, orig_idx in current_match:
                    match_id_by_idx[orig_idx] = mid
                current_match = [items[i]]
        if current_match:
            match_counter += 1
            mid = f"{file_id}_m{match_counter:02d}"
            for _, orig_idx in current_match:
                match_id_by_idx[orig_idx] = mid

    return match_id_by_idx


def flatten_frame(frame_data: dict, row_id: str, match_id: str, outcome: Optional[int] = None) -> dict:
    """
    Flattens a single game frame dictionary into a flat record.
    Preserves None for absent fields rather than masking them with artificial defaults.

    Args:
        frame_data: Raw unpacked dictionary representing a frame.
        row_id: Identifier corresponding to the file and frame index (e.g., '0000_0').
        match_id: Reconstructed identifier of the match (e.g. '0000_m01').
        outcome: Match winner outcome (1, 2, or None if test set).

    Returns:
        Dictionary with flattened column attributes.
    """
    players = frame_data.get("players", {})
    p1 = players.get(1, {})
    p2 = players.get(2, {})

    p1_pos = p1.get("position", {}) if isinstance(p1.get("position"), dict) else {}
    p2_pos = p2.get("position", {}) if isinstance(p2.get("position"), dict) else {}

    record = {
        # Identifiers and global match context
        "row_id": row_id,
        "match_id": match_id,
        "frame": frame_data.get("frame"),
        "stage": frame_data.get("stage"),
        
        # Player 1 Attributes (no masking defaults)
        "p1_character": p1.get("character"),
        "p1_costume": p1.get("costume"),
        "p1_stock": p1.get("stock"),
        "p1_percent": p1.get("percent"),
        "p1_shield_strength": p1.get("shield_strength"),
        "p1_is_powershield": p1.get("is_powershield"),
        "p1_facing": p1.get("facing"),
        "p1_action": p1.get("action"),
        "p1_action_frame": p1.get("action_frame"),
        "p1_invulnerable": p1.get("invulnerable"),
        "p1_invulnerability_left": p1.get("invulnerability_left"),
        "p1_hitlag_left": p1.get("hitlag_left"),
        "p1_hitstun_frames_left": p1.get("hitstun_frames_left"),
        "p1_jumps_left": p1.get("jumps_left"),
        "p1_on_ground": p1.get("on_ground"),
        "p1_off_stage": p1.get("off_stage"),
        "p1_x": p1_pos.get("x"),
        "p1_y": p1_pos.get("y"),
        "p1_speed_air_x_self": p1.get("speed_air_x_self"),
        "p1_speed_y_self": p1.get("speed_y_self"),
        "p1_speed_ground_x_self": p1.get("speed_ground_x_self"),

        # Player 2 Attributes (no masking defaults)
        "p2_character": p2.get("character"),
        "p2_costume": p2.get("costume"),
        "p2_stock": p2.get("stock"),
        "p2_percent": p2.get("percent"),
        "p2_shield_strength": p2.get("shield_strength"),
        "p2_is_powershield": p2.get("is_powershield"),
        "p2_facing": p2.get("facing"),
        "p2_action": p2.get("action"),
        "p2_action_frame": p2.get("action_frame"),
        "p2_invulnerable": p2.get("invulnerable"),
        "p2_invulnerability_left": p2.get("invulnerability_left"),
        "p2_hitlag_left": p2.get("hitlag_left"),
        "p2_hitstun_frames_left": p2.get("hitstun_frames_left"),
        "p2_jumps_left": p2.get("jumps_left"),
        "p2_on_ground": p2.get("on_ground"),
        "p2_off_stage": p2.get("off_stage"),
        "p2_x": p2_pos.get("x"),
        "p2_y": p2_pos.get("y"),
        "p2_speed_air_x_self": p2.get("speed_air_x_self"),
        "p2_speed_y_self": p2.get("speed_y_self"),
        "p2_speed_ground_x_self": p2.get("speed_ground_x_self"),

        # Target outcome (1: Player 1 won, 2: Player 2 won)
        "outcome": outcome
    }
    return record


def extract_and_flatten_file(
    file_path: str,
    labels_dict: Optional[Dict[str, int]] = None,
    max_frames: Optional[int] = None
) -> pd.DataFrame:
    """
    Decompresses and unpacks a .msgpack.zst file, reconstructs matches, and flattens all frames.

    Args:
        file_path: Path to the .msgpack.zst file.
        labels_dict: Optional lookup dictionary of row_id -> outcome.
        max_frames: Optional maximum number of frames to extract.

    Returns:
        Pandas DataFrame containing flattened observations with match_id.
    """
    file_path_obj = Path(file_path)
    file_id = file_path_obj.name.split(".")[0]  # e.g., "0000"

    print(f"Processing '{file_path_obj.name}'...")
    dctx = zstd.ZstdDecompressor()
    records = []

    with open(file_path, "rb") as compressed_file:
        with dctx.stream_reader(compressed_file) as reader:
            unpacker = msgpack.Unpacker(reader, raw=False, strict_map_key=False)
            for raw_content in unpacker:
                if isinstance(raw_content, list):
                    match_map = assign_match_ids(raw_content, file_id)
                    for idx, frame_dict in enumerate(raw_content):
                        if max_frames is not None and len(records) >= max_frames:
                            break
                        row_id = f"{file_id}_{idx}"
                        match_id = match_map.get(idx, f"{file_id}_m00")
                        outcome = labels_dict.get(row_id) if labels_dict else None
                        records.append(flatten_frame(frame_dict, row_id, match_id, outcome))
                if max_frames is not None and len(records) >= max_frames:
                    break

    df = pd.DataFrame(records)
    print(f"Extracted {len(df):,} rows and {df['match_id'].nunique()} matches from '{file_path_obj.name}'.")
    return df


def process_dataset(
    file_paths: List[str],
    labels_csv_path: Optional[str] = None,
    output_path: Optional[str] = None,
    max_frames_per_file: Optional[int] = None
) -> pd.DataFrame:
    """
    Processes multiple .msgpack.zst files and returns or saves a concatenated DataFrame.

    Args:
        file_paths: List of file paths to process.
        labels_csv_path: Path to training_labels.csv (if training data).
        output_path: Optional output file path (.parquet or .csv) to save the flattened data.
        max_frames_per_file: Optional limit of frames per file.

    Returns:
        Concatenated Pandas DataFrame.
    """
    file_ids = {Path(p).name.split(".")[0] for p in file_paths}
    labels_dict = load_labels_subset(labels_csv_path, file_ids) if labels_csv_path else None

    dfs = []
    for fp in file_paths:
        df_file = extract_and_flatten_file(fp, labels_dict=labels_dict, max_frames=max_frames_per_file)
        dfs.append(df_file)

    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"Total flattened dataset shape: {combined_df.shape} with {combined_df['match_id'].nunique()} unique matches.")

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if output_path.endswith(".parquet"):
            combined_df.to_parquet(output_path, index=False)
            print(f"Saved dataset to Parquet: {output_path}")
        else:
            combined_df.to_csv(output_path, index=False)
            print(f"Saved dataset to CSV: {output_path}")

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompress, unpack, and flatten SSBM match data.")
    parser.add_argument(
        "--input_files",
        nargs="+",
        default=["data/archive/train/0000.msgpack.zst"],
        help="List of .msgpack.zst files to process."
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="data/archive/training_labels.csv",
        help="Path to training_labels.csv (optional for test data)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output (e.g. data/flattened_sample.parquet or .csv)."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to extract per file (e.g. 5000 for quick testing)."
    )

    args = parser.parse_args()
    process_dataset(
        file_paths=args.input_files,
        labels_csv_path=args.labels_csv,
        output_path=args.output_path,
        max_frames_per_file=args.max_frames
    )
