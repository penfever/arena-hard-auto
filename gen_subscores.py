import argparse
from pathlib import Path
from utils import write_with_subscores

# Set up argument parser
parser = argparse.ArgumentParser(description="Process JSONL files from a target directory.")
parser.add_argument("--target_path", type=Path, help="Path to the target directory containing JSONL files.")
args = parser.parse_args()

# Define destination path
dest_path = args.target_path.with_name(args.target_path.name + "_processed")
dest_path.mkdir(parents=True, exist_ok=True)

# Find all JSONL files in the target directory and its subdirectories
jsons_to_process = list(args.target_path.rglob("*.jsonl"))

# Process each JSONL file
for item in jsons_to_process:
    tgt_file_path = dest_path / item.name
    if tgt_file_path.exists():
        tgt_file_path.unlink()
    write_with_subscores(item, tgt_file_path)
