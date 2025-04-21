from pathlib import Path
from utils import write_with_subscores
import os

target_path = "/home/benfeuer/arena-hard-auto/data/arena-hard-v0.1/model_judgment/gpt-4o-mini-2024-07-18_judge/gpt-4-0314_base_v4_pairwise_noselfref"

dest_path = Path(target_path + "_processed")

dest_path.mkdir(parents=True, exist_ok=True)

jsons_to_process = list(Path(target_path).rglob("**/*.jsonl"))

for item in jsons_to_process:
    filename = item.stem + item.suffix
    tgt_file_path = dest_path / filename
    if tgt_file_path.is_file():
        Path.unlink(tgt_file_path)
    write_with_subscores(item, tgt_file_path)