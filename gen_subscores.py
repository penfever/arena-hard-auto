from pathlib import Path
from utils import write_with_subscores
import os

# get target path from command line arg
target_path = os.sys.argv[1]

dest_path = Path(target_path + "_processed")

dest_path.mkdir(parents=True, exist_ok=True)

jsons_to_process = list(Path(target_path).rglob("**/*.jsonl"))

for item in jsons_to_process:
    filename = item.stem + item.suffix
    filename_ct = item.stem + "_ct.txt"
    filename_ct_path = dest_path / filename_ct
    tgt_file_path = dest_path / filename
    if tgt_file_path.is_file():
        Path.unlink(tgt_file_path)
    write_with_subscores(item, tgt_file_path, filename_ct_path)