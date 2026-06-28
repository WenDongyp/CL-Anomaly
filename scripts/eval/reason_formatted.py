import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_json", type=str, required=True, help="Raw reason output JSONL")
args = parser.parse_args()

src_file = args.data_json
dst_file = os.path.splitext(src_file)[0] + '_formatted.json'

records = []
with open(src_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

with open(dst_file, 'w', encoding='utf-8') as f_out:
    json.dump(records, f_out, ensure_ascii=False, indent=4)

print(f'Formatted file generated: {dst_file}')
print(f'Total records: {len(records)}')
