import argparse
import os
import re
from collections import defaultdict

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="")
args = parser.parse_args()

# ---------- Paths ----------
txt_base_path = "/data1/dongwen_data/CL-Anomaly/router"
txt_file_path = os.path.join(txt_base_path, args.task + "_router_count.txt")
output_file_path = txt_file_path.replace("_router_count.txt", "_router_index.txt")

# ---------- Aggregate by layer ----------
layer_cnt = defaultdict(lambda: defaultdict(int))   # {layer_idx: {expert_id: total}}

with open(txt_file_path, "r") as f:
    for line in f:
        m = re.match(r"Layer(\d+):\{([^}]+)\}", line.strip())
        if not m:
            continue
        layer_idx, body = int(m.group(1)), m.group(2)
        for pair in body.split(","):
            eid, cnt = pair.split(":")
            eid, cnt = int(eid), int(cnt)
            if eid != -1:                      # skip tie markers
                layer_cnt[layer_idx][eid] += cnt

# ---------- Top-1 per layer ----------
top1_per_layer = {}
for lyr in sorted(layer_cnt.keys()):
    sorted_eids = sorted(layer_cnt[lyr].items(), key=lambda x: x[1], reverse=True)
    top1 = sorted_eids[0][0]          # take only the top-1
    top1_per_layer[lyr] = top1

# ---------- Write to file ----------
with open(output_file_path, "w") as f:
    for lyr in sorted(top1_per_layer.keys()):
        t1 = top1_per_layer[lyr]
        f.write(f"Layer{lyr}:{{{t1}}}\n")

print("Top-1 experts per layer ->", top1_per_layer)

print("\nLayer-wise expert occurrence counts:")
for lyr in sorted(layer_cnt.keys()):
    # print by descending count, format: Layerxx: eid1:cnt1 eid2:cnt2 ...
    items = sorted(layer_cnt[lyr].items(), key=lambda x: x[1], reverse=True)

    line = " ".join(f"{eid}:{cnt}" for eid, cnt in items)
    print(f"Layer{lyr}: {line}")
