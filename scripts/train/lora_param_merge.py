#!/usr/bin/env python3
import os
import torch
import shutil
import argparse
import re
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--task1", type=str, required=True, help="full dir name of task1")
parser.add_argument("--task2", type=str, required=True, help="full dir name of task2")
args = parser.parse_args()

# ---------- Paths ----------
base_path      = "/data1/dongwen_data/CL-Anomaly/checkpoints"
pretrain_path  = "/data1/dongwen_data/CL-Anomaly/pretrained_checkpoints"

task1_bin = os.path.join(base_path,      args.task1, "adapter_model.bin")
task2_bin = os.path.join(pretrain_path,  args.task2, "adapter_model.bin")

router_path = "/data1/dongwen_data/CL-Anomaly/router"
router1_txt = os.path.join(router_path, args.task1 + "_router_index.txt")
router2_txt = os.path.join(router_path, args.task2 + "_router_index.txt")

# ---------- Hyperparameters ----------
EXPERT_TOTAL = 16
PRIVATE_NUM  = 8
shared_idxs  = list(range(PRIVATE_NUM, EXPERT_TOTAL))   # 8,9,10,11

# ---------- 1. Parse router_index.txt (Top-1) ----------
def parse_router_top1(txt):
    res = {}
    with open(txt) as f:
        for line in f:
            m = re.match(r"Layer(\d+):\{(\d+)\}", line.strip())
            if m:
                lyr, top1 = int(m.group(1)), int(m.group(2))
                res[lyr] = [top1]          # wrap as list to avoid int vs list
    return res

r1_top1 = parse_router_top1(router1_txt)   # task1 top1 per layer
r2_top1 = parse_router_top1(router2_txt)   # task2 top1 per layer

print(r1_top1)
print("****************")
print(r2_top1)
# ---------- 2. Load weights ----------
theta_old = torch.load(task1_bin)
theta_new = torch.load(task2_bin)
merged = theta_new.copy()          # default to all task2

# ---------- 3. Merge shared experts only (8-11) ----------
for name in theta_old:
    m = re.search(r"\.lora_(A|B)\.lora(A|B)\.(\d+)\.", name)
    if not m:
        continue
    expert_id = int(m.group(3))
    if expert_id in shared_idxs:           # only modify 8-11
        layer_id = int(name.split(".layers.")[1].split(".")[0])
        # determine weights
        expert_id -= PRIVATE_NUM
        if expert_id in r1_top1.get(layer_id, []) and expert_id not in r2_top1.get(layer_id, []):
            a1, a2 = 0.75, 0.25  # only in task1
        elif expert_id in r2_top1.get(layer_id, []) and expert_id not in r1_top1.get(layer_id, []):
            a1, a2 = 0.25, 0.75  # only in task2
        else:
            a1, a2 = 0.5, 0.5  # both in / neither in
        merged[name] = a1 * theta_old[name] + a2 * theta_new[name]
        print(f"[Merged] {name}  layer={layer_id}  expert={expert_id + PRIVATE_NUM}  alpha=({a1},{a2})")

# ---------- 4. Save ----------
dst_dir = os.path.join(base_path, args.task2)
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.makedirs(dst_dir, exist_ok=True)
shutil.copytree(os.path.join(pretrain_path, args.task2), dst_dir, dirs_exist_ok=True)
torch.save(merged, os.path.join(dst_dir, "adapter_model.bin"))

print(">>> Shared expert momentum fusion done, private experts directly use task2, Top-1 parsed, written to", dst_dir)
