import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from constants import *  # Ensure MAP_PATH, DATA_ROOT, etc. are defined here

# Utility to build the shared scene grid (as before)
def build_scene_grid(is_north):
    origin = [-474, 52322]
    green = [(-455 - origin[0], 52322 - origin[1]), (-455 - origin[0], 52468 - origin[1])]
    transp = [(21,27), (32,66), (82,107), (126,133)]

    cols = int(green[0][0]) + 1
    rows = int(green[1][1]) + 1
    grid = torch.zeros(rows, cols, dtype=torch.float32)
    grid[:,0] = 1
    grid[:,-1] = 2
    for s, e in transp:
        grid[s:e+1, -1] = 3

    corners = [[4.5, 0], [7.1, 0], [4.5, 5.8], [7.1, 5.8],
               [4.5, 38.6], [7.1, 38.6], [4.5, 43.7], [7.1, 43.7],
               [4.5, 80], [7.1, 80], [4.5, 86], [7.1, 86],
               [4.5, 117.8], [7.1, 117.8], [4.5, 124.1], [7.1, 124.1]]
    for i in range(0, len(corners), 4):
        xs = [corners[i+j][0] for j in range(4)]
        ys = [corners[i+j][1] for j in range(4)]
        c0, c1 = int(min(xs)), int(max(xs))
        r0, r1 = int(min(ys)), int(max(ys))
        grid[r0:r1+1, c0:c1+1] = 4
    return grid

# Settings
DATA_ROOT = '/home/xiangmin/PycharmProjects/Xinjiekou/Data/Xinjiekou/North'
SOPHIE_IMG_ROOT = './img'
SOPHIE_DATA_ROOT = './datasets'
STEP = 4                # every 0.4s from 0.1s
MIN_STEPS = 20          # 20 time steps after subsampling
HEADING = 1             # 1 = southward
TIME_PERIODS = ['AM', 'NOON', 'PM']

# Create directories
os.makedirs(SOPHIE_IMG_ROOT, exist_ok=True)
os.makedirs(SOPHIE_DATA_ROOT, exist_ok=True)

# Generate shared map once (as before)
grid = build_scene_grid(is_north=True)
img8 = (grid / grid.max() * 255).byte().cpu().numpy()
rgb = np.stack([img8]*3, axis=2)
shared_map_path = os.path.join(SOPHIE_IMG_ROOT, "North_shared_map.png")
Image.fromarray(rgb).save(shared_map_path)
print(f"✔ Shared map saved to {shared_map_path}")

# Process each time period
for tp in TIME_PERIODS:
    print(f"\n→ Processing: North_{tp}")
    data_dir = os.path.join(SOPHIE_DATA_ROOT, f"North_{tp}")
    os.makedirs(data_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(data_dir, split), exist_ok=True)

    origin = [-474, 52322]
    h_thresh = v_thresh = 5 if tp == 'AM' else 10
    all_files = glob.glob(os.path.join(DATA_ROOT, tp, "*.txt"))

    scenes = []
    for file in all_files:
        df = pd.read_csv(file, sep='\t', header=None)
        if df.shape[0] < STEP * MIN_STEPS:
            continue
        x = df.iloc[::STEP, 2].values - origin[0]
        y = df.iloc[::STEP, 4].values - origin[1]
        if len(x) < MIN_STEPS:
            continue
        if int(y[-1] - y[0] > 0) != HEADING:
            continue
        if (x.max() - x.min() > h_thresh and y.max() - y.min() < v_thresh):
            continue
        traj = np.stack([x, y], axis=1)
        scenes.append(traj)


    print(f"  • Total valid scenes: {len(scenes)}")
    train, temp = train_test_split(scenes, test_size=0.2, random_state=1)
    val, test = train_test_split(temp, test_size=0.5, random_state=1)
    splits = {'train': train, 'val': val, 'test': test}

    for split_name, trajs in splits.items():
        split_dir = os.path.join(data_dir, split_name)
        for idx, traj in enumerate(trajs):
            scene_id = f"North_{tp}_{split_name}_{idx}"
            traj_file = os.path.join(split_dir, f"{scene_id}.txt")
            with open(traj_file, 'w') as f:
                for t, (xx, yy) in enumerate(traj):
                    f.write(f"{t},0,{xx:.4f},{yy:.4f}\n")
        print(f"  • {split_name} set: {len(trajs)} scenes")

print("\n✅ Done! Now update `DATASET_NAME` in `sophie/constants.py` to one of:")
print("   ", ', '.join([f'North_{tp}' for tp in TIME_PERIODS]))
print("   and set `MAP_PATH` to:", shared_map_path)