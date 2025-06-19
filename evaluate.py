import os
import torch
import gc
import re
from tqdm import tqdm
from glob import glob

from data import data_loader
from utils import get_dset_path, relative_to_abs, displacement_error, final_displacement_error
from models import TrajectoryGenerator
from constants import *

device = torch.device('cpu')

def evaluate_helper(error):
    error = torch.stack(error, dim=1)
    error = torch.sum(error, dim=0)
    return torch.min(error)

def evaluate(loader, generator):
    ade_outer, fde_outer = [], []
    total_traj = 0
    generator.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(NUM_SAMPLES):
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, vgg_list)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))

            ade_sum = evaluate_helper(ade)
            fde_sum = evaluate_helper(fde)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

    ade = sum(ade_outer) / (total_traj * PRED_LEN)
    fde = sum(fde_outer) / total_traj
    return ade.item(), fde.item()

def load_and_evaluate_checkpoint(model_path, dataset_name):
    path = get_dset_path(dataset_name, 'test')
    _, loader = data_loader(path)

    checkpoint = torch.load(model_path, map_location=device)
    generator = TrajectoryGenerator()
    generator.load_state_dict(checkpoint['g'])
    generator.to(device)
    generator.eval()

    ade, fde = evaluate(loader, generator)
    print(f"[{dataset_name}] {os.path.basename(model_path)} ➤ ADE: {ade:.2f}, FDE: {fde:.2f}")

def extract_iter_key(path):
    match = re.search(r'_iter(\d+)_model\.pt$', path)
    return int(match.group(1)) if match else float('inf')  # final model goes last

def main():
    for dataset in DATASET_LIST:
        print(f"\n=== Evaluating {dataset} ===")
        model_files = glob(f'./models/{dataset}_iter*_model.pt')
        final_model = f'./models/{dataset}_model.pt'
        if os.path.exists(final_model):
            model_files.append(final_model)

        model_files = sorted(model_files, key=extract_iter_key)

        if not model_files:
            print(f"No checkpoint found for {dataset}")
            continue

        for ckpt_path in model_files:
            load_and_evaluate_checkpoint(ckpt_path, dataset)
            gc.collect()

if __name__ == '__main__':
    main()
