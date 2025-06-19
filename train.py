import gc
import os
import math
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from data import data_loader
from utils import get_dset_path, relative_to_abs, l2_loss, displacement_error, final_displacement_error
from models import TrajectoryGenerator, TrajectoryDiscriminator

from constants import *

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes():
    return torch.LongTensor, torch.FloatTensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for DATASET_NAME in DATASET_LIST:
        train_path = get_dset_path(DATASET_NAME, 'train')
        val_path = get_dset_path(DATASET_NAME, 'val')
        long_dtype, float_dtype = get_dtypes()

        print(f"\nInitializing train dataset: {DATASET_NAME}")
        train_dset, train_loader = data_loader(train_path)
        print(f"Initializing val dataset: {DATASET_NAME}")
        _, val_loader = data_loader(val_path)

        iterations_per_epoch = len(train_dset) / D_STEPS
        print(f'There are {iterations_per_epoch:.1f} iterations per epoch')

        generator = TrajectoryGenerator().to(device)
        generator.apply(init_weights)
        generator.train()
        print('Here is the generator:')
        print(generator)

        discriminator = TrajectoryDiscriminator().to(device)
        discriminator.apply(init_weights)
        discriminator.train()
        print('Here is the discriminator:')
        print(discriminator)

        optimizer_g = optim.Adam(
            generator.parameters(),
            lr=G_LR,
            betas=(0.5, 0.9),
            weight_decay=1e-5
        )
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=D_LR * 4,
            betas=(0.5, 0.9),
            weight_decay=1e-5
        )

        def lr_lambda(current_step):
            if current_step < NUM_ITERATIONS / 3:
                return 1.0
            elif current_step < 2 * NUM_ITERATIONS / 3:
                return 0.5
            else:
                return 0.25

        scheduler_g = LambdaLR(optimizer_g, lr_lambda=lr_lambda)
        scheduler_d = LambdaLR(optimizer_d, lr_lambda=lr_lambda)

        t, epoch = 0, 0
        min_ade = None

        while t < NUM_ITERATIONS:
            gc.collect()
            d_steps_left = D_STEPS
            g_steps_left = G_STEPS
            epoch += 1
            print(f'Starting epoch {epoch}')
            for batch in train_loader:
                batch = [tensor.to(device) for tensor in batch]

                if d_steps_left > 0:
                    losses_d = discriminator_step(batch, generator, discriminator, optimizer_d, device)
                    d_steps_left -= 1
                elif g_steps_left > 0:
                    losses_g = generator_step(batch, generator, discriminator, optimizer_g, device)
                    g_steps_left -= 1

                if d_steps_left > 0 or g_steps_left > 0:
                    continue

                if t % PRINT_EVERY == 0:
                    print(f't = {t+1} / {NUM_ITERATIONS}')
                    for k, v in sorted(losses_d.items()):
                        print(f'  [D] {k}: {v:.3f}')
                    for k, v in sorted(losses_g.items()):
                        print(f'  [G] {k}: {v:.3f}')

                    print('Checking stats on val ...')
                    metrics_val = check_accuracy(val_loader, generator, discriminator, device)

                    print('Checking stats on train ...')
                    metrics_train = check_accuracy(train_loader, generator, discriminator, device, limit=True)

                    for k, v in sorted(metrics_val.items()):
                        print(f'  [val] {k}: {v:.3f}')
                    for k, v in sorted(metrics_train.items()):
                        print(f'  [train] {k}: {v:.3f}')

                    checkpoint = {
                        't': t,
                        'g': generator.state_dict(),
                        'd': discriminator.state_dict(),
                        'g_optim': optimizer_g.state_dict(),
                        'd_optim': optimizer_d.state_dict()
                    }
                    print(f"Saving checkpoint to ./models/{DATASET_NAME}_iter{t}_model.pt")
                    torch.save(checkpoint, f"./models/{DATASET_NAME}_iter{t}_model.pt")
                    print("Done.")

                scheduler_g.step()
                scheduler_d.step()
                t += 1
                d_steps_left = D_STEPS
                g_steps_left = G_STEPS
                if t >= NUM_ITERATIONS:
                    break

        final_checkpoint = {
            't': t,
            'g': generator.state_dict(),
            'd': discriminator.state_dict(),
            'g_optim': optimizer_g.state_dict(),
            'd_optim': optimizer_d.state_dict()
        }
        print(f"\nSaving final model to ./models/{DATASET_NAME}_model.pt")
        torch.save(final_checkpoint, f"./models/{DATASET_NAME}_model.pt")
        print("Final model saved.")

def discriminator_step(batch, generator, discriminator, optimizer_d, device):
    obs, pred_gt, obs_rel, pred_gt_rel, vgg = batch
    with torch.no_grad():
        fake_rel = generator(obs, obs_rel, vgg)
        fake_abs = relative_to_abs(fake_rel, obs[-1, :, 0, :])

    real_abs = torch.cat([obs[:, :, 0, :], pred_gt], dim=0)
    real_rel = torch.cat([obs_rel[:, :, 0, :], pred_gt_rel], dim=0)
    fake_seq_abs = torch.cat([obs[:, :, 0, :], fake_abs], dim=0)
    fake_seq_rel = torch.cat([obs_rel[:, :, 0, :], fake_rel], dim=0)

    scores_real = discriminator(real_abs, real_rel)
    scores_fake = discriminator(fake_seq_abs, fake_seq_rel)

    loss_real = torch.relu(1.0 - scores_real).mean()
    loss_fake = torch.relu(1.0 + scores_fake).mean()
    d_loss = 0.5 * (loss_real + loss_fake)

    losses = {'D_data_loss': d_loss.item(), 'D_total_loss': d_loss.item()}

    optimizer_d.zero_grad()
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    optimizer_d.step()

    return losses

def generator_step(batch, generator, discriminator, optimizer_g, device):
    obs, pred_gt, obs_rel, pred_gt_rel, vgg = batch
    g_l2_terms = [l2_loss(generator(obs, obs_rel, vgg), pred_gt_rel, mode='raw') for _ in range(BEST_K)]
    g_l2_stack = torch.stack(g_l2_terms, dim=1)
    best_l2 = torch.min(g_l2_stack.sum(dim=0)) / (obs.size(1) * PRED_LEN)

    with torch.no_grad():
        fake_rel = generator(obs, obs_rel, vgg)
        fake_abs = relative_to_abs(fake_rel, obs[-1, :, 0, :])
    fake_seq_abs = torch.cat([obs[:, :, 0, :], fake_abs], dim=0)
    fake_seq_rel = torch.cat([obs_rel[:, :, 0, :], fake_rel], dim=0)
    scores_fake = discriminator(fake_seq_abs, fake_seq_rel)

    g_adv = -scores_fake.mean()
    g_loss = best_l2 + g_adv

    losses = {
        'G_l2_loss_rel': best_l2.item(),
        'G_discriminator_loss': g_adv.item(),
        'G_total_loss': g_loss.item()
    }

    optimizer_g.zero_grad()
    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    optimizer_g.step()

    return losses

def check_accuracy(loader, generator, discriminator, device, limit=False):
    d_losses, g_l2_abs, g_l2_rel, disp_err, f_disp_err = [], [], [], [], []
    total, mask_sum = 0, 0

    generator.eval()
    with torch.no_grad():
        for batch in loader:
            obs, pred_gt, obs_rel, pred_gt_rel, vgg = [t.to(device) for t in batch]

            fake_rel = generator(obs, obs_rel, vgg)
            fake_abs = relative_to_abs(fake_rel, obs[-1, :, 0, :])

            g_l2_abs.append(l2_loss(fake_abs, pred_gt, mode='sum').item())
            g_l2_rel.append(l2_loss(fake_rel, pred_gt_rel, mode='sum').item())
            disp_err.append(displacement_error(fake_abs, pred_gt).item())
            f_disp_err.append(final_displacement_error(fake_abs[-1], pred_gt[-1]).item())

            real_abs = torch.cat([obs[:, :, 0, :], pred_gt], dim=0)
            real_rel = torch.cat([obs_rel[:, :, 0, :], pred_gt_rel], dim=0)
            fake_seq_abs = torch.cat([obs[:, :, 0, :], fake_abs], dim=0)
            fake_seq_rel = torch.cat([obs_rel[:, :, 0, :], fake_rel], dim=0)
            s_real = discriminator(real_abs, real_rel)
            s_fake = discriminator(fake_seq_abs, fake_seq_rel)
            d_losses.append(0.5 * (torch.relu(1.0 - s_real).mean() + torch.relu(1.0 + s_fake).mean()).item())

            mask_sum += pred_gt.size(1) * PRED_LEN
            total += pred_gt.size(1)
            if limit and total >= NUM_SAMPLES_CHECK:
                break

    metrics = {
        'd_loss': sum(d_losses) / len(d_losses),
        'g_l2_loss_abs': sum(g_l2_abs) / mask_sum,
        'g_l2_loss_rel': sum(g_l2_rel) / mask_sum,
        'ade': sum(disp_err) / (total * PRED_LEN),
        'fde': sum(f_disp_err) / total
    }
    generator.train()
    return metrics

if __name__ == '__main__':
    main()