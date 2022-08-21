import os
import logging
import pandas as pd


def get_ckpt_list(ckpt_dir):
    ckpts = []

    for ckpt_name in os.listdir(ckpt_dir):
        if ckpt_name.endswith('.index'):
            ckpt_nr = int(ckpt_name.split('.')[1])
            ckpts.append((ckpt_nr, ckpt_name.replace('.index', '')))

    return sorted(ckpts)


def get_ckpt(ckpt_dir, nr):
    if os.path.exists(ckpt_dir):
        ckpt_list = get_ckpt_list(ckpt_dir)
        for ckpt in ckpt_list:
            if ckpt[0] == nr:
                return ckpt

    logging.warning(f'Could not find checkpoint with number {nr}')
    return -1, None


def get_latest_ckpt(ckpt_dir):
    if os.path.exists(ckpt_dir):
        ckpt_list = get_ckpt_list(ckpt_dir)

        if ckpt_list:
            return ckpt_list[-1]
    return -1, None


def get_best_ckpt(ckpt_dir, log_file_path, metric='val_loss', mode='min'):
    if mode not in ['min', 'max']:
        raise NotImplementedError

    if os.path.exists(ckpt_dir):
        get_best = min if mode == 'min' else max
        ckpt_list = get_ckpt_list(ckpt_dir)

        if len(ckpt_list) == 1:
            return ckpt_list[0]
        elif len(ckpt_list) > 1 and os.path.exists(log_file_path):
            metrics = pd.read_csv(log_file_path, usecols=[metric]).squeeze().values
            metrics = [metrics[ckpt_nr-1] for ckpt_nr, _ in ckpt_list[1:]]
            best_idx = metrics.index(get_best(metrics)) if metrics else 0

            return ckpt_list[best_idx+1]
    return -1, None
