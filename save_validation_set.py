import warnings
warnings.filterwarnings('ignore')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
from dataloader.NUSC import NuscenseDataset
import matplotlib.pyplot as plt
import numpy as np
from trainer import PLModel
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle

if __name__ == '__main__':
    config = yaml.safe_load(open('configs/nusc_config.yaml', 'r'))
    dataset = NuscenseDataset(cfg=config)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    save_dir = 'E:\Research\Datasets/range_img_val_v2'

    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        x = batch["projected_rangeimg"].cpu().numpy()
        y = batch["projected_labels"].cpu().numpy()
        x = np.squeeze(x, axis=0).astype('float16')
        y = np.squeeze(y, axis=0).astype('int8')

        proj_depth = batch["proj_depth"].cpu().numpy()
        unproj_range = batch["unproj_range"].cpu().numpy()
        proj_x = batch['proj_x'].cpu().numpy()
        proj_y = batch['proj_y'].cpu().numpy()
        points = batch['points'].cpu().numpy()
        labels = batch['labels'].cpu().numpy()
        mask = batch['mask'].cpu().numpy()
        proj_depth = np.squeeze(proj_depth, axis=0).astype('float16')
        unproj_range = np.squeeze(unproj_range, axis=0).astype('float16')
        proj_x = np.squeeze(proj_x, axis=0).astype('int16')
        proj_y = np.squeeze(proj_y, axis=0).astype('int16')
        points = np.squeeze(points, axis=0).astype('float16')
        labels = np.squeeze(labels, axis=0).astype('int8')
        mask = np.squeeze(mask, axis=0)

        token = batch['token']

        dict_data = {'projected_rangeimg': x, 'projected_labels': y, 'proj_x': proj_x, 'proj_y': proj_y,
                     'proj_depth': proj_depth,
                     'unproj_range': unproj_range, 'points': points, 'labels': labels, 'mask': mask, 'token':token }

        file_name = str(batch_idx).zfill(6) + '_dict.npy'
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(dict_data, f)

    print("program-finished")