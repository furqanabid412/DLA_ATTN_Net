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
from torch.utils.data import Dataset
from utils.range_projection import do_range_projection
import numpy as np
import os.path as osp
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm


def available_scenes(nusc):
    available_scenes = []
    # print("total scene num:", len(self.nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    # print("exist scene num:", len(available_scenes))
    return available_scenes


if __name__ == '__main__':

    cfg =  config = yaml.safe_load(open('configs/nusc_config.yaml', 'r'))



    nusc = NuScenes(version=cfg['version'], dataroot=cfg['root'], verbose=True)
    val_scenes = np.load('saved/val_scenes_token.npy',allow_pickle=True).tolist()
    val_scenes = list(val_scenes)
    scene_no = 0
    for scene_token in val_scenes:
    # scene_token = val_scenes[scene_no]
    # scene_token = 'b51869782c0e464b8021eb798609f35f'
        out_path = os.path.join('E:\source_code\saved/rendered_results','gt_{}_token_{}.avi'.format(scene_no,scene_token))
        nusc.render_scene_lidarseg(scene_token=scene_token,out_path=out_path,imsize=(3*640, 3*360),dpi=200,
                                   lidarseg_preds_folder = 'E:/Research/Datasets/predictions/')
        # nusc.render_scene_lidarseg(scene_token=scene_token, out_path=out_path, imsize=(3 * 640, 3 * 360), dpi=200,
        #                            lidarseg_preds_folder=None)


    print('wait here')