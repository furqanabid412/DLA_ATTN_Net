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
import cv2
from typing import Tuple, List, Iterable
from nuscenes.lidarseg.lidarseg_utils import plt_to_cv2,colormap_to_colors
from nuscenes.utils.data_classes import LidarPointCloud

def render_scene_lidarseg(nusc, scene_token,out_path = None, filter_lidarseg_labels = None, imsize = (640, 360),
                          freq = 2, dpi = 200,lidarseg_preds_folder = None, show_panoptic = False):


    scene_record = nusc.get('scene', scene_token)
    total_num_samples, first_sample_token, last_sample_token = scene_record['nbr_samples'],scene_record['first_sample_token'], scene_record['last_sample_token']
    current_token = first_sample_token
    layout = {'CAM_FRONT_LEFT': (0, 0), 'CAM_FRONT': (imsize[0], 0), 'CAM_FRONT_RIGHT': (2 * imsize[0], 0),
        'CAM_BACK_LEFT': (0, imsize[1]),'CAM_BACK': (imsize[0], imsize[1]),'CAM_BACK_RIGHT': (2 * imsize[0], imsize[1]),}
    horizontal_flip = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    slate = np.ones((2 * imsize[1], 3 * imsize[0], 3), np.uint8)
    range_image = np.ones((32, 1024, 3), np.uint8)

    output_size = (1024, 32)

    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # out = cv2.VideoWriter(out_path, fourcc, freq, slate.shape[1::-1])

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('E:\source_code\saved/range_pred.avi', fourcc, freq, output_size)

    keep_looping = True
    i = 0
    while keep_looping:
        if current_token == last_sample_token:
            keep_looping = False
        sample_record = nusc.get('sample', current_token)

        for camera_channel in layout:
            pointsensor_token = sample_record['data']['LIDAR_TOP']
            camera_token = sample_record['data'][camera_channel]
            lidarseg_preds_bin_path = osp.join(lidarseg_preds_folder, pointsensor_token + '_lidarseg.bin')

            points, coloring, im = nusc.explorer.map_pointcloud_to_image(pointsensor_token, camera_token,
                                                                render_intensity=False,
                                                                show_lidarseg=not show_panoptic,
                                                                show_panoptic=show_panoptic,
                                                                filter_lidarseg_labels=filter_lidarseg_labels,
                                                                lidarseg_preds_bin_path=lidarseg_preds_bin_path)

            mat = plt_to_cv2(points, coloring, im, imsize, dpi=dpi)

            if camera_channel in horizontal_flip:
                mat = cv2.flip(mat, 1)

            slate[layout[camera_channel][1]: layout[camera_channel][1] + imsize[1], layout[camera_channel][0]:layout[camera_channel][0] + imsize[0], :] = mat

            # range-image visualization
            pointsensor = nusc.get('sample_data', pointsensor_token)
            pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
            data = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)[:, :4]
            # ref_lidarseg = nusc.get("lidarseg", pointsensor_token)
            ref_lidarseg = pointsensor_token+'_lidarseg.bin'
            lidarseg_path = osp.join(lidarseg_preds_folder, ref_lidarseg)
            label = np.fromfile(lidarseg_path, dtype=np.uint8)
            # label = np.expand_dims(label, axis=1)
            _, _, _, projected_labels, _, _, _ = do_range_projection (data, label)
            colors = colormap_to_colors(nusc.colormap, nusc.lidarseg_name2idx_mapping)
            range_image = colors[projected_labels]
            range_image = range_image[:,:,::-1]
            # print('wait here')

        # window_name = 'cameras'
        # cv2.imshow(window_name, slate)
        window_name = 'range_image'
        cv2.imshow(window_name,range_image)

        range_image = cv2.normalize(range_image,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

        key = cv2.waitKey(1)
        if key == 32:
            key = cv2.waitKey()

        if key == 27:
            plt.close('all')
            # if out_path:
            #     # out.write(slate)
            #     # out.release()
            cv2.destroyAllWindows()
            break

        plt.close('all')

        out.write(range_image)
        # out.write(range_image.astype(np.uint8))

        next_token = sample_record['next']
        current_token = next_token

        i += 1

    cv2.destroyAllWindows()
    out.release()
    if out_path:
        assert total_num_samples == i, 'Error: There were supposed to be {} keyframes, ' \
                                       'but only {} keyframes were processed'.format(total_num_samples, i)
        # out.release()
        # range_out.release()



if __name__ == '__main__':
    cfg =  config = yaml.safe_load(open('configs/nusc_config.yaml', 'r'))
    nusc = NuScenes(version=cfg['version'], dataroot=cfg['root'], verbose=True)

    render_scene_lidarseg(nusc, scene_token='fcbccedd61424f1b85dcbf8f897f9754',
                          out_path='E:\source_code\saved/video.avi',lidarseg_preds_folder= 'E:/Research/Datasets/predictions/')

