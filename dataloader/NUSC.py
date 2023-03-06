from torch.utils.data import Dataset
from utils.range_projection import do_range_projection
import numpy as np
import os.path as osp
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm


class NuscenseDataset(Dataset):
    def __init__(self, cfg,stage='val'):
        # setting up nuscenes
        self.nusc = NuScenes(version=cfg['version'], dataroot=cfg['root'], verbose=False)
        train_scenes = splits.train
        val_scenes = splits.val
        available_scenes = self.available_scenes()
        available_scene_names = [s["name"] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        self.stage = stage

        self.train_scenes = set([available_scenes[available_scene_names.index(s)]["token"] for s in train_scenes])
        self.val_scenes = set([available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes])

        self.pointcloud_path = []
        self.label_path = []
        self.lidar_token = []

        self.learning_map = cfg['learning_map']
        self.maxPoints = cfg['max_points']

        self.loadfilenames()



    def available_scenes(self):
        available_scenes = []
        # print("total scene num:", len(self.nusc.scene))
        for scene in self.nusc.scene:
            scene_token = scene["token"]
            scene_rec = self.nusc.get("scene", scene_token)
            sample_rec = self.nusc.get("sample", scene_rec["first_sample_token"])
            sd_rec = self.nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, boxes, _ = self.nusc.get_sample_data(sd_rec["token"])
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

    def loadfilenames(self):
        scenes = self.train_scenes if self.stage == 'train' else self.val_scenes
        ref_chan = "LIDAR_TOP"
        for sample in self.nusc.sample:
            if not (sample["scene_token"] in scenes):
                continue
            ref_sd_token = sample["data"][ref_chan]
            data_path = self.nusc.get_sample_data_path(ref_sd_token)
            ref_lidarseg = self.nusc.get("lidarseg", ref_sd_token)
            self.lidar_token.append(ref_sd_token)
            self.pointcloud_path.append(data_path)
            self.label_path.append(osp.join(self.nusc.dataroot, ref_lidarseg['filename']))

        print('Loaded {} set'.format(self.stage))
    def loadPC(self,pointcloud_path, label_path=None):
        data = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        label = np.fromfile(label_path, dtype=np.uint8)
        label = np.expand_dims(label,axis=1)
        return data,label

    def class_mapping(self,labels,learning_map):
        lmap = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
        for k, v in learning_map.items():
            lmap[k] = v
        return lmap[labels]

    def limit_points(self,points,labels):
        concatData = np.hstack((points, labels))
        datalength = concatData.shape[0]

        mask = np.ones((self.maxPoints), dtype=np.bool_)
        if (datalength > self.maxPoints):
            concatData = concatData[:self.maxPoints, :]
        if (datalength < self.maxPoints):
            concatData = np.pad(concatData, [(0, self.maxPoints - datalength), (0, 0)], mode='constant')
            mask[datalength:] = False
        return concatData[:, 0:-1],concatData[:, -1],mask

    def __len__(self):
        return len(self.pointcloud_path)

    def __getitem__(self, index):
        points, labels = self.loadPC(self.pointcloud_path[index], self.label_path[index])
        labels = self.class_mapping(labels, self.learning_map)
        points, labels, mask = self.limit_points(points, labels)
        proj_range, proj_xyz, proj_remission, projected_labels, proj_x, proj_y, unproj_range = do_range_projection(points, labels)
        projected_rangeimg = np.concatenate((np.expand_dims(proj_remission, axis=0), np.expand_dims(proj_range, axis=0), np.rollaxis(proj_xyz, 2)), axis=0)
        return {"projected_rangeimg": projected_rangeimg, "projected_labels": projected_labels,
                "proj_depth": proj_range, "unproj_range": unproj_range, "proj_x": proj_x, "proj_y": proj_y,
                "points": points, "labels": labels, "mask": mask,"token":self.lidar_token[index]}



# config = yaml.safe_load(open('../configs/nusc_config.yaml', 'r'))
# dataset = NuscenseDataset(cfg=config)
# dict = dataset[0]
# print('wait here')




