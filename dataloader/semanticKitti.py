import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import Dataset
import math
from utils.range_projection import do_range_projection

class SemanticKitti(Dataset):
    def __init__(self,cfg):
        self.root = cfg['dataset']['root']
        self.seq = cfg['dataset']['split']['train']
        self.kitti_lmap = cfg['dataset']['learning_map']
        self.label_colormap = cfg['dataset']['color_map']
        self.kitti_to_nusc_lmap = cfg['dataset']['kitti_to_nusc']
        self.maxPoints = cfg['dataset']['max_points']
        self.pointcloud_path = []
        self.label_path = []
        self.loadfilenames()

    def loadfilenames(self):
        for s in self.seq:
            folder_pc = os.path.join(self.root,'{:02d}'.format(s), 'velodyne')
            folder_lb = os.path.join(self.root,'{:02d}'.format(s), 'labels')

            file_pc = os.listdir(folder_pc)
            file_pc.sort(key=lambda x: str(x[:-4]))
            file_lb = os.listdir(folder_lb)
            file_lb.sort(key=lambda x: str(x[:-4]))

            for index in range(len(file_pc)):
                self.pointcloud_path.append('%s/%s' % (folder_pc, file_pc[index]))
                self.label_path.append('%s/%s' % (folder_lb, file_lb[index]))

    def loadPC(self,pointcloud_path, label_path=None):
        data = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        label = np.fromfile(label_path, dtype=np.int32).reshape((-1, 1))
        label = label & 0xFFFF
        data[:,-1] = np.ceil(data[:,-1]*255.0)
        return data, label.astype(int)

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


    # def visualize_range_image(self,y, title, image_name):
    #     my_dpi = 600
    #     factor = 32
    #     w = (960 / my_dpi) * factor
    #     h = (32 / my_dpi) * factor
    #
    #     fig, ax = plt.subplots(layout='constrained')
    #     ax.set_axis_off()
    #     fig.set_size_inches(w, h, forward=True)
    #     # ax.set_title(title)
    #     # label_colormap = yaml.safe_load(open('configs/colormap.yaml', 'r'))
    #     # label_colormap = label_colormap['short_color_map']
    #     colors = np.array([[self.label_colormap[val] for val in row] for row in y], dtype='B')
    #     ax.imshow(colors)
    #     ax.figure.savefig(image_name, dpi=my_dpi)

    def __len__(self):
        return len(self.pointcloud_path)

    def __getitem__(self, index):

        points, labels = self.loadPC(self.pointcloud_path[index],self.label_path[index])
        labels = self.class_mapping(labels,self.kitti_lmap)
        labels = self.class_mapping(labels,self.kitti_to_nusc_lmap)
        points, labels, mask = self.limit_points(points, labels)
        # labels = np.squeeze(labels,axis=1)
        proj_range, proj_xyz, proj_remission, projected_labels, proj_x, proj_y, unproj_range = do_range_projection(points, labels)
        projected_rangeimg = np.concatenate((np.expand_dims(proj_remission, axis=0), np.expand_dims(proj_range, axis=0),
                                             np.rollaxis(proj_xyz, 2)), axis=0)

        # proj_remission = np.ceil(proj_remission*255.0)

        # self.visualize_range_image(y=projected_labels,title=None,image_name='old.png')
        #
        # # labels = np.expand_dims(labels)
        # labels = self.class_mapping(labels, self.kitti_lmap)
        # labels = self.class_mapping(labels, self.kitt_to_nusc_lmap)
        # # labels = np.squeeze(labels, axis=1)
        # proj_range, proj_xyz, proj_remission, projected_labels, proj_x, proj_y, unproj_range = do_range_projection(
        #     points, labels)
        # projected_rangeimg = np.concatenate((np.expand_dims(proj_remission, axis=0), np.expand_dims(proj_range, axis=0),
        #                                      np.rollaxis(proj_xyz, 2)), axis=0)
        #
        # nusc_config = yaml.safe_load(open('../configs/config.yaml', 'r'))
        # new_colormap = nusc_config['dataset']['color_map']
        # self.label_colormap = new_colormap
        # self.visualize_range_image(y=projected_labels,title=None,image_name='new.png')

        return {"projected_rangeimg": projected_rangeimg, "projected_labels": projected_labels,
                "proj_depth": proj_range, "unproj_range": unproj_range, "proj_x": proj_x, "proj_y": proj_y,
                "points": points, "labels": labels, "mask": mask}



# config = yaml.safe_load(open('../configs/kitti_config.yaml', 'r'))
# dataset = SemanticKitti(cfg=config)
# example = dataset[0]
