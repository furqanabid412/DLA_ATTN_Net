import numpy as np
import numba

# settings for nuscenes dataset
# proj_fov_up = 10
# proj_fov_down = -30
# proj_W = 1024
# proj_H = 32

proj_fov_up = 10
proj_fov_down = -30
proj_W = 1024
proj_H = 32



# @numba.jit(nopython=True)
def do_range_projection(points,labels):
    """ Project a pointcloud into a spherical projection image.projection.
          Function takes no arguments because it can be also called externally
          if the value of the constructor was not set (in case you change your
          mind about wanting the projection)
    """
    remissions = points[:, 3]
    points = points [:,:3]


    proj_range = np.full((proj_H, proj_W), -1,dtype=np.float32) # projected range image - [H,W] range (-1 is no data)
    unproj_range = np.zeros((0, 1), dtype=np.float32) # unprojected range (list of depths for each point)
    proj_xyz = np.full((proj_H, proj_W, 3), -1,dtype=np.float32) # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    proj_remission = np.full((proj_H, proj_W), -1,dtype=np.float32) # projected remission - [H,W] intensity (-1 is no data)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1, dtype=np.int32)
    # for each point, where it is in the range image
    proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
    proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y
    # mask containing for each pixel, if it contains a point or not
    proj_mask = np.zeros((proj_H, proj_W),dtype=np.int32)  # [H,W] mask
    proj_sem_label = np.full((proj_H, proj_W), 0, dtype=np.int32)


    # laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    # depth = np.linalg.norm(points, 2, axis=1)

    depth = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    # pitch = np.nan_to_num(pitch)
    # pitch1 = np.copy(pitch)
    pitch[np.isnan(pitch)] = 0

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W                              # in [0.0, W]
    proj_y *= proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = points[order]
    remission = remissions[order]

    orig_proj_y = proj_y
    orig_proj_x = proj_x

    proj_y = proj_y[order]
    proj_x = proj_x[order]
    sem_labels = labels[order]
    # sem_labels = np.squeeze(sem_labels, axis=1)

    # assing to images
    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = points
    proj_remission[proj_y, proj_x] = remission
    proj_idx[proj_y, proj_x] = indices
    proj_mask = (proj_idx > 0).astype(np.int32)
    proj_sem_label[proj_y, proj_x] = sem_labels


    # return proj_range,proj_xyz,proj_remission,proj_sem_label,proj_x,proj_y,unproj_range
    return proj_range,proj_xyz,proj_remission,proj_sem_label,orig_proj_x,orig_proj_y,unproj_range



# final
'''
# FIX the points,label array size to maxPoints
# If there are more points, trim it (usually points are less than 40k)
# If there are more points, pad zeros at the end of array
points = data['lidar']['points']
if data['lidar']['lidarseg'].ndim == 1:
    labels = np.expand_dims(data['lidar']['lidarseg'], axis=1)
else:
    labels = data['lidar']['lidarseg']
concatData = np.hstack((points, labels))
datalength = concatData.shape[0]

mask = np.ones((self.maxPoints),dtype=np.bool_)
if (datalength > self.maxPoints):
    concatData = concatData[:self.maxPoints, :]
if (datalength < self.maxPoints):
    concatData = np.pad(concatData, [(0, self.maxPoints - datalength), (0, 0)], mode='constant')
    mask[datalength:]=False
points = concatData[:, 0:-1]
labels = concatData[:, -1]

# map labels to prediction classes
from tools.utils import class_mapping
# labels = class_mapping(labels.astype(np.int))

proj_range, proj_xyz, proj_remission, projected_labels, proj_x, proj_y,unproj_range = do_range_projection(points, labels)
projected_rangeimg = np.concatenate((np.expand_dims(proj_remission, axis=0), np.expand_dims(proj_range, axis=0), np.rollaxis(proj_xyz, 2)), axis=0)
dict = {"projected_rangeimg": projected_rangeimg,"projected_labels": projected_labels,
        "proj_depth":proj_range,"unproj_range":unproj_range,"proj_x":proj_x,"proj_y":proj_y,"points":points,"labels":labels,"mask":mask}

'''


