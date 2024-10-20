# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import numba
import numpy as np
import matplotlib.pyplot as plt

@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    # coors,
                                    max_points=35,
                                    max_voxels=200000):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    ndim = 2
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(2, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            # ???z??voxel
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1]]
        voxelidx_true = int(coor[0]*grid_size[0] + coor[1])
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1]] = voxelidx
            # coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            # voxels[voxelidx, num, :4] = points[i]
            # num_points_per_voxel[voxelidx] += 1
            voxels[voxelidx_true, num, :4] = points[i]
            num_points_per_voxel[voxelidx_true] += 1
    # ???????9
    x_sum = 0
    y_sum = 0
    z_sum = 0
    for i in range(voxel_num):
        num_points = num_points_per_voxel[i]
        for j in range(num_points):
            x_sum = x_sum+voxels[i,j,0]
            y_sum = x_sum+voxels[i,j,1]
            z_sum = x_sum+voxels[i,j,2]
        xc = x_sum/max_points
        yc = y_sum/max_points
        zc = z_sum/max_points
        for j in range(max_points):
            xp = voxels[i,j,0]-xc
            yp = voxels[i,j,1]-yc
            voxels[i,j,4] = xc
            voxels[i,j,5] = yc
            voxels[i,j,6] = zc
            voxels[i,j,7] = xp
            voxels[i,j,8] = yp
    return voxel_num

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=200000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 2
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(2, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(points,
                     voxel_size = [0.2, 0.2, 0.4],
                     coors_range = [-32, -32, -3, 32, 32, 1],
                     max_points=34,
                     reverse_index=True,
                     max_voxels=200000):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points. ?????pillar
        coordinates: [M, 3] int32 tensor. contain 2d voxel index. ?????pillar?3d voxel??
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:5] - coors_range[:2]) / voxel_size[:2]
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    pillars_num = voxelmap_shape[0]*voxelmap_shape[1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(pillars_num, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # voxels = np.zeros(shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    voxels = np.zeros(shape=(max_voxels, max_points, 9), dtype=points.dtype)
    # coors = np.zeros(shape=(max_voxels, 2), dtype=np.int32)
    if reverse_index:
        # ????????????????????
        # import pdb;pdb.set_trace()
        voxel_num = _points_to_voxel_reverse_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, max_points, max_voxels)
    else:
        voxel_num = _points_to_voxel_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)
    # import pdb;pdb.set_trace()
    # coors = coors[:pillars_num]
    voxels = voxels[:pillars_num]
    num_points_per_voxel = num_points_per_voxel[:pillars_num]
    # import pdb;pdb.set_trace()
    # return voxels, coors, num_points_per_voxel, coor_to_voxelidx
    return voxels


@numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N, ), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices
