# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import os
from data.datasets.oxford_dataset.point_cloud_ops import points_to_voxel
import cv2
from loguru import logger
import time

inputDataPath = '/scratch/project/cpautodriving/yanlongyang/RaLiBEV/2019-01-10-11-46-21-radar-oxford-10k/processed/'

class OxfordDataset(Dataset):
    def __init__(self, annotation_lines, opt, input_shape=[320, 320], train=True):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train
        self.opt = opt
        self.randRotAngle = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        lidar_pillar, lidarData, radar_data, box, class_arr = self.get_data(self.annotation_lines[index])
        return lidar_pillar, radar_data, box, class_arr

    def randRotBbox(self, x, y, rotation_center, angle, shift_angle):
        new_angle = angle + shift_angle
        if new_angle > 180:
            new_angle = new_angle - 360
        elif new_angle < -180:
            new_angle = new_angle + 360
        shift_angle = np.pi * shift_angle / 180
        rotMat2D = np.array([[np.cos(shift_angle), np.sin(shift_angle)], [-np.sin(shift_angle), np.cos(shift_angle)]])
        inputArr = np.array([x,y], dtype='float')-rotation_center
        new_xy = rotMat2D@inputArr
        new_xy = new_xy + rotation_center
        return new_xy, new_angle

    def randRotPC(self, points_3d, rotation_center, shift_angle):
        shift_angle = np.pi * (90+shift_angle) / 180
        rotMat2D = np.array([[np.cos(shift_angle), np.sin(shift_angle)], [-np.sin(shift_angle), np.cos(shift_angle)]])
        inputArr = points_3d[...,:2] -  rotation_center
        new_BEV_2d = np.dot(rotMat2D, inputArr.T)
        new_BEV_2d = new_BEV_2d.T + rotation_center
        new_pc = np.concatenate((new_BEV_2d, points_3d[...,2:]), axis=1)
        return new_pc

    def randRotRadarImg(self, image, angle):
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        newW = w
        newH = h
        M[0, 2] += (newW - w) / 2
        M[1, 2] += (newH - h) / 2
        return cv2.warpAffine(image, M, (w, h))

    def readLidarData(self, filename):
        lidarData = np.fromfile(filename, dtype=np.float32)
        lidarData = np.reshape(lidarData, (-1, 4))
        # ---------------------------------
        # random rotate
        # ---------------------------------
        center_bev = np.array([0,0], dtype='float')
        lidarData = self.randRotPC(lidarData, center_bev, self.randRotAngle)
        # turn the point cloud data format to voxel format
        pillars = points_to_voxel(lidarData)
        lidarData[..., 0] = 5*lidarData[..., 0] + 319/2
        lidarData = lidarData[lidarData[..., 0] > 0,...]
        lidarData = lidarData[lidarData[..., 0] < 320,...]
        lidarData[..., 1] = 5*lidarData[..., 1] + 319/2
        lidarData = lidarData[lidarData[..., 1] > 0,...]
        lidarData = lidarData[lidarData[..., 1] < 320,...]
        lidarData[..., 2] = 5*(lidarData[...,2])
        return pillars, lidarData

    def readLabel(self, labelFilename):
        this_labelFilename = os.path.join(inputDataPath, 'label_2d', labelFilename)
        class_arr = []
        id_arr = []
        leftup_x_arr = []
        leftup_y_arr = []
        center_x_arr = []
        center_y_arr = []
        width_arr = []
        height_arr = []
        angle_arr = []
        with open(this_labelFilename, "r") as f:
            for line in f.readlines():
                tmp = line.split(' ')
                center_x = 5 * float(tmp[2]) + 319 / 2
                center_y = 5 * float(tmp[3]) + 319 / 2
                width = 5 * float(tmp[4])
                height = 5 * float(tmp[5])
                # angle adjust
                angle = float(tmp[6])
                rotation_center = np.array([319 / 2, 319 / 2], dtype='float')
                try:
                    new_xy, new_angle = self.randRotBbox(center_x, center_y, rotation_center, angle, self.randRotAngle)
                    if new_xy[0] < 0 or new_xy[0] > 320 or new_xy[1] < 0 or new_xy[1] > 320:
                        continue
                except:
                    import pdb;pdb.set_trace()
                class_arr.append(tmp[0])
                id_arr.append(int(tmp[1]))
                center_x_arr.append(new_xy[0])
                center_y_arr.append(new_xy[1])
                width_arr.append(width)
                height_arr.append(height)
                angle_arr.append(new_angle)
        return class_arr, id_arr, center_x_arr, center_y_arr, width_arr, height_arr, angle_arr

    def get_radarImg(self, filename):
        radar_data = Image.open(filename)
        radar_data = np.asarray(radar_data)
        new_radar_data = self.randRotRadarImg(radar_data, self.randRotAngle)
        return new_radar_data

    def get_data(self, annotation_line):
        if self.opt.rotate_angle == 1:
            self.randRotAngle = 360 * np.random.rand()
        line = annotation_line.split()
        frame_num = line[0]
        timestamp = line[1]
        lidarFilename = '{}.bin'.format(timestamp)
        lidar_rawDataFilename = os.path.join(inputDataPath, 'lidar', lidarFilename)
        pillars, lidarData = self.readLidarData(lidar_rawDataFilename)
        radarFilename = '{}.jpg'.format(timestamp)
        radar_rawDataFilename = os.path.join(inputDataPath, 'radar', radarFilename)
        radar_data = self.get_radarImg(radar_rawDataFilename)
        labelFilename = '{}.txt'.format(timestamp)
        [class_arr, id, label_center_x, label_center_y, label_width, label_height, label_angle] = self.readLabel(
            labelFilename)
        box = np.zeros((len(id),6))
        box[:,0] = id[:]
        box[:,1] = label_center_x[:]
        box[:,2] = label_center_y[:]
        box[:,3] = label_width[:]
        box[:,4] = label_height[:]
        box[:,5] = label_angle[:]
        return pillars, lidarData, radar_data, box, class_arr

def oxford_dataset_collate(batch):
    lidar_pillar = []
    radardata = []
    bboxes = []
    labels = []
    for lidarpillar, radar, box, label in batch:
        lidar_pillar.append(lidarpillar)
        radardata.append(radar)
        bboxes.append(box)
        labels.append(label)
    lidar_pillar = np.array(lidar_pillar)
    radardata = np.array(radardata)
    return lidar_pillar, radardata, bboxes, labels
