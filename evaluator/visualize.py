# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model.loss.fusion_loss_base import gene_multiScaleGTmap
import torch
import numpy as np
from evaluator.utils_mAP import calc_mAP, multi_channel_object_decode
# from evaluator.utils_mAP_old_version import calc_mAP, multi_channel_object_decode
import time
from loguru import logger

def drawBbox(corners, label_width, label_height, label_angle, color='blue'):
    leftup_x, leftup_y = corners[0,0], corners[1,0]
    frontCar_line = corners[:,0:2].T
    plt.gca().add_patch(
        patches.Rectangle((leftup_x, leftup_y), label_width, label_height,
                          angle=360 - label_angle,
                          edgecolor=color,
                          facecolor='none',
                          lw=2))
    plt.gca().add_patch(
        patches.Polygon(frontCar_line, closed=False, edgecolor='green', lw=3))


def rot2D(x,y,w,h,theta):
    '''
    :param x:       center x array
    :param y:       center y array
    :param theta:   degree(0-360)
    :param w:       bbox width
    :param h:       bbox height
    :return:        x_arr, y_arr
    '''
    theta = np.pi*theta/180
    rotMat2D = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    inputArr = np.array([x,y]).reshape(2,-1)
    leftUp_corners = np.array([-w/2,-h/2]).reshape(2,-1)
    rightUP_corners = np.array([w/2,-h/2]).reshape(2,-1)
    leftDown_corners = np.array([-w/2,h/2]).reshape(2,-1)
    rightDown_corners = np.array([w/2,h/2]).reshape(2,-1)
    corner_lu = np.dot(rotMat2D,leftUp_corners)+inputArr
    corner_ru = np.dot(rotMat2D,rightUP_corners)+inputArr
    corner_ld = np.dot(rotMat2D,leftDown_corners)+inputArr
    corner_rd = np.dot(rotMat2D,rightDown_corners)+inputArr
    corner_lu = corner_lu.reshape(-1)
    corner_ru = corner_ru.reshape(-1)
    corner_ld = corner_ld.reshape(-1)
    corner_rd = corner_rd.reshape(-1)
    corners = np.array([corner_lu[0], corner_ru[0], corner_ld[0], corner_rd[0],corner_lu[1], corner_ru[1], corner_ld[1], corner_rd[1]]).reshape(2,-1)
    return corners

def gtDraw(lidarpc, radar, bboxes, color='blue'):
    if radar != []:
        plt.imshow(radar,cmap='gray')
    if lidarpc != []:
        x_arr, y_arr = lidarpc[:,0], lidarpc[:,1]
        # show lidar point cloud
        plt.scatter(x_arr, y_arr, s=0.05, c='white')
    if len(bboxes)==0:
        return
    # show bbox
    ids, center_x, center_y, label_width, label_height, label_angle = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3], bboxes[:,4], bboxes[:,5]
    num = len(center_x)
    for i in range(num):
        id = ids[i]
        center_x_ = center_x[i]
        center_y_ = center_y[i]
        label_w_ = label_width[i]
        label_h_ = label_height[i]
        label_a_ = label_angle[i]
        corners = rot2D(center_x_, center_y_, label_w_, label_h_, label_a_)
        drawBbox(corners, label_w_, label_h_, label_a_)

def predDraw(lidarpc, radar, bboxes, color='red'):
    if radar != []:
        plt.imshow(radar,cmap='jet')
    if lidarpc != []:
        x_arr, y_arr = lidarpc[:,0], lidarpc[:,1]
        # show lidar point cloud
        plt.scatter(x_arr, y_arr, s=0.05, c='white')
    if len(bboxes)==0:
        return
    # show bbox
    center_x, center_y, label_width, label_height, label_angle = bboxes[:,0], bboxes[:,1], bboxes[:,3], bboxes[:,4], bboxes[:,5]
    num = len(center_x)
    for i in range(num):
        center_x_ = center_x[i]
        center_y_ = center_y[i]
        label_w_ = label_width[i]
        label_h_ = label_height[i]
        label_a_ = label_angle[i]
        corners = rot2D(center_x_, center_y_, label_w_, label_h_, label_a_)
        drawBbox(corners, label_w_, label_h_, label_a_, color)

def drawTxt(predBox, pr_table):
    num = predBox.shape[0]
    for i in range(num):
        x = predBox[i,0]
        y = predBox[i,1]
        iou = pr_table[i,1]
        plt.text(x, y, '{:.2f}'.format(iou), bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10)


def showPredResult(raw_lidarpc, raw_radar, gt_bboxes, prediction, nms_val, opts, cnt=None):
    '''
    :param raw_lidarpc: Nx4
    :param raw_radar:   widthxheight
    :param gt_bboxes:   mx6
    :param prediction:  kx11
    :param nms_val:     0.2 like
    :return:            null
    '''
    ################################
    # 预处理预测值和真实值
    ################################
    # Convert raw_radar and raw_lidarpc to torch tensors if they are numpy arrays
    if isinstance(raw_radar, np.ndarray):
        raw_radar = torch.from_numpy(raw_radar)
    if isinstance(raw_lidarpc, np.ndarray):
        raw_lidarpc = torch.from_numpy(raw_lidarpc)

    [width,height] = torch.squeeze(raw_radar).shape
    [pred_width,pred_height] = torch.squeeze(prediction[0]).shape[1], prediction[0].shape[2]
    scale = width/pred_width

    bbox_generator = multi_channel_object_decode()

    if opts.using_multi_scale == 1:
        # pred_objects, peakMap = bbox_generator.gene_objects(prediction[0], scale, opts)
        single_batch_output = []
        for j in range(len(prediction)):
            single_batch_output.append(prediction[j][0, ...])
        start_time = time.time()

        pred_objects, peakMap = bbox_generator.non_max_suppression(single_batch_output,
                                                                        [width, height], opts, 0.2)
        logger.info('nms time consuming: {}'.format(time.time()-start_time))
    else:
        pred_objects, peakMap = bbox_generator.non_max_suppression(prediction, [width,height], opts, nms_val)
    end_time = time.time()

    [gt_map, _] = gene_multiScaleGTmap(gt_bboxes, scale)
    gt_bboxes = gt_bboxes.cpu().detach().numpy()
    gt_map = np.squeeze(gt_map)
    pred_heatmap = torch.squeeze(prediction[0])[0, :, :]
    pred_heatmap = torch.sigmoid(pred_heatmap)
    pred_heatmap = pred_heatmap.cpu().detach().numpy()
    gt_heatmap = gt_map[0, :, :]
    raw_radar = torch.squeeze(raw_radar).cpu().detach().numpy()

    plt.figure(3)
    gtDraw(raw_lidarpc, raw_radar, gt_bboxes)
    predDraw(raw_lidarpc, raw_radar, pred_objects)
    plt.title('final result')
    plt.axis('off')
    plt.tight_layout(pad=0)

    return pred_objects, end_time
