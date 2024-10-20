# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from evaluator.intersect_iou import box_intersection_area, box2corners
from evaluator.min_enclosing_box import smallest_bounding_box
import torch

def calc_iou(box1, box2):
    box1 = np.asarray([box1[0],box1[1],box1[3],box1[4],box1[5]*np.pi/180])
    box1 = box1.astype(float)
    box2 = box2[1:].astype(float)
    box2[-1] = box2[-1]*np.pi/180
    inter_area, corners = box_intersection_area(box1, box2)
    area1 = box1[2]*box1[3]
    area2 = box2[2]*box2[3]
    u = area1 + area2 - inter_area
    iou = inter_area / (u+1e-6)
    return iou, u

def calc_giou(box1, box2):
    iou, u = calc_iou(box1, box2)
    box1 = np.asarray([box1[0],box1[1],box1[3],box1[4],box1[5]*np.pi/180])
    box2 = box2[1:]
    box2[-1] = box2[-1]*np.pi/180
    corners1 = box2corners(*box1) # 4, 2
    corners2 = box2corners(*box2) # 4, 2
    tensor1 = torch.FloatTensor(np.concatenate([corners1, corners2], axis=0))
    w, h = smallest_bounding_box(tensor1)
    area_c = w*h
    giou = iou+(area_c-u)/area_c
    return giou

def calc_diou(box1, box2):
    iou, u = calc_iou(box1, box2)
    box1 = np.asarray([box1[0],box1[1],box1[3],box1[4],box1[5]*np.pi/180])
    box2 = box2[1:]
    box2[-1] = box2[-1]*np.pi/180
    corners1 = box2corners(*box1) # 4, 2
    corners2 = box2corners(*box2) # 4, 2
    tensor1 = torch.FloatTensor(np.concatenate([corners1, corners2], axis=0))
    w, h = smallest_bounding_box(tensor1)
    c2 = w*w+h*h
    x_offset = box1[0] - box2[1]
    y_offset = box1[1] - box2[2]
    d2 = x_offset*x_offset + y_offset*y_offset
    diou_loss = 1-iou+d2/c2
    return diou_loss

def calc_ciou(box1, box2):
    iou, u = calc_iou(box1, box2)
    box1 = box1[0,1,3,4,5]
    box2 = box2[1:]
    corners1 = box2corners(*box1) # 4, 2
    corners2 = box2corners(*box2) # 4, 2
    tensor1 = torch.FloatTensor(np.concatenate([corners1, corners2], axis=0))
    w, h = smallest_bounding_box(tensor1)
    c2 = w*w+h*h
    x_offset = box1[0] - box2[1]
    y_offset = box1[1] - box2[2]
    d2 = x_offset*x_offset + y_offset*y_offset
    w_gt, h_gt = box2[2], box2[3]
    w_pred, h_pred = box1[2], box1[3]
    arctan = np.arctan(w_gt/h_gt) - np.arctan(w_pred/h_pred)
    v = (4/np.pi*np.pi)*np.power((arctan), 2)
    s = 1 - iou
    alpha = v / (s + v)
    w_temp = 2*w_pred
    ar = (8 / (np.pi ** 2)) * arctan * ((w_pred - w_temp) * h_pred)
    ciou_loss = iou - (u + alpha * ar)
    return ciou_loss
