# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment
from evaluator.calc_iou import calc_iou, calc_diou
import os
import numba


@numba.jit(nopython=True)
def twoToTen(twoBin, binNum):
    sum=np.zeros(twoBin.shape[1])
    for i in range(0, binNum):
        num = twoBin[i, :] * (2 ** (binNum - i - 1))
        sum += num
    return sum

@numba.jit(nopython=True)
def grayToTen(grayBin, binNum):
    binay = np.zeros(grayBin.shape)
    binay[0, :] = grayBin[0, :]
    for i in range(1, binNum):
        tempArr =  (grayBin[i, :] != binay[i - 1, :])+0
        binay[i, :] = tempArr
    return twoToTen(binay, binNum)

@numba.jit(nopython=True)
def multi_bin_angle_decode(encoded_arr, angle_offset):
    '''
    :param encoded_arr:     dense encoded bin array
    :param angle_offset:    offset
    :return:
    '''
    binNum = len(encoded_arr)
    angle_res = 180.0 / (2 ** binNum)
    int_binNum = grayToTen(encoded_arr, binNum)
    # 解码的时候，从angle offset中能看出来角度的正负
    flag = (angle_offset > 0)+0
    flag = 2*flag-1
    int_binNum *= flag
    angle = angle_offset + int_binNum * angle_res
    return angle

@numba.jit(nopython=True)
def gene_gaussianMap(gaussian_map, mu, conv):
    covmat_inv = np.linalg.inv(conv).astype(np.float64)
    [width, height] = gaussian_map.shape
    for i in range(height):
        for j in range(width):
            arr = np.array([[j], [i]]).astype(np.float64)
            # import pdb;pdb.set_trace()
            v_exp = -np.dot((arr-mu).T, covmat_inv)
            v_exp = np.dot(v_exp, arr-mu)
            value = np.exp(v_exp)
            value = value[0,0]
            if value > gaussian_map[i,j]:
                gaussian_map[i,j] = value


class peakProcess(object):
    def __init__(self, pfa=1e-2, guardWin=1, trainWin=1, cfar_type='staticThresh', thresh=0.2):
        # cfar parameters
        self.pfa = pfa
        self.thresh = thresh
        self.cfar_type = cfar_type
        self.guardWin = guardWin
        self.trainWin = trainWin
        self.CFAR_UNITS = 1 + 2 * self.guardWin + 2 * self.trainWin
        self.N = (self.CFAR_UNITS * self.CFAR_UNITS) - (((self.trainWin * 2) + 1) * ((self.trainWin * 2) + 1))
        if self.cfar_type == 'CA':
            # self.alpha = self.N*(np.power(self.pfa, -1/self.N) - 1)
            self.alpha = (np.power(self.pfa, -1 / self.N) - 1)
        # 意思是简单地用倍数关系
        elif self.cfar_type == 'staticThresh':
            self.alpha = pfa

    # 基本思路是8邻域的峰值提取，就是一个聚类问题
    def peakPrune(self, heatmap, cfarMap):
        [width, height] = heatmap.shape
        new_heatmap = heatmap*cfarMap
        peakMap = np.zeros(cfarMap.shape)
        peakIndex = np.zeros((2, width * height))
        peakCnt = 0
        for i in range(height - 1):
            for j in range(width - 1):
                center_x = i + 1
                center_y = j + 1
                val = new_heatmap[center_x, center_y]
                neighborMat = new_heatmap[center_x - 1:center_x + 2, center_y - 1:center_y + 2]
                # # 难道这样的置0操作改变了内存吗: 是的，真的改变了
                # neighborMat[1,1] =  0
                if val >= neighborMat.max() and val != 0:
                    # print('number of peaks: {}'.format(peakCnt))
                    # print(center_x,center_y)
                    # import pdb;pdb.set_trace()
                    peakMap[center_x, center_y] = 1
                    # import pdb;pdb.set_trace()
                    peakIndex[:, peakCnt] = np.array([center_y, center_x])
                    peakCnt += 1
        peakIndex = peakIndex[:, :peakCnt]
        return peakMap, peakIndex

    def cfar2d(self, heatmap):
        [width, height] = heatmap.shape
        new_width = width + 2 * (self.guardWin + self.trainWin)
        new_height = height + 2 * (self.guardWin + self.trainWin)
        # paddingMap = np.zeros((new_width, new_height))
        paddingMap = torch.zeros([new_width, new_height])
        paddingMap[self.guardWin + self.trainWin:width + self.guardWin + self.trainWin,
        self.guardWin + self.trainWin:height + self.guardWin + self.trainWin] = heatmap
        # import pdb;pdb.set_trace()
        cfarMap = torch.zeros([width, height])
        for i in range(height - self.CFAR_UNITS):
            center_cell_x = i + self.guardWin + self.trainWin
            for j in range(width - self.CFAR_UNITS):
                center_cell_y = j + self.guardWin + self.trainWin
                average = 0
                # 取trainWin以内guardWin以外的pixel
                trainWin_mat = paddingMap[
                               center_cell_x - self.guardWin - self.trainWin:center_cell_x + self.guardWin + self.trainWin + 1,
                               center_cell_y - self.guardWin - self.trainWin:center_cell_y + self.guardWin + self.trainWin + 1]
                guardWin_mat = paddingMap[
                               center_cell_x - self.guardWin:center_cell_x + self.guardWin + 1,
                               center_cell_y - self.guardWin:center_cell_y + self.guardWin + 1]
                winSize = (2 * (self.guardWin + self.trainWin) ** 2) - (2 * self.guardWin) ** 2
                average = (trainWin_mat.sum() - guardWin_mat.sum()) / winSize
                # for k in range(self.CFAR_UNITS):
                #     for l in range(self.CFAR_UNITS):
                #         if (k >= self.guardWin) and (k < (self.CFAR_UNITS - self.guardWin)) and (l >= self.guardWin) and (l < (self.CFAR_UNITS - self.guardWin)):
                #             continue
                #         # average += heatmap[i + k, j + l]
                #         average += paddingMap[i + k, j + l]
                # average /= self.N
                # import pdb;pdb.set_trace()
                # if heatmap[center_cell_x, center_cell_y] > (average * self.alpha):
                if paddingMap[center_cell_x, center_cell_y] > (average * self.alpha):
                    cfarMap[i, j] = 1
        return cfarMap

    def staticThresh(self, heatmap, thresh):
        peakMap = (heatmap > thresh).int()
        # [width,height] = heatmap.shape
        # peakMap = np.zeros((width,height))
        # for i in range(heatmap.shape[0]):
        #     for j in range(heatmap.shape[1]):
        #         val = heatmap[i,j]
        #         if val > thresh:
        #             peakMap[i,j] = 1
        return peakMap


class gaussian_decoder(object):
    def __init__(self):
        self.iou_calculator = peakProcess()
        pass

    #---------------------------------
    # 8邻域聚类
    # 长宽遍历，找到非0值后，对比其8邻域内是否有非0标签，
    # 如果有，则将该位置置同样标签，并将最新标签重置为当前标签
    # 如果没有，则将最新标签赋值给当前索引
    # 如果当前值发生向上跳变，则是从0-1进入了某个区块中，此时类别号加一
    # ---------------------------------
    def nn_clustering(self, bin_heatmap):
        [width, height] = bin_heatmap.shape
        cluster_num = 0
        cluster_map = torch.zeros([width, height])
        last_val = 0
        last_cluster_num = 0
        for i in range(height-1):
            for j in range(width-1):
                center_x = i + 1
                center_y = j + 1
                val = bin_heatmap[center_x, center_y]
                if val != 0:
                    if val - last_val > 0:
                        cluster_num += 1
                    neighbor_class = cluster_map[center_x-1:center_x+2,center_y-1:center_y+2]
                    nn_cluster_num = neighbor_class.max()
                    if nn_cluster_num > 0:
                        cluster_map[center_x, center_y] = nn_cluster_num
                        cluster_num = last_cluster_num
                    else:
                        cluster_map[center_x, center_y] = cluster_num
                        last_cluster_num = cluster_num
                    # print('current location: {}'.format([center_x, center_y]))
                    # print('neighbor label max: {}'.format(nn_cluster_num))
                    # print('this pixel label: {}'.format(cluster_map[center_x, center_y]))
                    # # import pdb;pdb.set_trace()
                last_val = val
        return cluster_map

    # ---------------------------------
    # Guassian Mixture fitting method 1
    # 这个方法有问题，因为聚类的时候缺乏了峰值信息，导致多个峰变成了一个
    # ---------------------------------
    def guassian_mixture_fit_v1(self, pred_heatmap, thresh=0.5):
        # 1. 先二值化
        # cfarMap = self.iou_calculator.staticThresh(pred_heatmap, thresh)
        cfarMap = self.iou_calculator.cfar2d(pred_heatmap)
        # 2. 8邻域聚类: nearest neighbor clustering
        cluster_map = self.nn_clustering(cfarMap)
        # 3. 对每个类别计算加权均值和方差
        cluster_num = np.int(cluster_map.max().item())
        mu_arr = torch.zeros([cluster_num, 2]).cuda()
        conv_arr = torch.zeros([cluster_num, 2, 2]).cuda()

        new_gaussian_map = np.zeros(cfarMap.shape)
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_title('prediction heatmap')
        plt.imshow(pred_heatmap.cpu().detach().numpy(), origin='lower')
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title('CFAR result')
        plt.imshow(cfarMap.cpu().detach().numpy(), origin='lower')
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_title('cluster result')
        plt.imshow(cluster_map.cpu().detach().numpy(), origin='lower')

        for i in range(cluster_num):
            mask = (cluster_map == i+1).int().cuda()
            weight_sum = (pred_heatmap*mask).sum()
            indexes = torch.nonzero(mask)
            weight = pred_heatmap[indexes[...,0], indexes[...,1]]
            # plt.plot(indexes[..., 1].cpu().detach().numpy(), indexes[..., 0].cpu().detach().numpy(), 'yo')
            # 计算加权均值
            # u=sum(w*x)/sum(w)
            mu = torch.cat((torch.mul(indexes[...,1], weight).view(-1,1), torch.mul(indexes[...,0], weight).view(-1,1)), dim=1)
            mu_arr[i,...] = torch.sum(mu, dim=0)/weight_sum
            # 计算加权协方差
            # conv = sum(w*(x-u).T*(x-u))
            # import pdb;pdb.set_trace()
            indexes = torch.cat((indexes[...,1].view(-1,1), indexes[...,0].view(-1,1)), dim=1)
            conv = torch.matmul(torch.mul((indexes-mu_arr[i,...]), weight.view(-1,1)).t(), indexes-mu_arr[i,...])
            conv = (conv/weight_sum)
            conv_arr[i,...] = conv
            # # 新生成的gaussian map看看
            # new_mu, new_conv = mu_arr[i,...].cpu().detach().numpy().reshape((2,1)), conv.cpu().detach().numpy().T
            # if new_conv.max() == 0:
            #     continue
            # gene_gaussianMap(new_gaussian_map, new_mu, new_conv)

        # plt.plot(mu_arr[...,0].cpu().detach().numpy(), mu_arr[...,1].cpu().detach().numpy(), 'ro')
        # ax4 = plt.subplot(2, 2, 4)
        # ax4.set_title('new gaussian map')
        # plt.imshow(new_gaussian_map, origin='lower')
        # plt.show()

        return mu_arr, conv_arr

    # ---------------------------------
    # seed_clustering: 种子点出发的聚类方法
    # 采用另一个流程：
    # 1.先做峰值提取，找到种子点
    # 2.从种子点出发搜索周围
    # 3.得到每一个分类
    # ---------------------------------
    def seed_clustering(self, cfarMap, peakMap):
        # 取所有的备选点和种子点
        cfarMap_index = torch.nonzero(cfarMap)
        peakMap_index = torch.nonzero(peakMap)
        # 遍历备选点，计算其到每个种子点的'距离'
        seed_num = peakMap_index.shape[0]
        clusterMap = torch.zeros(cfarMap.shape)
        if seed_num == 0:
            return clusterMap
        for i in range(cfarMap_index.shape[0]):
            tobe_classified_index = cfarMap_index[i,...]
            distance = torch.zeros((seed_num, ))
            for j in range(seed_num):
                seed = peakMap_index[j, ...]
                distance[j] = ((tobe_classified_index - seed) ** 2).sum()
            try:
                idx = distance.argmin()
            except:
                import pdb;pdb.set_trace()
            clusterMap[tobe_classified_index[0], tobe_classified_index[1]] = idx+1
        return clusterMap

    # ---------------------------------
    # Guassian Mixture fitting method
    # 采用另一个流程：
    # 1.先做峰值提取
    # 2.从峰值点出发搜索周围
    # 3.得到每一个分类后拟合高斯参数
    # ---------------------------------
    def guassian_mixture_fit(self, pred_heatmap, thresh=0.8):
        pred_heatmap = pred_heatmap.numpy()
        # 1.先做峰值提取
        cfarMap = self.iou_calculator.staticThresh(pred_heatmap, thresh)
        peakMap, peakIndex = self.iou_calculator.peakPrune(pred_heatmap, cfarMap)
        # 2.从峰值点出发搜索周围，得到每一个分类
        clusterMap = self.seed_clustering(pred_heatmap, peakMap)
        # 3.拟合高斯参数
        # 对每个类别计算加权均值和方差
        cluster_num = np.int(clusterMap.max().item())
        mu_arr = torch.zeros([cluster_num, 2])
        conv_arr = torch.zeros([cluster_num, 2, 2])
        new_gaussian_map = np.zeros(cfarMap.shape)
        # import pdb;pdb.set_trace()
        for i in range(cluster_num):
            mask = (clusterMap == i+1).int()
            weight_sum = (pred_heatmap*mask).sum()
            indexes = torch.nonzero(mask)
            weight = pred_heatmap[indexes[...,0], indexes[...,1]]
            # plt.plot(indexes[..., 1].cpu().detach().numpy(), indexes[..., 0].cpu().detach().numpy(), 'yo')
            # 计算加权均值
            # u=sum(w*x)/sum(w)
            mu = torch.cat((torch.mul(indexes[...,1], weight).view(-1,1), torch.mul(indexes[...,0], weight).view(-1,1)), dim=1)
            mu_arr[i,...] = torch.sum(mu, dim=0)/weight_sum
            # 计算加权协方差
            # conv = sum(w*(x-u).T*(x-u))
            indexes = torch.cat((indexes[...,1].view(-1,1), indexes[...,0].view(-1,1)), dim=1)
            conv = torch.matmul(torch.mul((indexes-mu_arr[i,...]), weight.view(-1,1)).t(), indexes-mu_arr[i,...])
            conv = (conv/weight_sum)
            conv_arr[i,...] = conv
            # ----------------------
            # 新生成的gaussian map看看
            # ----------------------
            new_mu, new_conv = mu_arr[i,...].numpy().reshape((2,1)), conv.numpy()
            try:
                gene_gaussianMap(new_gaussian_map, new_mu, new_conv)
            except:
                continue

        # ---------------------------------
        # 显示feature map 的高斯分类拟合结果
        # # ---------------------------------
        # import pdb;pdb.set_trace()
        ax1 = plt.subplot(2, 3, 1)
        ax1.set_title('prediction heatmap')
        plt.imshow(pred_heatmap.cpu().detach().numpy(), origin='lower')
        ax2 = plt.subplot(2, 3, 2)
        ax2.set_title('CFAR result')
        plt.imshow(cfarMap.cpu().detach().numpy(), origin='lower')
        ax3 = plt.subplot(2, 3, 3)
        ax3.set_title('peak result')
        plt.imshow(peakMap.cpu().detach().numpy(), origin='lower')
        ax3 = plt.subplot(2, 3, 4)
        ax3.set_title('cluster result')
        plt.imshow(clusterMap.cpu().detach().numpy(), origin='lower')
        ax4 = plt.subplot(2, 3, 5)
        ax4.set_title('new gaussian result')
        plt.imshow(new_gaussian_map, origin='lower')
        plt.show()

        # return mu_arr, conv_arr, new_gaussian_map
        return mu_arr, conv_arr, peakMap


class multi_channel_object_decode(object):
    def __init__(self):
        self.peakDetector = peakProcess()
        # 解码gaussian loss结果
        self.gaussian_decoder = gaussian_decoder()

    def gene_objects(self, prediction, scale, opts):
        prediction = torch.squeeze(prediction)
        if opts.cuda:
            prediction = prediction.cpu().detach()
        # 计算bbox
        # prediction: (location, x_offset, y_offset, W, H, orientation[bin1,bin2,offset,obj_score], class[car,background])
        pred_heatmap = prediction[0, :, :]
        pred_heatmap = torch.sigmoid(pred_heatmap)
        #################################################
        # gaussian based fusion loss
        #################################################
        if opts.fusion_loss_arch == 'fusion_loss_gaussian':
            mu_arr, conv_arr, peakMap = self.gaussian_decoder.guassian_mixture_fit(pred_heatmap, thresh=0.5)
            obj_num = mu_arr.shape[0]
            # 定义最终目标矩阵
            # final objects: x,y,p,w,h,angle,p,class:0/1
            final_objects = np.zeros((obj_num, 9))
            final_objects[:, :2] = scale*mu_arr
            # final_objects[:, :2] = mu_arr.cpu().detach().numpy()
            final_objects[:, 2] = pred_heatmap[peakMap.numpy()==1].numpy()
            new_gaussian_map = np.zeros(pred_heatmap.shape)
            for i in range(obj_num):
                # ----------------------
                # 新生成的gaussian map看看
                # ----------------------
                new_mu, new_conv = mu_arr[i,...].numpy().reshape((2,1)), conv_arr[i,...].numpy().T
                try:
                    gene_gaussianMap(new_gaussian_map, new_mu, new_conv)
                except:
                    continue

                conv = np.squeeze(conv_arr[i,...].cpu().detach().numpy())
                eig_vals, eig_vecs = np.linalg.eig(conv)
                eig_vals = scale*eig_vals
                # 以长边为求解角度的方向
                if eig_vals[0] > eig_vals[1]:
                    angle_vec = eig_vecs[:,0]
                    final_objects[i, 3] = eig_vals[1]
                    final_objects[i, 4] = eig_vals[0]
                else:
                    angle_vec = eig_vecs[:, 1]
                    final_objects[i, 3] = eig_vals[0]
                    final_objects[i, 4] = eig_vals[1]
                # 按照四个象限分角度
                #               0
                #               ^
                #               |
                #            2  |  1
                # -90      ----------->   90
                #            3  |  4
                #               |
                #              +/-180
                delta_y = angle_vec[0]
                delta_x = angle_vec[1]
                angle = np.arctan(np.abs(delta_y/delta_x))*180/np.pi
                # 1:
                if delta_y >= 0 and delta_x >= 0:
                    final_objects[i, 5] = angle
                # 2:
                elif delta_y >= 0 and delta_x < 0:
                    # final_objects[i, 5] = 180 - angle
                    final_objects[i, 5] = -angle
                # 3:
                elif delta_y < 0 and delta_x < 0:
                    final_objects[i, 5] = -180 + angle
                # 4:
                elif delta_y < 0 and delta_x >= 0:
                    final_objects[i, 5] = 180 - angle
            # 目前没有角度置信度和类别
            # import pdb;pdb.set_trace()
            return final_objects, peakMap, new_gaussian_map

        #################################################
        # multi channel fusion loss
        #################################################
        # -----------------------------
        # 生成中心点位置
        # 通过cfar算法从keymap里找到峰值后，再用峰值修剪算法剩下峰值中心点位置，然后从这个关键点位置计算bbox
        # -----------------------------
        if self.peakDetector.cfar_type == 'staticThresh':
            cfarMap = self.peakDetector.staticThresh(pred_heatmap, opts.heatmap_thresh)
        else:
            cfarMap = self.peakDetector.cfar2d(pred_heatmap)
        peakMap, peakIndex = self.peakDetector.peakPrune(pred_heatmap, cfarMap)
        # 根据peak mask生成对应的bounding box
        peakMask = peakMap == 1
        pred_objects = prediction.numpy()[..., peakMask]
        pred_objects = torch.from_numpy(pred_objects)
        # 定义最终目标矩阵
        # final objects: x,y,p,w,h,angle,p,class:0/1
        final_objects = np.zeros((pred_objects.shape[1], 9))
        # 生成目标的位置,x,y,p
        pred_location_x = scale * peakIndex[0,:]
        pred_location_y = scale * peakIndex[1,:]
        # 中心点位置
        pred_offset_x = torch.sigmoid(pred_objects[1, ...]) * scale
        pred_offset_y = torch.sigmoid(pred_objects[2, ...]) * scale
        pred_offset_x = pred_offset_x.numpy()
        pred_offset_y = pred_offset_y.numpy()
        final_objects[..., 0] = pred_location_x + pred_offset_x
        final_objects[..., 1] = pred_location_y + pred_offset_y
        final_objects[..., 2] = torch.sigmoid(pred_objects[0, ...]).numpy().T
        # -----------------------------
        # 生成宽高
        # -----------------------------
        final_objects[..., 3:5] = pred_objects[3:5, ...].numpy().T
        # -----------------------------
        # 生成angle
        # opts fusion_loss_arch:
        #   normal_l1_loss
        #   rotated_iou_loss
        #   multi_angle_bin_loss
        # -----------------------------
        if opts.fusion_loss_arch == 'fusion_loss_base':
            # 2 bin loss
            angleBins = torch.sigmoid(pred_objects[5,...]).numpy()
            angleOffset = 180*torch.sigmoid(pred_objects[6, ...]).numpy()
            for i in range(len(angleBins)):
                # mvdnet angle loss
                if opts.fusion_loss_arch == 'fusion_loss_base':
                    if angleBins[i] > 0.5:
                        final_objects[i, 5] = angleOffset[i]
                        final_objects[i, 6] = angleBins[i]
                    else:
                        final_objects[i, 5] = -angleOffset[i]
                        final_objects[i, 6] = 1-angleBins[i]
        # multi bin loss
        elif opts.fusion_loss_arch == 'multi_angle_bin_loss':
            angleResolution = 180 / (2**opts.binNum)
            angleBins = torch.sigmoid(pred_objects[5:5+opts.binNum, ...])
            angleBins_binary = (angleBins > 0.5).int()
            angleBins_binary = angleBins_binary.numpy()
            angleOffset = angleResolution*(2*torch.sigmoid(pred_objects[5+opts.binNum, ...]).numpy()-1)
            final_objects[..., 5] = np.squeeze(multi_bin_angle_decode(angleBins_binary, angleOffset))
            # 2022-03-22: 用每个bin距离1的平均置信度来表示角度置信度，暂时
            final_objects[..., 6] = np.sum(angleBins.numpy(), axis=0)/opts.binNum
        # -----------------------------
        # 生成class
        # -----------------------------
        final_objects[...,7] = torch.sigmoid(pred_objects[7,...]).numpy().T
        # final objects: x,y,p,w,h,angle,p,class:0/1
        return final_objects, peakMap,

    def gene_objects_anchor_angle(self, prediction, scale, opts, anchor_angle_list=[0,45,90,135]):
        prediction = torch.squeeze(prediction)
        if opts.cuda:
            prediction = prediction.cpu().detach()
        anchor_num = len(anchor_angle_list)
        final_objects_list = []
        for num in range(anchor_num):
            anchor_angle = anchor_angle_list[num]
            # 计算bbox
            # prediction: (location, x_offset, y_offset, W, H, orientation[bin1,angle_offset], class)*anchor_num = 8*4[0,45,90,135]
            pred_heatmap = prediction[0+8*num, :, :]
            pred_heatmap = torch.sigmoid(pred_heatmap)
            #################################################
            # multi channel fusion loss
            #################################################
            # -----------------------------
            # 生成中心点位置
            # 通过cfar算法从keymap里找到峰值后，再用峰值修剪算法剩下峰值中心点位置，然后从这个关键点位置计算bbox
            # -----------------------------
            if self.peakDetector.cfar_type == 'staticThresh':
                cfarMap = self.peakDetector.staticThresh(pred_heatmap, self.peakDetector.thresh)
            else:
                cfarMap = self.peakDetector.cfar2d(pred_heatmap)
            peakMap, peakIndex = self.peakDetector.peakPrune(pred_heatmap, cfarMap)
            # 根据peak mask生成对应的bounding box
            peakMask = peakMap == 1
            pred_objects = prediction.numpy()[..., peakMask]
            pred_objects = torch.from_numpy(pred_objects)
            # 定义最终目标矩阵
            # final objects: x,y,p,w,h,angle,p,class:0/1
            final_objects = np.zeros((pred_objects.shape[1], 9))
            # 生成目标的位置,x,y,p
            pred_location_x = scale * peakIndex[0,:]
            pred_location_y = scale * peakIndex[1,:]
            # 中心点位置
            pred_offset_x = torch.sigmoid(pred_objects[1+8*num, ...]) * scale
            pred_offset_y = torch.sigmoid(pred_objects[2+8*num, ...]) * scale
            pred_offset_x = pred_offset_x.numpy()
            pred_offset_y = pred_offset_y.numpy()
            final_objects[..., 0] = pred_location_x + pred_offset_x
            final_objects[..., 1] = pred_location_y + pred_offset_y
            # 用class分数作为指标
            final_objects[..., 2] = torch.sigmoid(pred_objects[7+8*num, ...]).numpy().T
            # -----------------------------
            # 生成宽高
            # -----------------------------
            final_objects[..., 3:5] = pred_objects[3+8*num:5+8*num, ...].numpy().T
            # -----------------------------
            # 生成angle
            # opts fusion_loss_arch:
            #   normal_l1_loss
            #   rotated_iou_loss
            #   multi_angle_bin_loss
            # -----------------------------
            # 2 bin loss
            angleBins = torch.sigmoid(pred_objects[5+8*num,...]).numpy()
            angleOffset = 180*torch.sigmoid(pred_objects[6+8*num, ...]).numpy()
            for i in range(len(angleBins)):
                # mvdnet angle loss
                if opts.fusion_loss_arch == 'fusion_loss_base':
                    if angleBins[i] > 0.5:
                        final_objects[i, 5] = angleOffset[i]
                        final_objects[i, 6] = angleBins[i]
                    else:
                        final_objects[i, 5] = -angleOffset[i]
                        final_objects[i, 6] = angleBins[i]
                elif opts.fusion_loss_arch == 'anchor_angle_loss':
                    if angleBins[i] > 0.5:
                        final_objects[i, 5] = angleOffset[i]+anchor_angle
                        final_objects[i, 6] = angleBins[i]
                    else:
                        final_objects[i, 5] = -(angleOffset[i]+anchor_angle)
                        final_objects[i, 6] = angleBins[i]
                else:
                    if angleBins[i] > 0.5:
                        final_objects[i, 5] = -90 + angleOffset[i]
                        final_objects[i, 6] = angleBins[i]
                    else:
                        final_objects[i, 5] = 90 + angleOffset[i]
                        final_objects[i, 6] = angleBins[i]
            # -----------------------------
            # 生成class
            # -----------------------------
            final_objects[...,7:9] = torch.sigmoid(pred_objects[7,...]).numpy().T
            final_objects_list.append(final_objects)
        final_objects_array = np.concatenate(final_objects_list)
        # final objects: x,y,p,w,h,angle,p,class
        return final_objects_array, peakMap,

    def non_max_suppression(self, predictions, raw_size, opts, nms_thresh=0.2):
        '''
        :param predictions: prediction map batch_size=1*channel*width*height
        :param scale:       feature map scale
        :return:
        '''

        [raw_width, raw_height] = raw_size
        multiScale_detection_objects = np.zeros((2200*4,9))
        multiScale_number_objects = 0
        # predictions: 3*list[batch_size=1*channel*width*height]
        cnt = 0
        for pred_map in predictions:
            pred_map = torch.squeeze(pred_map)
            [_, width, height] = pred_map.shape
            scale = raw_width/width
            cnt += 1
            if opts.fusion_loss_arch == 'fusion_loss_base':
                detections_objects, peakMap = self.gene_objects(pred_map, scale, opts)
                if cnt == 1:
                    first_scale_peakMap = peakMap
            elif opts.fusion_loss_arch == 'anchor_angle_loss':
                detections_objects, peakMap = self.gene_objects_anchor_angle(pred_map, scale, opts)
            number_objects = detections_objects.shape[0]
            multiScale_detection_objects[multiScale_number_objects:multiScale_number_objects+number_objects, :] = detections_objects
            multiScale_number_objects += number_objects
            # if detections_objects.size == 0:
            #     return detections_objects, peakMap
            # import pdb;pdb.set_trace()
            # break
        multiScale_detection_objects = multiScale_detection_objects[:multiScale_number_objects, :]
        # 按照存在物体的置信度排序
        multiScale_detection_objects = torch.from_numpy(multiScale_detection_objects)
        _, conf_sort_index = torch.sort(multiScale_detection_objects[:, 2], descending=True)
        detections_objects = multiScale_detection_objects[conf_sort_index]
        #########################
        # 进行非极大抑制
        #########################
        max_detections = []
        cnt = 0
        while detections_objects.size(0):
            # print("boxes number: {}".format(detections_objects.size(0)))
            ious_array = np.zeros((detections_objects.size(0)-1))
            # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thresh，如果是则去除掉
            max_detections.append(detections_objects[0].unsqueeze(0))
            if len(detections_objects) == 1:
                break
            number_objects = detections_objects.shape[0]
            for i in range(number_objects-1):
                other_detection_boxes = np.zeros((number_objects, 6))
                other_detection_boxes[...,1:3] = detections_objects[...,0:2]
                other_detection_boxes[...,3:6] = detections_objects[...,3:6]
                ious_array[i], _ = calc_iou(torch.squeeze(max_detections[-1]), other_detection_boxes[i+1,...])
            detections_objects = detections_objects[1:,...]
            detections_objects = detections_objects[ious_array < nms_thresh, ...]
        # 堆叠
        if len(max_detections) == 0:
            return torch.empty(0,9), first_scale_peakMap
        max_detections = torch.cat(max_detections).data
        return max_detections.numpy(), first_scale_peakMap


class calc_mAP(object):
    def __init__(self):
        pass

    def savePredResult(self, final_objects, out_path, timestamp):
        # define result path
        f = open(os.path.join(out_path, "detection-results/" + timestamp + ".txt"), "w")
        f.write("x y p w h angle p class:0/1\n")
        for i in range(final_objects.shape[0]):
            try:
                object = final_objects[i,...]
                # import pdb;pdb.set_trace()
                f.write("%s %s %s %s %s %s %s %s %s\n" % (
                str(object[0]), str(object[1]), str(object[2]), str(object[3]), str(object[4]), str(object[5]),
                str(object[6]), str(object[7]), str(object[8])))
            except:
                import pdb;pdb.set_trace()
        f.close()

    def boxesMatch(self, pred_boxes, gt_boxes, iou_thresh, pr_table_flag=True, calc_diou_flag=True):
        '''
        :param pred_boxes:  N*8 - N*[x,y,p,w,h,angle,p,class:0/1]
        :param gt_boxes:    N*6 - N*[id,x,y,w,h,angle]
        :return:
            TP: IoU>0.5的检测框数量（同一Ground Truth只计算一次）
            FP: IoU<=0.5的检测框，或者是检测到同一个GT的多余检测框的数量
            FN: 没有检测到的GT的数量
        pred_array = ((iou>0.5)+(iou<=0.5))*(confidence, iou 0/1)
        residual_pred = (number pred - number gt)
        miss_gt = (number gt - number pred)
        '''
        num_preds = pred_boxes.shape[0]
        num_gts = gt_boxes.shape[0]
        # calculate cost matrix
        costMat = np.zeros((num_preds, num_gts))
        for i in range(num_preds):
            for j in range(num_gts):
                try:
                    if calc_diou_flag:
                        costMat[i, j] = calc_diou(pred_boxes[i, ...], gt_boxes[j, ...]).detach().numpy()
                    else:
                        costMat[i, j], _ = calc_iou(pred_boxes[i, ...], gt_boxes[j, ...])
                except:
                    import pdb;pdb.set_trace()
        ############################################
        # 这个函数是以行列更小的数值为标准，更小的数值是多少，输出的匹配矩阵就是多少
        # 例如输入12×13(13*12)大小的costMat，输出就是12×2的matches
        try:
            if calc_diou_flag:
                # 计算diou时需要匹配最低diou的对象, 代表两个框更靠近
                matches = linear_sum_assignment(costMat)
            else:
                # 计算iou时候需要匹配最高iou的对象, 代表两个框更靠近
                matches = linear_sum_assignment(1-costMat)
        except:
            print("="*20+" nan shows up !"+"="*20)
            print(pred_boxes)
            return np.array([]), 0
        ############################################
        len_matches = matches[0].shape[0]
        min_iter = np.min([num_preds, num_gts])
        pr_table = np.zeros((num_preds, 2))
        try:
            pr_table[:len_matches,0] = pred_boxes[matches[0], 2]
            pr_table[:len_matches,1] = costMat[matches]
            # 把重复检测的结果存在后面
            pr_table[len_matches:, 0] = pred_boxes[len_matches:, 2]
            pr_table[len_matches:, 1] = np.zeros((1,num_preds-len_matches))
        except:
            import pdb;pdb.set_trace()
        if pr_table_flag:
            pr_table[:, 1] = pr_table[:, 1] > iou_thresh
            # pr_table = pr_table[mask]
        num_fn = 0
        num_pred_tp = pr_table.shape[0]
        if num_pred_tp < num_gts:
            num_fn = num_gts - num_pred_tp
        # import pdb;pdb.set_trace()
        return pr_table, num_fn

    '''
    precision-recall table
     prediction | GT | confidence | T/F(IOU>?) | precision | recall
    ----------------------------------------------------------------
                |    |            |            |           |

    mAP: mean Average Precision, 即各类别AP的平均值
    AP: PR曲线下面积
    PR曲线: Precision-Recall曲线
    Precision: TP / (TP + FP)
    Recall: TP / (TP + FN)
    TP: IoU>0.5的检测框数量（同一Ground Truth只计算一次）
    FP: IoU<=0.5的检测框，或者是检测到同一个GT的多余检测框的数量
    FN: 没有检测到的GT的数量
    :return: precision-recall table
    '''
    def calc_pr_table(self, pred_boxes, gt_boxes, iou_thresh=0.5, pr_table_flag=True, calc_diou_flag=True):
        '''
        :param pred_boxes:      输入预测框
        :param gt_boxes:        输入真值框
        :param iou_thresh:      输入iou阈值，默认为0.5,为了计算pr table
        :param pr_table_flag:   是否通过iou thresh计算pr table
        :param calc_diou:       是否使用diou loss，训练时采用diou loss，评价时采用iou，另外，diou loss可能大于1
        :return:
        '''
        # 匹配检测目标与gt目标
        if len(pred_boxes)==0 and len(gt_boxes) == 0:
            return np.array([]),0
        # gt_boxes = gt_boxes[3:]
        pr_table, num_fn = self.boxesMatch(pred_boxes, gt_boxes, iou_thresh, pr_table_flag, calc_diou_flag)
        # pr_table 包含TP/FP两种检测结果，是所有的检测结果集合
        return pr_table, num_fn

    def calc_pr_curve(self, pred_table, all_gt):
        '''
        :param pred_table:      num_pred*[conf, T/F]
        :param all_gt:          all gt number
        :return:
        '''
        # 排序
        sorted_arr = np.sort(pred_table, axis=0)
        truePositive_arr = sorted_arr[:,1]
        num = pred_table.shape[0]
        # 计算pr曲线
        p = np.zeros(num)
        r = np.zeros(num)
        # 按置信度高低进行索引
        for i in range(num):
            positive_arr = truePositive_arr[i:] == 1
            tp = positive_arr.sum()   # 置信度符合并且IOU>0.5
            p[i] = (tp+0.0)/(num-i+1e-6) # 所有的检测中是真的比例
            r[i] = (tp+0.0)/(all_gt+1e-6) # 所有的真值中检测到了的比例
        return p ,r

    def save_pr(self, out_path, p, r, mAPtype='norm0.5'):
        pr_curve = np.asarray([p,r]).T
        filename = os.path.join(out_path, "pr-curve/" + mAPtype + ".txt")
        np.savetxt(filename, pr_curve, fmt='%.4f')

    # calc_ap
    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        # 倒序
        rec = rec[::-1] # 0-1
        prec = prec[::-1] # 1-0
        if use_07_metric:  #VOC在2010之后换了评价方法，所以决定是否用07年的
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):  #  07年的采用11个点平分recall来计算
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])  # 取一个recall阈值之后最大的precision
                ap = ap + p / 11.  # 将11个precision加和平均
        else:  # 这里是用2010年后的方法，取所有不同的recall对应的点处的精度值做平均，不再是固定的11个点
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))  #recall和precision前后分别加了一个值，因为recall最后是1，所以
            mpre = np.concatenate(([1.], prec, [0.])) # 右边加了1，precision加的是0
            # mrec = np.concatenate(([1.], rec, [0.]))
            # mpre = np.concatenate(([0.], prec, [1.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  #从后往前，排除之前局部增加的precison情况
            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]  # 这里巧妙的错位，返回刚好TP的位置，
                                                                                          # 可以看后面辅助的例子

            # and sum (\Delta recall) * prec   用recall的间隔对精度作加权平均
            # plt.subplot(2, 2, 1)
            # plt.plot(i)
            # plt.subplot(2, 2, 2)
            # plt.plot(mrec[i + 1] - mrec[i])
            # plt.subplot(2, 2, 3)
            # plt.plot(mpre[i+1])
            # plt.subplot(2, 2, 4)
            # plt.plot((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            # plt.show()
            # import pdb;pdb.set_trace()
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


if __name__=="__main__":
    N = 8
    pfa = np.arange(1e-5,1e-1,1e-5)

    alpha = N*(np.power(pfa, -1/N) - 1)
    alpha2 = np.power(pfa, -1/N) - 1
    plt.figure()
    plt.plot(pfa, alpha,color='r')
    plt.plot(pfa, alpha2,color='b')
    plt.show()