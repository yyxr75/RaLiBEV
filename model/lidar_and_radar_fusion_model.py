# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from model.backbone import yolo
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def saveSingleMap(map, fname):
    plt.figure(2)
    plt.clf()
    plt.imshow(map, cmap='jet', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(fname, dpi=600)

def save4Maps(map1, map2, map3, map4, fname):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(map1, cmap='jet', interpolation='nearest')
    axs[0, 0].set_title("Query Weight")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(map2, cmap='jet', interpolation='nearest')
    axs[0, 1].set_title("Input Value")
    axs[0, 1].axis('off')
    axs[1, 0].imshow(map3, cmap='jet', interpolation='nearest')
    axs[1, 0].set_title("Output Value")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(map4, cmap='jet', interpolation='nearest')
    axs[1, 1].set_title("Head Feature")
    axs[1, 1].axis('off')
    fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=axs.ravel().tolist(), location='right')

    plt.savefig(fname, dpi=600)

def normalizeMap(map):
    map = map.reshape(1,-1,320,320)
    map = torch.sum(torch.abs(map), dim=1, keepdim=True)
    map = torch.squeeze(map)

    map_norm = (map-map.min())/(map.max()-map.min())
    map_sigmoid = torch.sigmoid(map_norm)
    map_numpy_normalized = map_sigmoid.detach().cpu().numpy()


    return map_numpy_normalized

def attention_base(q, k, v):
    bs,c,w,h = q.shape
    weight_mat = torch.matmul(q,k.permute(0,1,3,2))
    weight_mat = weight_mat.reshape(bs,c,w*h)
    weight_mat = torch.softmax(weight_mat,dim=2)
    weight_mat = weight_mat.reshape(bs, c, w, h)
    output = weight_mat.mul(v)+v

    return output


def attention_variable_query(variable_query, k, v):
    bs,c,w,h = k.shape
    q_num, q_dim = variable_query.shape
    variable_query = variable_query.reshape(1, w, h, q_dim).repeat(bs, 1, 1, 1).permute(0,3,1,2)
    return attention_base(variable_query, k ,v)

def deformable_attention(q, v):
    pass

def conv2d(filter_in, filter_out, kernel_size=3, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.LeakyReLU(0.1),
    )


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.in_channels = in_channels
    
        self.linear= nn.Linear(self.in_channels, self.units, bias = False)
        self.norm = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.01)

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.units, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=1, stride=1)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 12), stride=(1, 1), dilation=(1,3))

    def forward(self, input):
        x = self.conv1(input)
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


class lidar_psedo_image_generate(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.PFNLayer = PFNLayer(in_channels,out_channels,True,True)

    def forward(self, x):
        x = self.PFNLayer(x)
        x = x.squeeze(3)
        return x


class radarBranch(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super(radarBranch, self).__init__()
        self.downSampling = nn.Sequential(conv2d(in_channel, 2, 3, stride=1),
                                                conv2d(2, out_channel, 1, stride=1))

    def forward(self,x):
        x = self.downSampling(x)
        return x


class fusionHead(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super(fusionHead, self).__init__()
        self.head = nn.Sequential(conv2d(in_channel,in_channel),
                                  nn.Conv2d(in_channel, out_channel, 1))

    def forward(self, x):
        x = self.head(x)
        return x


class lidar_and_radar_fusion(nn.Module):
    def __init__(self, opts):
        super().__init__()
        if opts.modality=='fusion':

            self.radar_channel = 16
            if opts.fusion_arch == 'feature_level_concatenation' or opts.fusion_arch == 'last_stage_feature_level_concatenation':
                self.radar_channel = 4
            self.radar_branch = radarBranch(1,self.radar_channel)
            self.lidar_channel = 16
            self.lidar_branch = lidar_psedo_image_generate(9, self.lidar_channel, True, True)
            if opts.fusion_arch == 'feature_level_concatenation':
                self.yolo_precat = yolo.YoloDecoder(self.lidar_channel + self.radar_channel)
            elif opts.fusion_arch == 'Direct_Interactive_Transformer_for_BEV_Fusion':
                self.yolo_precat = yolo.YoloDecoder(self.lidar_channel)
            elif opts.fusion_arch == 'Dense_Query_Map-based_Interactive_Transformer_for_BEV_Fusion_(DQMITBF)':
                self.query_embed = nn.Embedding(opts.data_shape[0]*opts.data_shape[1], self.radar_channel)
                self.yolo_precat = yolo.YoloDecoder(self.lidar_channel+self.radar_channel)
        elif opts.modality=='lidar-only':
            self.lidar_channel = 16
            self.lidar_branch = lidar_psedo_image_generate(9, self.lidar_channel, True, True)
            self.yolo_precat = yolo.YoloDecoder(self.lidar_channel)
        elif opts.modality=='radar-only':
            self.radar_channel = 16
            if opts.fusion_arch == 'feature_level_concatenation' or opts.fusion_arch == 'last_stage_feature_level_concatenation':
                self.radar_channel = 4
            self.radar_branch = radarBranch(1,self.radar_channel)
            self.yolo_precat = yolo.YoloDecoder(self.radar_channel)

        self.head3 = fusionHead(128, 8)
        self.head4 = fusionHead(256, 8)
        self.head5 = fusionHead(512, 8)


    def forward(self, pillars=None, radar_image=None, opts=None):
        if opts.modality=='fusion':
            pillars = pillars.permute(0,3,1,2)
            [bs, channels, pillar_num, points] = pillars.shape
            [_,_,width,height] = radar_image.shape
            newPillar = self.lidar_branch(pillars)

            pillarImg = newPillar.reshape((bs, self.lidar_channel, width, height))
            pillarImg = pillarImg.permute(0,1,2,3)
            # -----------------------------
            # # radar data process
            # -----------------------------
            radardata = self.radar_branch(radar_image)
            if opts.fusion_arch == "feature_level_concatenation":
                fusion_data = torch.cat([pillarImg,radardata],dim=1)
                p3,p4,p5 = self.yolo_precat(fusion_data)
            # radar as q, lidar as k,v
            elif opts.fusion_arch == 'Direct_Interactive_Transformer_for_BEV_Fusion':
                fusion_data = attention_base(radardata, pillarImg, pillarImg)
                p3, p4, p5 = self.yolo_precat(fusion_data)
            elif opts.fusion_arch == 'Dense_Query_Map-based_Interactive_Transformer_for_BEV_Fusion_(DQMITBF)':
                dtype = radardata.dtype
                query = self.query_embed.weight.to(dtype)
                fusion_data_1 = attention_variable_query(query, radardata, pillarImg)
                fusion_data_2 = attention_variable_query(query, pillarImg, radardata)
                fusion_data = torch.cat([fusion_data_1,fusion_data_2],dim=1)
                p3, p4, p5 = self.yolo_precat(fusion_data)
        elif opts.modality=='lidar-only':
            pillars = pillars.permute(0,3,1,2)
            [bs, channels, pillar_num, points] = pillars.shape
            newPillar = self.lidar_branch(pillars)

            pillarImg = newPillar.reshape((bs, self.lidar_channel, 320, 320))
            pillarImg = pillarImg.permute(0,1,2,3)
            p3, p4, p5 = self.yolo_precat(pillarImg)
        elif opts.modality=='radar-only':
            radardata = self.radar_branch(radar_image)
            p3, p4, p5 = self.yolo_precat(radardata)
        out3 = self.head3(p3)
        out4 = self.head4(p4)
        out5 = self.head5(p5)
        return out3, out4, out5
