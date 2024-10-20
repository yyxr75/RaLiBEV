# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from data.datasets.oxford_dataset.oxford_dataloader import OxfordDataset
from model.lidar_and_radar_fusion_model import lidar_and_radar_fusion
import torch
from torch import nn
from evaluator.visualize import showPredResult
import numpy as np
from utils.opts import opts

inputDataPath = '/scratch/project/cpautodriving/yanlongyang/RaLiBEV/2019-01-10-11-46-21-radar-oxford-10k/processed/'


def geneModel(model_path, opts):
    # initialize model
    model = lidar_and_radar_fusion(opts)
    device = torch.device('cuda' if opts.cuda else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    if 'epoch' in pretrained_dict.keys():
        pretrained_dict = pretrained_dict['model']
        new_state_dict = {}
        for k,v in pretrained_dict.items():
            new_key = k[7:]
            new_state_dict[new_key] = v
        pretrained_dict = new_state_dict
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.eval()
    print('{} model loaded.'.format(model_path))
    if opts.cuda:
        model = nn.DataParallel(model)
        model = model.cuda()
    return model


def main(opts):
    # generate model
    model_path = opts.model_path
    model = geneModel(model_path, opts)

    nms_val = 0.2
    test_annotation_path = 'data/datasets/oxford_dataset/dataset_filename/test.txt'
    with open(test_annotation_path) as f:
        test_lines = f.readlines()

    dataset = OxfordDataset(test_lines, opts)

    for annotation_line in test_lines:
        # get model input data
        [pillars, lidarData, radar_data, box, class_arr] = dataset.get_data(annotation_line)
        pillars = np.expand_dims(pillars, 0)
        radar_data = np.expand_dims(radar_data, 0)
        radar_data = np.expand_dims(radar_data, 0)
        radar_data = radar_data.astype(np.float32)
        # get model prediction result
        with torch.no_grad():
            if opts.cuda:
                pillars = torch.from_numpy(pillars).type(torch.FloatTensor).cuda()
                radar_data = torch.from_numpy(radar_data).type(torch.FloatTensor).cuda()
                box = torch.from_numpy(box).type(torch.FloatTensor).cuda()
            predictions = model(pillars, radar_data, opts)
            if opts.cuda:
                radar_data = radar_data.cpu().detach().numpy()
            radar_data = np.squeeze(radar_data)
            showPredResult(lidarData, radar_data, box, predictions, nms_val, opts)

        timestamp = annotation_line.split()[1]
        save_path = f'output/prediction_{timestamp}.png'
        plt.savefig(save_path)
        print(f"Prediction result saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    opt = opts().parse()
    main(opt)