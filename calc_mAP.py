# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from model.lidar_and_radar_fusion_model import lidar_and_radar_fusion
import torch
from evaluator.visualize import showPredResult
from evaluator.utils_mAP import calc_mAP
from data.datasets.oxford_dataset.oxford_dataloader import *
import matplotlib.pyplot as plt
import pdb
from utils.opts import opts

inputDataPath = '/scratch/project/cpautodriving/yanlongyang/2019-01-10-11-46-21-radar-oxford-10k/processed/'


def geneModel_old(model_path, opts):
    # initialize model
    model = lidar_and_radar_fusion(opts)
    device = torch.device('cuda' if opts.cuda else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model    = model.eval()
    print('{} model loaded.'.format(model_path))
    if opts.cuda:
        # model = nn.DataParallel(model)
        model = model.cuda()
    return model

def geneModel(model_path, opts):
    model = lidar_and_radar_fusion(opts)
    model_dict = model.state_dict()
    stat = torch.load(model_path)
    pretrained_dict = stat['model']
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if opts.from_distributed:
            new_key = k[7:]
        else:
            new_key = k
        try:
            if np.shape(model_dict[new_key]) == np.shape(v):
                temp_dict[new_key] = v
                load_key.append(new_key)
            else:
                no_load_key.append(k)
        except:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    return model.cuda()

def main(opts):
    print(opts)
    # ----------------------------------------
    # 基础参数
    # ----------------------------------------
    # 0: 全部过程
    # 1: 读检测结果
    # 2: 读pr曲线
    procedure = 0
    iou_thresh = 0.5
    # ----------------------------------------
    # 读取网络模型
    # ----------------------------------------
    model_path = os.path.join(opts.map_out, opts.exp_name, opts.model_path)
    model = geneModel(model_path, opts)
    # ----------------------------------------
    # 读取测试数据
    # ----------------------------------------
    test_annotation_path = opts.test_data
    with open(test_annotation_path) as f:
        test_lines = f.readlines()

    dataset = OxfordDataset(test_lines, opts)

    # ----------------------------------------
    # 创建指标输出文件夹
    # ----------------------------------------
    path = opts.output_dir
    exp_name = opts.experiment_name
    map_out_path = os.path.join(path, exp_name)
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)

    # --------------------
    # 使用pycocotools
    # --------------------
    if opts.mAP_type == "pycoco":
        from evaluator.MVDNet_mAPtools import calc_mvdnet_mAP
        calc_mvdnet_mAP(model, test_lines, dataset, opts, map_out_path)
        return

    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'pr-curve')):
        os.makedirs(os.path.join(map_out_path, 'pr-curve'))

    mAP_calculator = calc_mAP()
    all_pred_array = np.zeros((200000,2))
    all_pred = 0
    all_gt = 0
    all_fn = 0
    all_pred_coco_array = np.zeros((10, 200000, 2))
    if procedure != 2:
        cnt = 0
        for annotation_line in test_lines:
            line = annotation_line.split()
            frame_num = line[0]
            timestamp = line[1]

            # read lidar pc
            # get model input data
            [pillars, lidarData, radar_data, gt_boxes, class_arr] = dataset.get_data(annotation_line)
            if procedure==0:
                pillars = np.expand_dims(pillars, 0)
                radar_data = np.expand_dims(radar_data, 0)
                radar_data = np.expand_dims(radar_data, 0)
                radar_data = radar_data.astype(np.float32)
                # get model prediction result
                with torch.no_grad():
                    if opts.cuda:
                        pillars = torch.from_numpy(pillars).type(torch.FloatTensor).cuda()
                        radar_data = torch.from_numpy(radar_data).type(torch.FloatTensor).cuda()
                        gt_boxes = torch.from_numpy(gt_boxes).type(torch.FloatTensor).cuda()
                    predictions = model(pillars, radar_data, opts)
                    # transfer radar back to cpu
                    if opts.cuda:
                        radar_data = radar_data.cpu().detach().numpy()
                    radar_data = np.squeeze(radar_data)
                    pred_boxes = showPredResult(lidarData, radar_data, gt_boxes, predictions, 0.2, opts)
                mAP_calculator.savePredResult(pred_boxes, map_out_path, timestamp)
            if procedure==1:
                prediction_filename = os.path.join(map_out_path, "detection-results/" + timestamp + ".txt")
                with open(prediction_filename) as f:
                    pred_lines = f.readlines()

                skip = 0
                pred_boxes = np.zeros((len(pred_lines)-1, 9))
                cnt = 0
                for pred_line in pred_lines:
                    if skip == 0:
                        skip += 1
                        continue
                    line_ = pred_line.split()
                    # import pdb;pdb.set_trace()
                    # x y p w h angle p class:0/1
                    pred_boxes[cnt, ...] = np.array([float(line_[0]), float(line_[1]), float(line_[2]),
                                                     float(line_[3]), float(line_[4]), float(line_[5]),
                                                     float(line_[6]), float(line_[7]), float(line_[8])])
                    cnt += 1
            if opts.mAP_type == 'norm0.5':
                pr_table, num_fn = mAP_calculator.calc_pr_table(pred_boxes, gt_boxes, iou_thresh, calc_diou_flag=False)
                num_pred = pr_table.shape[0]
                if num_pred != 0:
                    all_pred_array[all_pred:all_pred + num_pred, ...] = pr_table
                    all_pred += num_pred
                    this_tp = pr_table[:,1].sum()
                all_gt += gt_boxes.shape[0]
                all_fn += num_fn
                print('all detection boxes: {}'.format(all_pred))
                print('all true detection boxes: {}'.format(this_tp))
                print('all ground truth boxes: {}'.format(all_gt))
                print('all miss detection(fn): {}'.format(all_fn))
                # import pdb;pdb.set_trace()
                plt.pause(0.0001)
                plt.clf()
            # 用coco指标
            elif opts.mAP_type == 'my_coco':
                iou_thresh_coco = range(50, 95, 5)
                for iou_thresh in iou_thresh_coco:
                    iou_thresh  = iou_thresh/100
                    # pr_table, num_fn = mAP_calculator.calc_pr_table(pred_boxes, gt_boxes, iou_thresh)
                    pr_table, num_fn = mAP_calculator.calc_pr_table(pred_boxes, gt_boxes, iou_thresh, calc_diou_flag=False)
                    num_pred = pr_table.shape[0]
                    if num_pred != 0:
                        # import pdb;pdb.set_trace()
                        all_pred_array[all_pred:all_pred + num_pred, ...] = pr_table
                        all_pred += num_pred

                    all_gt += gt_boxes.shape[0]
                    all_fn += num_fn
                    print('*'*20+'iou thresh: {}'.format(iou_thresh)+'*'*20)
                    print('all detection boxes: {}'.format(all_pred))
                    print('all ground truth boxes: {}'.format(all_gt))
                    print('all miss detection(fn): {}'.format(all_fn))
        all_pred_array = all_pred_array[:all_pred, ...]
        p, r = mAP_calculator.calc_pr_curve(all_pred_array, all_gt)
        mAP_calculator.save_pr(map_out_path, p ,r, opts.mAP_type)

    if procedure == 2:
        pr_curve_filename = os.path.join(map_out_path, "pr-curve/" + "pr_curve.txt")
        pr = np.loadtxt(pr_curve_filename)
        p = pr[:,0]
        r = pr[:,1]
        pass
    ap = mAP_calculator.voc_ap(r, p)
    print('mAP: {}'.format(ap))
    r = np.concatenate(([1.], r, [0.]))
    p = np.concatenate(([0.], p, [1.]))
    try:
        plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ax1 = plt.subplot(1, 3, 1)
        ax1.set_title('precision curve')
        plt.plot(p)
        ax2 = plt.subplot(1, 3, 2)
        ax2.set_title('recall curve')
        plt.plot(r)
        ax3 = plt.subplot(1, 3, 3)
        ax3.set_title('p-r curve')
        # import pdb;pdb.set_trace()
        plt.plot(r, p)
        fig_filename = os.path.join(map_out_path, "pr-curve/pr_curve_" + str(ap) + "_.png")
        plt.savefig(fig_filename, bbox_inches='tight')
    except:
        pdb.set_trace()
    plt.show()


if __name__ == "__main__":
    opt = opts().parse()
    main(opt)