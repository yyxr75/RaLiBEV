# RaLiBEV: Radar and LiDAR BEV Fusion Learning for Anchor Box Free Object Detection Systems
Paper Link: [arxiv](https://arxiv.org/abs/2211.06108)

> Yanlong Yang, Jianan Liu, Tao Huang, Qing-long Han, Gang Ma, and Bing Zhu


This is the official code base of RaLiBEV. In this novel work, we propose a bird's-eye view fusion learning-based anchor box-free object detection system. Our approach introduces a novel interactive transformer module for enhanced feature fusion and an advanced label assignment strategy for more consistent regression, addressing key limitations in existing methods. Specifically, experiments show that, our approach's average precision ranks $1^{st}$ and significantly outperforms the state-of-the-art method by 13.1\% and 19.0\% at Intersection of Union (IoU) of 0.8 under Clear+Foggy training conditions for Clear and Foggy testing, respectively. 


## Highlights

![image](https://github.com/user-attachments/assets/d9ba21f7-5755-456a-b656-e1d67931eede)
Fig.1 Visualization of Object Detection Results.


![image](https://github.com/user-attachments/assets/a100916e-fa22-4581-ae45-1c808423a798)
Table 1 Performance Comparison Between Different Methods.


## Install

### Prerequisites
- Python 3.8

- Pytorch 2.0.0

- Pycocotools 2.0.4

- Detectron2 0.4

<!-- to install detectron2 for using [RotatedCOCOeval](evaluator/MVDNet_mAPtools.py) following **MVDNet** -->
<!--```-->
<!-- python -m pip install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html -->
<!-- ``` -->

### Prepare Data
Following the instruction from [MVDNet](https://github.com/qiank10/MVDNet/tree/main).

Download the [Oxford Radar RobotCar Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset). Currently, only the vehicles in the first data record (Date: 10/01/2019, Time: 11:46:21 GMT) are labeled. After unzipping the files, the directory should look like this:
```
# Oxford Radar RobotCar Data Record
|-- DATA_PATH
    |-- gt
    |-- radar
    |-- velodyne_left
    |-- velodyne_right
    |-- vo
    |-- radar.timestamps
    |-- velodyne_left.timestamps
    |-- velodyne_right.timestamps
    |-- ...
```

Prepare the radar data:
```
python MVDNet/data/sdk/prepare_radar_data.py --data_path DATA_PATH --image_size 320 --resolution 0.2
```

Prepare the lidar data:
```
python MVDNet/data/sdk/prepare_lidar_data.py --data_path DATA_PATH
```

Prepare the foggy lidar test set with specified fog density, e.g., 0.05:
```
python MVDNet/data/sdk/prepare_fog_data.py --data_path DATA_PATH --beta 0.05
```

The processed data is organized as follows:
```
# Oxford Radar RobotCar Data Record
|-- DATA_PATH
    |-- processed
        |-- radar
            |-- 1547120789640420.jpg
            |-- ...
        |-- radar_history
            |-- 1547120789640420_k.jpg   # The k-th radar frame preceding the frame at the timestamp 1547120789640420, k=1,2,3,4.
            |-- ...
        |-- lidar
            |-- 1547120789640420.bin
            |-- ...
        |-- lidar_history
            |-- 1547120789640420_k.bin   # Link to the k-th lidar frame preceding the frame at the timestamp 1547120789640420, k=1,2,3,4.
            |-- 1547120789640420_k_T.bin # Transform matrix between the k-th preceding lidar frame and the current frame.
            |-- ...
        |-- lidar_fog_0.05               # Foggy lidar data with fog density as 0.05
            |-- 1547120789640420.bin
            |-- ...
        |-- lidar_history_fog_0.05
            |-- 1547120789640420_k.bin
            |-- 1547120789640420_k_T.bin
            |-- ...
```

Both 2D and 3D labels are in
```
./data/RobotCar/object/
```

Modify the [data path](https://github.com/yyxr75/RaLiBEV/blob/696835910bd45a13f64d073494a255cdb7f78d2f/data/datasets/oxford_dataset/oxford_dataloader.py#L15)


### Train RaLiBEV

The experiments was pre-defined in `lidar_and_4D_imaging_radar_fusion_demo/utils/opts.py`, all you need to do is modify the name to switch on/off these tests. The bash script example is shown in `lidar_and_4D_imaging_radar_fusion_demo/train.sh`.

**run training**
```
sh train.sh 
```
In train.sh, choose

`--fusion_arch` as [feature_level_concatenation](https://github.com/yyxr75/RaLiBEV/blob/696835910bd45a13f64d073494a255cdb7f78d2f/utils/opts.py#L64)

`--label_assign_strategy` as [0_Direct_Index-based_Positive_Sample_Assignment_(DIPSA)](https://github.com/yyxr75/lidar_and_4D_imaging_radar_fusion_perception/blob/489a2efd1c32ffd1b07eaabb4cb2f3f2f1c9c97f/RaLiBEV/utils/opts.py#L87)

`--experiment_name` as whatever you want.

These options can be queried in `lidar_and_4D_imaging_radar_fusion_demo/utils/opts.py`.

Example as below:

```Bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup python -u -X faulthandler \
      -m torch.distributed.launch \
      --nproc_per_node=4 \
      --master_port 348043 \
      train.py \
      --resume 0 \
      --distributed 0 \
      --using_multi_scale 1 \
      --batch_size 32 \
      --start_epoch 0 \
      --end_epoch 200 \
      --num_workers 0 \
      --cuda 1 \
      --rotate_angle 0 \
      --fusion_arch feature_level_concatenation \
      --fusion_loss_arch fusion_loss_base\
      --binNum 8 \
      --label_assign_strategy 0_Direct_Index-based_Positive_Sample_Assignment_(DIPSA) \
      --showHeatmap 0 \
      --chooseLoss 0 \
      --lr_scheduler warmup_multistep \
      --base_lr 0.01 \
      --min_lr_ratio 0.05 \
      --warmup_epochs 2 \
      --warmup_lr 0 \
      --weight_decay 1e-4 \
      --main_gpuid 0 \
      --mAP_type pycoco \
      --experiment_name 0819_0_Direct_Index-based_Positive_Sample_Assignment_(DIPSA) \
>> train_0819_0_Direct_Index-based_Positive_Sample_Assignment_(DIPSA).log 2>&1 &
```

### Test RaLiBEV

Test share the same config file `RaLiBEV/utils/opts.py` with train. The mAP function entry is `RaLiBEV/calc_mAP.py`

**run test**
```
python calc_mAP.py

```

In order to load weights from pre-trained model, the [`--experiment_name`]() and [`--model_path`](https://github.com/yyxr75/RaLiBEV/blob/696835910bd45a13f64d073494a255cdb7f78d2f/utils/opts.py#L31) should be specified.

For example:

```
python calc_mAP.py --experiment_name Label_Assignment_Base_Model_test --model_path epoch111-train_0.509-val_0.860_ckpt.pth
```

## Model Zoo

| Model | Link |
|----------|----------|
| Label Assignment Base Model | [Model](https://drive.google.com/file/d/1oSLwK7r3PLC9QsaL8pbtae-CTO8b1chO/view?usp=drive_link) |
| Direct Transformer for BEV Fusion | [Model](https://drive.google.com/file/d/1S2YIgP9Ag1F-k-Ukqd22IEYs6k5v1Zm5/view?usp=drive_link) |
| Dense Query Map-based Interactive Transformer for BEV Fusion (DQMITBF) | [Model](https://drive.google.com/file/d/1Qht3ENwe9nCkjmF9eQYCE9gsg64JZnXe/view?usp=drive_link) |

## Citation

If you find our code helpful, please cite:
```
@misc{yang2024ralibevradarlidarbev,
      title={RaLiBEV: Radar and LiDAR BEV Fusion Learning for Anchor Box Free Object Detection Systems}, 
      author={Yanlong Yang and Jianan Liu and Tao Huang and Qing-Long Han and Gang Ma and Bing Zhu},
      year={2024},
      eprint={2211.06108},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2211.06108}, 
}
```
