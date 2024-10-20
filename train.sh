CUDA_VISIBLE_DEVICES=1,2,3,4,5 \
nohup python -u -X faulthandler \
      -m torch.distributed.launch \
      --nproc_per_node=5 \
      --master_port 348043 \
      train.py \
      --distributed 1 \
      --using_multi_scale 1 \
      --batch_size 20\
      --start_epoch 0 \
      --end_epoch 200 \
      --num_workers 0 \
      --cuda 1 \
      --rotate_angle 0 \
      --fusion_arch Dense_Query_Map-based_Interactive_Transformer_for_BEV_Fusion_(DQMITBF) \
      --fusion_loss_arch fusion_loss_base\
      --binNum 8 \
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
      --experiment_name 0811_QVariable_KVRaLi_KVLiRa_cat \
>> train_0811_QVariable_KVRaLi_KVLiRa_cat.log 2>&1 &
