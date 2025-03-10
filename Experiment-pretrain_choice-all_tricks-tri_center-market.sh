# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('2')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/home/haoluo/data')" MODEL.PRETRAIN_CHOICE "('self')" MODEL.PRETRAIN_PATH "('/home/haoluo/log/gu/reid_baseline_review/Opensource_test/market1501/Experiment-pretrain_choice_all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/resnet50_model_2.pth')" OUTPUT_DIR "('/home/haoluo/log/gu/reid_baseline_review/Opensource_test/market1501/Experiment-pretrain_choice_all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005')"
python tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/root/workspace/reid/code/reid-strong-baseline-master/data')" MODEL.PRETRAIN_CHOICE "('self')" MODEL.PRETRAIN_PATH "('/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')" OUTPUT_DIR "('/root/workspace/reid/code/reid-strong-baseline-master/output')"