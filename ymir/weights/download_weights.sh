# view https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov5

# yolov5 640 n,s,m
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739-b804c1ad.pth
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth

# yolov6 640 n,t,s,m
wget https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco_20221030_202726-d99b2e82.pth
wget https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_t_syncbn_fast_8xb32-400e_coco/yolov6_t_syncbn_fast_8xb32-400e_coco_20221030_143755-cf0d278f.pth
wget https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth
wget https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_m_syncbn_fast_8xb32-300e_coco/yolov6_m_syncbn_fast_8xb32-300e_coco_20221109_182658-85bda3f4.pth

# yolov7 tiny, l, x
wget https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth
wget https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth
wget https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_x_syncbn_fast_8x16b-300e_coco/yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth

# yolov8 n, s, m
wget https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth
wget https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth
wget https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth

# yolox tiny, s
wget https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_8xb8-300e_coco/yolox_tiny_8xb8-300e_coco_20220919_090908-0e40a6fc.pth
wget https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20220917_030738-d7e60cb2.pth

# rtmdet tiny, s, m
wget https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth
wget https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-0a8c901a.pth
wget https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth

# ppyoloe+ s, m, l
wget https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco/ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052-9fee7619.pth
wget https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_m_fast_8xb8-80e_coco/ppyoloe_plus_m_fast_8xb8-80e_coco_20230104_193132-e4325ada.pth
wget https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_l_fast_8xb8-80e_coco/ppyoloe_plus_l_fast_8xb8-80e_coco_20230102_203825-1864e7b3.pth

# rtmdet imagenet pretrain, copy to /root/.cache/torch/hub/checkpoints
cd imagenet
wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth
wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth
