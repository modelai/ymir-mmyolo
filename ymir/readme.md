# mmyolo 镜像说明文档

## 仓库地址

> 参考[open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo)
- [modelai/ymir-mmyolo](https://github.com/modelai/ymir-mmyolo)

## 镜像地址

```
youdaoyzbx/ymir-executor:ymir2.1.0-mmyolo-cu113-tmi
```

## 性能表现

> 结果来自mmyolo官方的COCO评测, 表格外的其他结构同样支持，但镜像中没有提供相应的预训练模型

### yolov5-v8, yolox, ppyoloe+

| Backbone | Arch | size | SyncBN | AMP | Mem (GB) | box AP |                                                  Config                                                   |                                                                                                                                                           Download                                                                                                                                                           |
| :------: | :--: | :--: | :----: | :-: | :------: | :----: | :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv8-n |  P5  | 640  |  Yes   | Yes |   2.8    |  37.2  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_n_syncbn_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804.log.json) |
| YOLOv8-s |  P5  | 640  |  Yes   | Yes |   4.0    |  44.2  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_s_syncbn_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101.log.json) |
| YOLOv8-m |  P5  | 640  |  Yes   | Yes |   7.2    |  49.8  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_m_syncbn_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200.log.json) |
| YOLOv7-tiny |  P5  | 640  |  Yes   | Yes |   2.7    |  37.5  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719.log.json) |
|  YOLOv7-l   |  P5  | 640  |  Yes   | Yes |   10.3   |  50.9  |  [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601.log.json)       |
|  YOLOv7-x   |  P5  | 640  |  Yes   | Yes |   13.7   |  52.8  |  [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov7/yolov7_x_syncbn_fast_8x16b-300e_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_x_syncbn_fast_8x16b-300e_coco/yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_x_syncbn_fast_8x16b-300e_coco/yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331.log.json)       |
| YOLOv6-n |  P5  | 640  |  Yes   | Yes |   6.04   |  36.2  | [config](../yolov6/yolov6_n_syncbn_fast_8xb32-400e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco_20221030_202726-d99b2e82.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco_20221030_202726.log.json) |
| YOLOv6-t |  P5  | 640   |  Yes   | Yes |   8.13   |  41.0  | [config](../yolov6/yolov6_t_syncbn_fast_8xb32-400e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_t_syncbn_fast_8xb32-400e_coco/yolov6_t_syncbn_fast_8xb32-400e_coco_20221030_143755-cf0d278f.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_t_syncbn_fast_8xb32-400e_coco/yolov6_t_syncbn_fast_8xb32-400e_coco_20221030_143755.log.json) |
| YOLOv6-s |  P5  | 640  |  Yes   | Yes |   8.88   |  44.0  | [config](../yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035.log.json) |
| YOLOv6-m |  P5  | 640  |  Yes   | Yes |  16.69   |  48.4  | [config](../yolov6/yolov6_m_syncbn_fast_8xb32-400e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_m_syncbn_fast_8xb32-300e_coco/yolov6_m_syncbn_fast_8xb32-300e_coco_20221109_182658-85bda3f4.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_m_syncbn_fast_8xb32-300e_coco/yolov6_m_syncbn_fast_8xb32-300e_coco_20221109_182658.log.json) |
| YOLOv5-n |  P5  | 640  |  Yes   | Yes |   1.5    |  28.0  |  [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739-b804c1ad.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739.log.json)       |
| YOLOv5-s |  P5  | 640  |  Yes   | Yes |   2.7    |  37.7  |  [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json)       |
| YOLOv5-m |  P5  | 640  |  Yes   | Yes |   5.0    |  45.3  |  [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944.log.json)       |
| YOLOX-tiny | - | 416  | - | - |   2.8    |  32.7  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolox/yolox_tiny_8xb8-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_8xb8-300e_coco/yolox_tiny_8xb8-300e_coco_20220919_090908-0e40a6fc.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_8xb8-300e_coco/yolox_tiny_8xb8-300e_coco_20220919_090908.log.json) |
|  YOLOX-s   | - | 640  | - | - |   5.6    |  40.8  |  [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolox/yolox_s_8xb8-300e_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20220917_030738-d7e60cb2.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20220917_030738.log.json)       |
| PPYOLOE_plus_s |  P5  | 640  |  Yes   | - |    4.7    |  43.5  | [config](../ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco/ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052-9fee7619.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco/ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052.log.json) |
| PPYOLOE_plus_m |  P5  | 640  |  Yes   | - |    8.4    |  49.5  | [config](../ppyoloe/ppyoloe_plus_m_fast_8xb8-80e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_m_fast_8xb8-80e_coco/ppyoloe_plus_m_fast_8xb8-80e_coco_20230104_193132-e4325ada.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_m_fast_8xb8-80e_coco/ppyoloe_plus_m_fast_8xb8-80e_coco_20230104_193132.log.json) |


### RTMDet

|    Model    | size | box AP | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms) |                       Config                        |                                                                                                                                                                 Download                                                                                                                                                                 |
| :---------: | :--: | :----: | :-------: | :------: | :------------------: | :-------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-tiny | 640  |  41.0  |    4.8    |   8.1    |         0.98         | [config](./rtmdet_l_syncbn_fast_8xb32-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117.log.json) |
|  RTMDet-s   | 640  |  44.6  |   8.89    |   14.8   |         1.22         | [config](./rtmdet_s_syncbn_fast_8xb32-300e_coco.py) |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-0a8c901a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329.log.json)       |
|  RTMDet-m   | 640  |  49.3  |   24.71   |  39.27   |         1.62         | [config](./rtmdet_m_syncbn_fast_8xb32-300e_coco.py) |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952.log.json)       |
|  RTMDet-l   | 640  |  51.4  |   52.3    |  80.23   |         2.44         | [config](./rtmdet_l_syncbn_fast_8xb32-300e_coco.py) |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928.log.json)       |

## 训练参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| config_file |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| export_format | ark:raw | 字符串| 受ymir后台处理，ymir数据集导出格式 | - |
| model_name | yolov8_n | 字符串 | 模型简写, 如yolov7_tiny, yolov5_m, yolov6_t, rtmdet_m, ppyoloe_plus_s | 支持yolov5-v8, yolox, rtmdet, ppyoloe_plus |
| samples_per_gpu | 8 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| max_epochs | 100 | 整数 | 整个数据集的训练遍历次数 | 建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| args_options | '' | 字符串 | 训练命令行参数 | 参考 [ymir-mmyolo/tools/train.py](https://github.com/modelai/ymir-mmyolo/blob/ymir/tools/train.py)
| cfg_options | '' | 字符串 | 训练命令行参数 | 参考 [ymir-mmyolo/tools/train.py](https://github.com/modelai/ymir-mmyolo/blob/ymir/tools/train.py)
| metric | bbox | 字符串 | 模型评测方式 | 采用默认值即可 |
| val_interval | 1 | 整数 | 模型在验证集上评测的周期， 以epoch为单位 | 设置为1，每个epoch可评测一次 |
| max_keep_checkpoints | 1 | 整数 | 最多保存的权重文件数量 | 设置为k, 可保存k个最优权重和k个最新的权重文件，设置为-1可保存所有权重文件。


## 推理参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| conf_threshold | 0.2 | 浮点数 | 推理结果置信度过滤阈值 | 设置为0可保存所有结果，设置为0.6可过滤大量结果 |

## 挖掘参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| mining_algorithm | entropy | 字符串 | 挖掘算法可选 entropy 和 random | 建议采用entropy |
| conf_threshold | 0.1 | 浮点数 | 推理结果置信度过滤阈值 | 设置为0可保存所有结果，设置为0.1可过滤一些推理结果，避免挖掘算法受低置信度结果影响 |
