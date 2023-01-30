# 开发文档

## 训练

### 训练脚本调用链

1. 启动镜像时调用 `bash /usr/bin/start.sh`

2. `start.sh` 调用  `python3 ymir/start.py`

3. `start.py` 调用  `python3 ymir/ymir_training.py`

4. `ymir_training.py` 调用 `bash tools/dist_train.sh ...`

### 核心功能实现

- 数据格式转换

在 `ymir_training.py` 中首次调用 `convert_ymir_to_coco()` 进行数据格式转换，后续调用 `convert_ymir_to_coco()` 仅获得数据集信息。

- 加载预训练权重

    - 参考 `get_best_weight_file()`

    - 如果用户提供预训练权重， 则先其中找带 `best_` 或 `bbox_mAP_` 的权重，其次找带 `epoch_` 的权重， 最后选择其中最新的。

    - 如果用户没有提供预训练权重，则在镜像的 `/weights` 目录下， 通过超参数 `model_name` 获得 `config_file` 再通过相似度找到最相似的权重文件。

- 加载超参数

    - 参考 `modify_mmengine_config()`, 将ymir超参数覆盖 `mmengine.config.Config`

- 写进度

    - 参考 `YmirTrainingMonitorHook`, 该 hook 可实时返回进度信息， 并保存最新的权重文件到ymir中，以支持提前终止训练的功能。

- 写结果文件

    - 参考 `YmirTrainingMonitorHook` 与 `write_mmyolo_training_result()`, 其中后者支持依据超参数 `max_keep_checkpoints` 保存多个权重文件。

## 推理

1. 启动镜像时调用 `bash /usr/bin/start.sh`

2. `start.sh` 调用  `python3 ymir/start.py`

3. `start.py` 调用  `python3 ymir/ymir_infer.py`

    - 调用 `init_detector()` 与 `inference_detector()` 获取推理结果

    - 调用 `mmdet_result_to_ymir()` 将mmdet推理结果转换为ymir格式

    - 调用 `rw.write_infer_result()` 保存推理结果

## 挖掘

1. 启动镜像时调用 `bash /usr/bin/start.sh`

2. `start.sh` 调用  `python3 ymir/start.py`

3. `start.py` 调用  `python3 ymir/ymir_mining.py`

    - 调用 `init_detector()` 与 `inference_detector()` 获取推理结果

    - 调用 `compute_score()` 计算挖掘分数

    - 调用 `rw.write_mining_result()` 保存挖掘结果
