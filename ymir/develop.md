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

- 加载超参数

- 写进度

- 写结果文件
