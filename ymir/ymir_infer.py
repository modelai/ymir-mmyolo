import argparse
import os
import sys
from typing import List

import cv2
import torch.distributed as dist
from easydict import EasyDict as edict
from mmdet.apis import inference_detector, init_detector
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.config import DictAction
from mmengine.dist import collect_results_gpu, init_dist
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import (YmirStage, get_merged_config,
                           write_ymir_monitor_process)

from mmyolo.utils import register_all_modules
from ymir.utils.common import get_best_weight_file, get_config_file

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_option(cfg_options: str) -> dict:
    parser = argparse.ArgumentParser(description='parse cfg options')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')

    args = parser.parse_args(f'--cfg-options {cfg_options}'.split())
    return args.cfg_options


def mmdet_result_to_ymir(results: DetDataSample, class_names: List[str]) -> List[rw.Annotation]:
    """
    results: DetDataSample
    """
    ann_list = []
    scores = results.pred_instances.scores
    bboxes = results.pred_instances.bboxes
    labels = results.pred_instances.labels
    for idx, result in enumerate(zip(bboxes, scores, labels)):
        bbox, score, label = result
        x1, y1, x2, y2 = [x.item() for x in bbox]
        score = score.item()
        label = label.item()
        ann = rw.Annotation(class_name=class_names[label],
                            score=score,
                            box=rw.Box(x=round(x1), y=round(y1), w=round(x2 - x1), h=round(y2 - y1)))
        ann_list.append(ann)
    return ann_list


class YmirModel:

    def __init__(self, cfg: edict):
        self.cfg = cfg

        # Specify the path to model config and checkpoint file
        config_file = get_config_file(cfg)
        checkpoint_file = get_best_weight_file(cfg)

        gpu_id = max(0, RANK)
        # build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device=f'cuda:{gpu_id}')

    def infer(self, img):
        return inference_detector(self.model, img)


def main():
    register_all_modules()

    if LOCAL_RANK != -1:
        init_dist(launcher='pytorch', backend="nccl" if dist.is_nccl_available() else "gloo")

    cfg = get_merged_config()

    with open(cfg.ymir.input.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    max_barrier_times = len(images) // WORLD_SIZE
    if RANK == -1:
        N = len(images)
        tbar = tqdm(images)
    else:
        images_rank = images[RANK::WORLD_SIZE]
        N = len(images_rank)
        if RANK == 0:
            tbar = tqdm(images_rank)
        else:
            tbar = images_rank
    infer_result_list = []
    model = YmirModel(cfg)

    # write infer result
    monitor_gap = max(1, N // 100)
    conf_threshold = float(cfg.param.conf_threshold)
    for idx, asset_path in enumerate(tbar):
        img = cv2.imread(asset_path)
        result = model.infer(img)
        raw_anns = mmdet_result_to_ymir(result, cfg.param.class_names)

        # batch-level sync, avoid 30min time-out error
        if WORLD_SIZE > 1 and idx < max_barrier_times:
            dist.barrier()

        infer_result_list.append((asset_path, [ann for ann in raw_anns if ann.score >= conf_threshold]))

        if idx % monitor_gap == 0 and RANK in [0, -1]:
            write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=idx / N, stage=YmirStage.TASK)

    if WORLD_SIZE > 1:
        dist.barrier()
        infer_result_list = collect_results_gpu(infer_result_list, len(images))

    if RANK in [0, -1]:
        infer_result_dict = {k: v for k, v in infer_result_list}
        rw.write_infer_result(infer_result=infer_result_dict)
        write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
