import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from mmdet.apis import inference_detector, init_detector
from mmengine.dist import collect_results_gpu, init_dist
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import (YmirStage, get_merged_config,
                           write_ymir_monitor_process)

from mmyolo.utils import register_all_modules
from ymir.utils.common import get_best_weight_file, get_config_file
from ymir.ymir_infer import get_cfg_options

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class RandomMiner(object):

    def __init__(self, cfg: edict):
        if LOCAL_RANK != -1:
            init_dist(launcher='pytorch', backend="nccl" if dist.is_nccl_available() else "gloo")

        self.cfg = cfg
        gpu_id = max(0, LOCAL_RANK)
        self.device = f'cuda:{gpu_id}'

        self.conf_threshold = float(cfg.param.conf_threshold)
        config_file = get_config_file(cfg)
        checkpoint_file = get_best_weight_file(cfg)
        cfg_options = get_cfg_options(cfg)
        self.model = init_detector(config_file, checkpoint_file, device=f'cuda:{gpu_id}', cfg_options=cfg_options)

    def infer(self, img):
        return inference_detector(self.model, img)

    def mining(self):
        with open(self.cfg.ymir.input.candidate_index_file, 'r') as f:
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

        monitor_gap = max(1, N // 100)

        mining_result = []
        for idx, asset_path in enumerate(tbar):
            if idx % monitor_gap == 0:
                write_ymir_monitor_process(cfg=self.cfg,
                                           task='mining',
                                           naive_stage_percent=idx / N,
                                           stage=YmirStage.TASK,
                                           task_order='tmi')

            if WORLD_SIZE > 1 and idx < max_barrier_times:
                dist.barrier()

            with torch.no_grad():
                consistency = self.compute_score(asset_path=asset_path)
            mining_result.append((asset_path, consistency))

        if WORLD_SIZE > 1:
            mining_result = collect_results_gpu(mining_result, len(images))

        if RANK in [0, -1]:
            rw.write_mining_result(mining_result=mining_result)
            write_ymir_monitor_process(cfg=self.cfg,
                                       task='mining',
                                       naive_stage_percent=1,
                                       stage=YmirStage.POSTPROCESS,
                                       task_order='tmi')
        return mining_result

    def compute_score(self, asset_path: str) -> float:
        return random.random()


class EntropyMiner(RandomMiner):

    def compute_score(self, asset_path: str) -> float:
        results = self.infer(asset_path)
        conf = results.pred_instances.scores.data.cpu().numpy()
        conf = conf[conf > self.conf_threshold]

        # if not empty, mining_score > 0
        if len(conf) == 0:
            return 0

        mining_score = -np.sum(conf * np.log2(conf))
        return mining_score


def main():
    register_all_modules()

    cfg = get_merged_config()
    mining_algorithm = cfg.param.mining_algorithm
    supported_miner = ['random', 'entropy']

    assert mining_algorithm in supported_miner, f'unknown mining_algorithm {mining_algorithm}, not in {supported_miner}'
    if mining_algorithm == 'random':
        miner = RandomMiner(cfg)
    elif mining_algorithm == 'entropy':
        miner = EntropyMiner(cfg)

    miner.mining()
    return 0


if __name__ == "__main__":
    sys.exit(main())
