"""
hook for ymir training process, write the monitor.txt, save the latest model
"""
import glob
import logging
import os.path as osp
import re
import warnings
from typing import Dict, Optional, Union

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from ymir_exc.util import (get_merged_config, write_ymir_monitor_process,
                           write_ymir_training_result)


@HOOKS.register_module()
class YmirTrainingMonitorHook(Hook):
    """
    for epoch based training loop only.

    1. write monitor.txt
    2. save the latest checkpoint with id=last if exist, note the checkpoint maybe clear late.
    3. save the latest best checkpoint with id=best if exist, note the checkpoint maybe clear late.
    """
    # the priority should lower than CheckpointHook (priority = VERY_LOW)
    priority = 'LOWEST'

    def __init__(self, interval: int = 10):
        self.interval = interval
        self.ymir_cfg = get_merged_config()

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: Optional[Union[dict, tuple, list]] = None,
                         outputs: Optional[dict] = None) -> None:
        if runner.rank in [0, -1] and self.every_n_inner_iters(batch_idx, self.interval):
            percent = (runner.epoch + batch_idx / len(runner.train_dataloader)) / runner.max_epochs
            write_ymir_monitor_process(self.ymir_cfg, task='training', naive_stage_percent=percent, stage='task')

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        metrics: {'coco/bbox_mAP': 0.001, 'coco/bbox_mAP_50': 0.003,
        'coco/bbox_mAP_75': 0.0, 'coco/bbox_mAP_s': 0.0, 'coco/bbox_mAP_m': 0.0, 'coco/bbox_mAP_l': 0.001}

        evaluation_result: {'mAP': 0.001, 'mAP_50': 0.003, ...}
        """
        if runner.rank in [0, -1]:
            N = len('coco/bbox_')
            evaluation_result = {key[N:]: value for key, value in metrics.items()}  # type: ignore
            out_dir = self.ymir_cfg.ymir.output.models_dir
            cfg_files = glob.glob(osp.join(out_dir, '*.py'))

            try:
                test_cfg = runner.model.test_cfg
                evaluate_config = dict(iou_thr=test_cfg.nms.iou_threshold, conf_thr=test_cfg.score_thr)
            except (KeyError, AttributeError):
                evaluate_config = None

            best_ckpts = glob.glob(osp.join(out_dir, 'best_coco', '*.pth'))
            if len(best_ckpts) > 0:
                newest_best_ckpt = max(best_ckpts, key=osp.getctime)
                best_epoch = int(re.findall(r'\d+', newest_best_ckpt)[0])
                # if current checkpoint is the newest checkpoint, keep it
                if best_epoch == runner.epoch:
                    logging.info(f'epoch={runner.epoch}, save {newest_best_ckpt} to result.yaml')
                    write_ymir_training_result(self.ymir_cfg,
                                               files=[newest_best_ckpt] + cfg_files,
                                               id='best',
                                               evaluation_result=evaluation_result,
                                               evaluate_config=evaluate_config)
            else:
                warnings.warn(f'no best checkpoint found on {runner.epoch}')

            latest_ckpts = glob.glob(osp.join(out_dir, '*.pth'))
            if len(latest_ckpts) > 0:
                last_ckpt = max(latest_ckpts, key=osp.getctime)
                logging.info(f'epoch={runner.epoch}, save {last_ckpt} to result.yaml')
                write_ymir_training_result(self.ymir_cfg,
                                           files=[last_ckpt] + cfg_files,
                                           id='last',
                                           evaluation_result=evaluation_result,
                                           evaluate_config=evaluate_config)
            else:
                warnings.warn(f'no latest checkpoint found on {runner.epoch}')
