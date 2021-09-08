#!/usr/bin/env python
#Author: Daoming Zong
#Date: 2021-09-04 00:34:37
#LastEditTime: 2021-09-07 16:27:56
#LastEditors: Daoming Zong and Chunya Liu
#Description: 
#FilePath: /models/HOTR/hotr/engine/evaluator_hico.py
#Copyright (c) 2021 SenseTime IRDC Group. All Rights Reserved.
#
#
#Author: your name
#Date: 2021-09-03 01:14:54
#LastEditTime: 2021-09-03 01:18:07
#LastEditors: Please set LastEditors
#Description: In User Settings Edit
#FilePath: /models/HOTR/hotr/engine/evaluator_hico.py
#
'''
Author: your name
Date: 2021-09-02 23:58:02
LastEditTime: 2021-09-02 23:59:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /models/HOTR/hotr/engine/evaluator_hico.py
'''
import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import hotr.util.misc as utils
import hotr.util.logger as loggers
# from hotr.data.evaluators.hico_eval import HICOEvaluator
from hotr.data.evaluators.hico_eval_seen_unseen import HICOUSEvaluator

@torch.no_grad()
def hico_evaluate(model, postprocessors, data_loader, device, thr):
    model.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (HICO-DET)'

    preds = []
    gts = []
    indices = []
    hoi_recognition_time = []

    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: (v.to(device) if k != 'id' and k != 'file_name' else v) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes, threshold=thr, dataset='hico-det')
        hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

    print(f"[stats] HOI Recognition Time (avg) : {sum(hoi_recognition_time)/len(hoi_recognition_time):.4f} ms")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]
    
    evaluator = HICOUSEvaluator(preds, 
                                gts, 
                                data_loader.dataset.rare_triplets, 
                                data_loader.dataset.non_rare_triplets, 
                                data_loader.dataset.seen_triplets, 
                                data_loader.dataset.unseen_triplets, 
                                data_loader.dataset.correct_mat
                            )
    stats = evaluator.evaluate()

    return stats