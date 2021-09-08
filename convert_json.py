'''
Author: your name
Date: 2021-09-02 20:40:27
LastEditTime: 2021-09-03 00:07:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /models/HOTR/convert_json.py
'''
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file    :   convert_json.py
# @time    :   2021/09/02 17:22:36
# @authors  :  daoming zong, chunya liu
# @version :   1.0
# @contact :   zongdaoming@sensetime.com; liuchunya@sensetime.com
# @desc    :   None
# Copyright (c) 2021 SenseTime IRDC Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from hotr.data.datasets.hico_meta import HICO_OBJECTS
from hotr.models.known_novel_split import known_objects, novel_objects

known_objects_ids_to_names = {int(key["id"]): key['name']
                             for key in HICO_OBJECTS if key["isknown"] == 1}
unknown_objects_ids_to_names = {int(key["id"]): key['name']
                               for key in HICO_OBJECTS if key["isknown"] == 0}

def generate_seen_hico_train_seen(data_dir,data_save_dir):
    hico_seen_train = []
    with open(data_save_dir, 'w') as g:
        with open(data_dir, 'r') as f:
            dict_item = json.loads(f.readlines()[0])
            for elem in dict_item:
                keep_hoi_annotations = []
                for item in elem['hoi_annotation']:
                    if known_objects_ids_to_names.get(elem['annotations'][item['object_id']].get('category_id'), None):
                        keep_hoi_annotations.append(item)
                # new hoi_annotation
                elem['hoi_annotation'] = keep_hoi_annotations
                if len(elem['hoi_annotation'])>0:
                    hico_seen_train.append(elem)
        json.dump(hico_seen_train,g)


def test_seen_hico_train_seen(data_save_dir):
    with open(data_save_dir, 'r') as f:
        dict_item = json.loads(f.readlines()[0])
        count = 0
        for elem in dict_item:
            print(elem)
            count+=1
            if count>=20:
                break

# /mnt/lustre/zongdaoming/models/FireAssociation/data/hico_det/hico_20160224_det/annotations/trainval_hico.json
data_dir = "/mnt/lustre/zongdaoming/models/FireAssociation/data/hico_det/hico_20160224_det/annotations/trainval_hico.json"
data_save_dir = "/mnt/lustre/zongdaoming/models/FireAssociation/data/hico_det/hico_20160224_det/annotations/hico_seen_trainval.json"
# generate_seen_hico_train_seen(data_dir,data_save_dir)
# test_seen_hico_train_seen(data_save_dir)
