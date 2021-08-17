#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file    :   test.py
# @time    :   2021/08/17 22:21:15
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

def test_zero_shot_hoi(data_dir):
    with open(data_dir, 'r') as f:
        dic_ = json.loads(f.readlines()[0])
        for id, item in enumerate(dic_['images']):
            if item['file_name'] in ["HICO_train2015_00019135.jpg"]:
                print(f"Before update {item}")
                dic_['images'][id]['height']=396
                dic_['images'][id]['width']=640
    with open(data_dir, 'w') as f:
        f.write(json.dumps(dic_))
    
    with open(data_dir, 'r') as f:
        dic_ = json.loads(f.readlines()[0])
        for id, item in enumerate(dic_['images']):
            if item['file_name'] in ["HICO_train2015_00019135.jpg"]:
                print(f"Update file_json {item}")

json_dir="/mnt/lustre/zongdaoming/models/zero_shot_hoi/zero_shot_hoi/datasets/hico_20160224_det/annotations/instances_hico_train.json"
test_zero_shot_hoi(json_dir)
# {'file_name': 'HICO_train2015_00018679.jpg', 'height': 640, 'width': 427, 'id': 18679}