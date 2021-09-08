'''
Author: your name
Date: 2021-08-29 19:45:54
LastEditTime: 2021-09-02 21:38:50
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /models/HOTR/test_case.py
'''
from hotr.models.known_novel_split import known_objects, novel_objects
from  hotr.data.datasets.builtin_meta import COCO_CATEGORIES
import json
# known_ids = []
# unknown_ids=[]
# for dict_item  in  COCO_CATEGORIES:
#     if dict_item.get('name') in known_objects:
#         known_ids.append(dict_item.get(id))
#     elif dict_item.get('name') in novel_objects:
#         unknown_ids.append(dict_item.get(id))

# print(len(known_ids))
# print(len(unknown_ids))

# print(known_ids)
# print(unknown_ids)


def test_zero_shot_hoi(data_dir):
    with open(data_dir, 'r') as f:
        data = json.loads(f.readlines()[0])
        count = 0
        for image, anno, category in zip(data['images'],data['annotations'],data['categories']):
            print(image, anno, category)
            count+=1
            if count>=3:
                break
            
def normalhoi(data_dir):
    with open(data_dir, 'r') as f:
        count = 0
        data = json.loads(f.readlines()[0])
        print(len(data))
        # for item in data:
        #     print(item)
        #     count+=1
        #     if count>=3:
        #         break


json_dir0="/mnt/lustre/zongdaoming/models/FireAssociation/data/hico_det/hico_20160224_det/annotations/instances_hico_train_seen.json"
json_dir1="/mnt/lustre/zongdaoming/models/FireAssociation/data/hico_det/hico_20160224_det/annotations/instances_hico_test.json"
json_dir2 = "/mnt/lustre/zongdaoming/models/FireAssociation/data/hico_det/hico_20160224_det/annotations/test_hico.json"
json_dir3 = "/mnt/lustre/zongdaoming/models/FireAssociation/data/hico_det/hico_20160224_det/annotations/trainval_hico.json"

# import numpy as np
# correct_mat = "/mnt/lustre/zongdaoming/models/FireAssociation/data/hico_det/hico_20160224_det/annotations/corre_hico.npy"
# matrix= np.load(correct_mat)
# print(matrix.shape) # 117,80



# test_zero_shot_hoi(json_dir1)
# test_zero_shot_hoi(json_dir0)
normalhoi(json_dir3)

# test_zero_shot_hoi(json_dir3)
# instances_hico_test: 9658
# instances_hico_train: 38118

# instance_hico_test: 9546
# trainval_hico: 37633

# dic_ = json.loads(f.readlines()[0])        
# print(len(dic_['images']))


