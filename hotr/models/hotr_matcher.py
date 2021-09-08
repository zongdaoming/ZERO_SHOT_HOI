# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr_matcher.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from hotr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import hotr.util.misc as utils
from utils.log_helper import default_logger as logger
# # import wandb

class HungarianPairMatcher(nn.Module):
    def __init__(self, args):
        """Creates the matcher
        Params:
            cost_action: This is the relative weight of the multi-label action classification error in the matching cost
            cost_hbox: This is the relative weight of the classification error for human idx in the matching cost
            cost_obox: This is the relative weight of the classification error for object idx in the matching cost
        """
        super().__init__()
        self.cost_action = args.set_cost_act
        self.cost_hbox = self.cost_obox = args.set_cost_idx
        self.cost_target = args.set_cost_tgt
        self.cost_interaction =args.set_cost_interaction
        self.log_printer = args.log
        self.is_vcoco = (args.dataset_file == 'vcoco')
        self.is_hico = (args.dataset_file == 'hico-det')
        if self.is_vcoco:
            self.valid_ids = args.valid_ids
            self.invalid_ids = args.invalid_ids
        assert self.cost_action != 0 or self.cost_hbox != 0 or self.cost_obox != 0, "all costs cant be 0"

    def reduce_redundant_gt_box(self, tgt_bbox, indices):
        """Filters redundant Ground-Truth Bounding Boxes
        Due to random crop augmentation, there exists cases where there exists
        multiple redundant labels for the exact same bounding box and object class.
        This function deals with the redundant labels for smoother HOTR training.
        """
        tgt_bbox_unique, map_idx, idx_cnt = torch.unique(tgt_bbox, dim=0, return_inverse=True, return_counts=True)

        k_idx, bbox_idx = indices
        triggered = False
        if (len(tgt_bbox) != len(tgt_bbox_unique)):
            map_dict = {k: v for k, v in enumerate(map_idx)}
            map_bbox2kidx = {int(bbox_id): k_id for bbox_id, k_id in zip(bbox_idx, k_idx)}

            bbox_lst, k_lst = [], []
            for bbox_id in bbox_idx:
                if map_dict[int(bbox_id)] not in bbox_lst:
                    bbox_lst.append(map_dict[int(bbox_id)])
                    k_lst.append(map_bbox2kidx[int(bbox_id)])
            bbox_idx = torch.tensor(bbox_lst)
            k_idx = torch.tensor(k_lst)
            tgt_bbox_res = tgt_bbox_unique
        else:
            tgt_bbox_res = tgt_bbox
        bbox_idx = bbox_idx.to(tgt_bbox.device)

        return tgt_bbox_res, k_idx, bbox_idx

    @torch.no_grad()
    def forward(self, outputs, targets, indices, log=False):
        assert "pred_actions" in outputs, "There is no action output for pair matching"
        num_obj_queries = outputs["pred_boxes"].shape[1]
        bs, num_queries = outputs["pred_actions"].shape[:2]
        detr_query_num = outputs["pred_logits"].shape[1] \
            if (outputs["pred_oidx"].shape[-1] == (outputs["pred_logits"].shape[1] + 1)) else -1

        return_list = []
        if self.log_printer and log:
            log_dict = {'h_cost': [], 'o_cost': [], 'act_cost': []}
            if self.is_hico: log_dict['tgt_cost'] = []

        for batch_idx in range(bs):
            tgt_bbox = targets[batch_idx]["boxes"] # (num_boxes, 4)
            tgt_cls = targets[batch_idx]["labels"] # (num_boxes)

            if self.is_vcoco:
                targets[batch_idx]["pair_actions"][:, self.invalid_ids] = 0
                keep_idx = (targets[batch_idx]["pair_actions"].sum(dim=-1) != 0)
                targets[batch_idx]["pair_boxes"] = targets[batch_idx]["pair_boxes"][keep_idx]
                targets[batch_idx]["pair_actions"] = targets[batch_idx]["pair_actions"][keep_idx]
                targets[batch_idx]["pair_targets"] = targets[batch_idx]["pair_targets"][keep_idx]
                # TODO
                targets[batch_idx]["pair_interactions"] = targets[batch_idx]["pair_interactions"][keep_idx]

                tgt_pbox = targets[batch_idx]["pair_boxes"] # (num_pair_boxes, 8)
                tgt_act = targets[batch_idx]["pair_actions"] # (num_pair_boxes, 29)
                tgt_tgt = targets[batch_idx]["pair_targets"] # (num_pair_boxes)
                # TODO
                tgt_inter = targets[batch_idx]["pair_interactions"] # (num_pair_boxes)

                tgt_hbox = tgt_pbox[:, :4] # (num_pair_boxes, 4)
                tgt_obox = tgt_pbox[:, 4:] # (num_pair_boxes, 4)
            elif self.is_hico:
                tgt_act = targets[batch_idx]["pair_actions"] # (num_pair_boxes, 117)
                tgt_tgt = targets[batch_idx]["pair_targets"] # (num_pair_boxes)
                # TODO
                tgt_inter = targets[batch_idx]["pair_interactions"] # (num_pair_boxes)
                tgt_hbox = targets[batch_idx]["sub_boxes"] # (num_pair_boxes, 4)
                tgt_obox = targets[batch_idx]["obj_boxes"] # (num_pair_boxes, 4)

            # find which gt boxes match the h, o boxes in the pair
            if self.is_vcoco:
                hbox_with_cls = torch.cat([tgt_hbox, torch.ones((tgt_hbox.shape[0], 1)).to(tgt_hbox.device)], dim=1)
            elif self.is_hico:
                hbox_with_cls = torch.cat([tgt_hbox, torch.zeros((tgt_hbox.shape[0], 1)).to(tgt_hbox.device)], dim=1)
            
            tgt_tgt = torch.as_tensor(tgt_tgt, dtype=torch.float32, device = tgt_obox.device)
            obox_with_cls = torch.cat([tgt_obox, tgt_tgt.unsqueeze(-1)], dim=1)
            obox_with_cls[obox_with_cls[:, :4].sum(dim=1) == -4, -1] = -1 # turn the class of occluded objects to -1

            tgt_cls = torch.as_tensor(tgt_cls, dtype=torch.float32, device = tgt_bbox.device)
            bbox_with_cls = torch.cat([tgt_bbox, tgt_cls.unsqueeze(-1)], dim=1)
            bbox_with_cls, k_idx, bbox_idx = self.reduce_redundant_gt_box(bbox_with_cls, indices[batch_idx])
            bbox_with_cls = torch.cat((bbox_with_cls, torch.as_tensor([-1.]*5).unsqueeze(0).to(tgt_cls.device)), dim=0)

            cost_hbox = torch.cdist(hbox_with_cls, bbox_with_cls, p=1)
            cost_obox = torch.cdist(obox_with_cls, bbox_with_cls, p=1)

            # find which gt boxes matches which prediction in K
            h_match_indices = torch.nonzero(cost_hbox == 0, as_tuple=False) # (num_hbox, num_boxes)
            o_match_indices = torch.nonzero(cost_obox == 0, as_tuple=False) # (num_obox, num_boxes)

            tgt_hids, tgt_oids = [], []

            # obtain ground truth indices for h
            if len(h_match_indices) != len(o_match_indices):
                import pdb; pdb.set_trace()

            for h_match_idx, o_match_idx in zip(h_match_indices, o_match_indices):
                hbox_idx, H_bbox_idx = h_match_idx
                obox_idx, O_bbox_idx = o_match_idx
                if O_bbox_idx == (len(bbox_with_cls)-1): # if the object class is -1
                    O_bbox_idx = H_bbox_idx # happens in V-COCO, the target object may not appear

                GT_idx_for_H = (bbox_idx == H_bbox_idx).nonzero(as_tuple=False).squeeze(-1)
                query_idx_for_H = k_idx[GT_idx_for_H]
                tgt_hids.append(query_idx_for_H)

                GT_idx_for_O = (bbox_idx == O_bbox_idx).nonzero(as_tuple=False).squeeze(-1)
                query_idx_for_O = k_idx[GT_idx_for_O]
                tgt_oids.append(query_idx_for_O)

            # check if empty
            if len(tgt_hids) == 0: tgt_hids.append(torch.as_tensor([-1])) # we later ignore the label -1
            if len(tgt_oids) == 0: tgt_oids.append(torch.as_tensor([-1])) # we later ignore the label -1

            tgt_sum = (tgt_act.sum(dim=-1)).unsqueeze(0)
            flag = False
            if tgt_act.shape[0] == 0:
                tgt_act = torch.zeros((1, tgt_act.shape[1])).to(targets[batch_idx]["pair_actions"].device)
                targets[batch_idx]["pair_actions"] = torch.zeros((1, targets[batch_idx]["pair_actions"].shape[1])).to(targets[batch_idx]["pair_actions"].device)
                if self.is_hico:
                    pad_tgt = -1 # outputs["pred_obj_logits"].shape[-1]-1
                    tgt_tgt = torch.tensor([pad_tgt]).to(targets[batch_idx]["pair_targets"])
                    targets[batch_idx]["pair_targets"] = torch.tensor([pad_tgt]).to(targets[batch_idx]["pair_targets"].device)
                    # TODO
                    pad_tgt = -1 # newly added
                    tgt_inter = torch.tensor([pad_tgt]).to(targets[batch_idx]["pair_interactions"])
                    targets[batch_idx]["pair_interactions"] = torch.tensor([pad_tgt]).to(targets[batch_idx]["pair_interactions"].device) # newly added
                    
                tgt_sum = (tgt_act.sum(dim=-1) + 1).unsqueeze(0)

            # Concat target label
            tgt_hids = torch.cat(tgt_hids)
            tgt_oids = torch.cat(tgt_oids)

            out_hprob = outputs["pred_hidx"][batch_idx].softmax(-1)
            out_oprob = outputs["pred_oidx"][batch_idx].softmax(-1)
            out_act = outputs["pred_actions"][batch_idx].clone()
           
            if self.is_vcoco: out_act[..., self.invalid_ids] = 0
            
            if self.is_hico:
                out_tgt = outputs["pred_obj_logits"][batch_idx].softmax(-1)
                out_tgt[..., -1] = 0 # don't get cost for no-object
                # TODO
                out_inter = outputs["pred_interactiveness"][batch_idx].softmax(-1)
                
            tgt_act = torch.as_tensor(tgt_act, dtype=torch.int64, device = tgt_act.device)
            tgt_act = torch.cat([tgt_act, torch.zeros(tgt_act.shape[0], dtype=torch.int64).unsqueeze(-1).to(tgt_act.device)], dim=-1)
            cost_hclass = -out_hprob[:, tgt_hids] # [batch_size * num_queries, detr.num_queries+1]
            cost_oclass = -out_oprob[:, tgt_oids] # [batch_size * num_queries, detr.num_queries+1]

            cost_pos_act = (-torch.matmul(out_act, tgt_act.t().float())) / tgt_sum
            cost_neg_act = (torch.matmul(out_act, (~tgt_act.bool()).type(torch.int64).t().float())) / (~tgt_act.bool()).type(torch.int64).sum(dim=-1).unsqueeze(0)
            cost_action = cost_pos_act + cost_neg_act

            h_cost = self.cost_hbox * cost_hclass
            o_cost = self.cost_obox * cost_oclass
            act_cost = self.cost_action * cost_action

            C = h_cost + o_cost + act_cost
            if self.is_hico:
                tgt_tgt = torch.as_tensor(tgt_tgt, dtype=torch.int64)
                cost_target = -out_tgt[:, tgt_tgt]
                tgt_cost = self.cost_target * cost_target
                C += tgt_cost
                # TODO
                tgt_inter = torch.as_tensor(tgt_inter, dtype=torch.int64)
                cost_interaction = -out_inter[:, tgt_inter]
                inter_cost = self.cost_interaction * cost_interaction
                C += inter_cost

            C = C.view(num_queries, -1).cpu()

            return_list.append(linear_sum_assignment(C))
            targets[batch_idx]["h_labels"] = tgt_hids.to(tgt_hbox.device)
            targets[batch_idx]["o_labels"] = tgt_oids.to(tgt_obox.device)
            log_act_cost = torch.zeros([1]).to(tgt_act.device) if tgt_act.shape[0] == 0 else act_cost.min(dim=0)[0].mean()

            if self.log_printer and log:
                log_dict['h_cost'].append(h_cost.min(dim=0)[0].mean())
                log_dict['o_cost'].append(o_cost.min(dim=0)[0].mean())
                log_dict['act_cost'].append(act_cost.min(dim=0)[0].mean())
                if self.is_hico: 
                    log_dict['tgt_cost'].append(tgt_cost.min(dim=0)[0].mean())
                    # log_dict['inter_cost'].append(inter_cost.min(dim=0)[0].mean())
        if self.log_printer and log:
            log_dict['h_cost'] = torch.stack(log_dict['h_cost']).mean()
            log_dict['o_cost'] = torch.stack(log_dict['o_cost']).mean()
            log_dict['act_cost'] = torch.stack(log_dict['act_cost']).mean()
            if self.is_hico: 
                log_dict['tgt_cost'] = torch.stack(log_dict['tgt_cost']).mean()
                # log_dict['inter_cost'] = torch.stack(log_dict['inter_cost']).mean()

            if utils.get_rank() == 0: logger.info(f'log_dict: {log_dict}')
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in return_list], targets

def build_hoi_matcher(args):
    return HungarianPairMatcher(args)

###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################

# class HungarianPairMatcher(nn.Module):
#     def __init__(self, args):
#         """Creates the matcher
#         Params:
#             cost_action: This is the relative weight of the multi-label action classification error in the matching cost
#             cost_hbox: This is the relative weight of the classification error for human idx in the matching cost
#             cost_obox: This is the relative weight of the classification error for object idx in the matching cost
#         """
#         super().__init__()
#         self.cost_action = args.set_cost_act
#         self.cost_hbox = self.cost_obox = args.set_cost_idx
#         self.cost_target = args.set_cost_tgt
#         self.log_printer = True
#         self.is_vcoco = (args.dataset_file == 'vcoco')
#         self.is_hico = (args.dataset_file == 'hico-det')
#         if self.is_vcoco:
#             self.valid_ids = args.valid_ids
#             self.invalid_ids = args.invalid_ids
#         assert self.cost_action != 0 or self.cost_hbox != 0 or self.cost_obox != 0, "all costs cant be 0"

#     def reduce_redundant_gt_box(self, tgt_bbox, indices):
#         """Filters redundant Ground-Truth Bounding Boxes
#         Due to random crop augmentation, there exists cases where there exists
#         multiple redundant labels for the exact same bounding box and object class.
#         This function deals with the redundant labels for smoother HOTR training.
#         """
#         tgt_bbox_unique, map_idx, idx_cnt = torch.unique(tgt_bbox, dim=0, return_inverse=True, return_counts=True)

#         k_idx, bbox_idx = indices
#         triggered = False
#         if (len(tgt_bbox) != len(tgt_bbox_unique)):
#             map_dict = {k: v for k, v in enumerate(map_idx)}
#             map_bbox2kidx = {int(bbox_id): k_id for bbox_id, k_id in zip(bbox_idx, k_idx)}

#             bbox_lst, k_lst = [], []
#             for bbox_id in bbox_idx:
#                 if map_dict[int(bbox_id)] not in bbox_lst:
#                     bbox_lst.append(map_dict[int(bbox_id)])
#                     k_lst.append(map_bbox2kidx[int(bbox_id)])
#             bbox_idx = torch.tensor(bbox_lst)
#             k_idx = torch.tensor(k_lst)
#             tgt_bbox_res = tgt_bbox_unique
#         else:
#             tgt_bbox_res = tgt_bbox
#         bbox_idx = bbox_idx.to(tgt_bbox.device)

#         return tgt_bbox_res, k_idx, bbox_idx

#     @torch.no_grad()
#     def forward(self, outputs, targets, indices, log=False):
#         assert "pred_actions" in outputs, "There is no action output for pair matching"
#         num_obj_queries = outputs["pred_boxes"].shape[1]
#         bs, num_queries = outputs["pred_actions"].shape[:2]
#         detr_query_num = outputs["pred_logits"].shape[1] \
#             if (outputs["pred_oidx"].shape[-1] == (outputs["pred_logits"].shape[1] + 1)) else -1

#         return_list = []
#         if self.log_printer and log:
#             log_dict = {'h_cost': [], 'o_cost': [], 'act_cost': []}
#             if self.is_hico: log_dict['tgt_cost'] = []

#         for batch_idx in range(bs):
#             tgt_bbox = targets[batch_idx]["boxes"] # (num_boxes, 4)
#             tgt_cls = targets[batch_idx]["labels"] # (num_boxes)

#             if self.is_vcoco:
#                 targets[batch_idx]["pair_actions"][:, self.invalid_ids] = 0
#                 keep_idx = (targets[batch_idx]["pair_actions"].sum(dim=-1) != 0)
#                 targets[batch_idx]["pair_boxes"] = targets[batch_idx]["pair_boxes"][keep_idx]
#                 targets[batch_idx]["pair_actions"] = targets[batch_idx]["pair_actions"][keep_idx]
#                 targets[batch_idx]["pair_targets"] = targets[batch_idx]["pair_targets"][keep_idx]

#                 tgt_pbox = targets[batch_idx]["pair_boxes"] # (num_pair_boxes, 8)
#                 tgt_act = targets[batch_idx]["pair_actions"] # (num_pair_boxes, 29)
#                 tgt_tgt = targets[batch_idx]["pair_targets"] # (num_pair_boxes)

#                 tgt_hbox = tgt_pbox[:, :4] # (num_pair_boxes, 4)
#                 tgt_obox = tgt_pbox[:, 4:] # (num_pair_boxes, 4)
#             elif self.is_hico:
#                 tgt_act = targets[batch_idx]["pair_actions"] # (num_pair_boxes, 117)
#                 tgt_tgt = targets[batch_idx]["pair_targets"] # (num_pair_boxes)

#                 tgt_hbox = targets[batch_idx]["sub_boxes"] # (num_pair_boxes, 4)
#                 tgt_obox = targets[batch_idx]["obj_boxes"] # (num_pair_boxes, 4)

#             # find which gt boxes match the h, o boxes in the pair
#             if self.is_vcoco:
#                 hbox_with_cls = torch.cat([tgt_hbox, torch.ones((tgt_hbox.shape[0], 1)).to(tgt_hbox.device)], dim=1)
#             elif self.is_hico:
#                 hbox_with_cls = torch.cat([tgt_hbox, torch.zeros((tgt_hbox.shape[0], 1)).to(tgt_hbox.device)], dim=1)
            
#             tgt_tgt = torch.as_tensor(tgt_tgt, dtype=torch.float32, device = tgt_obox.device) # newly added                
#             obox_with_cls = torch.cat([tgt_obox, tgt_tgt.unsqueeze(-1)], dim=1)
#             obox_with_cls[obox_with_cls[:, :4].sum(dim=1) == -4, -1] = -1 # turn the class of occluded objects to -1

#             tgt_cls = torch.as_tensor(tgt_cls, dtype=torch.float32, device = tgt_bbox.device) # newly added
#             bbox_with_cls = torch.cat([tgt_bbox, tgt_cls.unsqueeze(-1)], dim=1)
#             bbox_with_cls, k_idx, bbox_idx = self.reduce_redundant_gt_box(bbox_with_cls, indices[batch_idx])
#             bbox_with_cls = torch.cat((bbox_with_cls, torch.as_tensor([-1.]*5).unsqueeze(0).to(tgt_cls.device)), dim=0)

#             cost_hbox = torch.cdist(hbox_with_cls, bbox_with_cls, p=1)
#             cost_obox = torch.cdist(obox_with_cls, bbox_with_cls, p=1)

#             # find which gt boxes matches which prediction in K
#             h_match_indices = torch.nonzero(cost_hbox == 0, as_tuple=False) # (num_hbox, num_boxes)
#             o_match_indices = torch.nonzero(cost_obox == 0, as_tuple=False) # (num_obox, num_boxes)

#             tgt_hids, tgt_oids = [], []

#             # obtain ground truth indices for h
#             if len(h_match_indices) != len(o_match_indices):
#                 import pdb; pdb.set_trace()

#             for h_match_idx, o_match_idx in zip(h_match_indices, o_match_indices):
#                 hbox_idx, H_bbox_idx = h_match_idx
#                 obox_idx, O_bbox_idx = o_match_idx
#                 if O_bbox_idx == (len(bbox_with_cls)-1): # if the object class is -1
#                     O_bbox_idx = H_bbox_idx # happens in V-COCO, the target object may not appear

#                 GT_idx_for_H = (bbox_idx == H_bbox_idx).nonzero(as_tuple=False).squeeze(-1)
#                 query_idx_for_H = k_idx[GT_idx_for_H]
#                 tgt_hids.append(query_idx_for_H)

#                 GT_idx_for_O = (bbox_idx == O_bbox_idx).nonzero(as_tuple=False).squeeze(-1)
#                 query_idx_for_O = k_idx[GT_idx_for_O]
#                 tgt_oids.append(query_idx_for_O)

#             # check if empty
#             if len(tgt_hids) == 0: tgt_hids.append(torch.as_tensor([-1])) # we later ignore the label -1
#             if len(tgt_oids) == 0: tgt_oids.append(torch.as_tensor([-1])) # we later ignore the label -1

#             tgt_sum = (tgt_act.sum(dim=-1)).unsqueeze(0)
#             flag = False
#             if tgt_act.shape[0] == 0:
#                 tgt_act = torch.zeros((1, tgt_act.shape[1])).to(targets[batch_idx]["pair_actions"].device)
#                 targets[batch_idx]["pair_actions"] = torch.zeros((1, targets[batch_idx]["pair_actions"].shape[1])).to(targets[batch_idx]["pair_actions"].device)
#                 if self.is_hico:
#                     pad_tgt = -1 # outputs["pred_obj_logits"].shape[-1]-1
#                     tgt_tgt = torch.tensor([pad_tgt]).to(targets[batch_idx]["pair_targets"])
#                     targets[batch_idx]["pair_targets"] = torch.tensor([pad_tgt]).to(targets[batch_idx]["pair_targets"].device)
#                 tgt_sum = (tgt_act.sum(dim=-1) + 1).unsqueeze(0)

#             # Concat target label
#             tgt_hids = torch.cat(tgt_hids)
#             tgt_oids = torch.cat(tgt_oids)

#             out_hprob = outputs["pred_hidx"][batch_idx].softmax(-1)
#             out_oprob = outputs["pred_oidx"][batch_idx].softmax(-1)
#             out_act = outputs["pred_actions"][batch_idx].clone()
#             if self.is_vcoco: out_act[..., self.invalid_ids] = 0
#             if self.is_hico:
#                 out_tgt = outputs["pred_obj_logits"][batch_idx].softmax(-1)
#                 out_tgt[..., -1] = 0 # don't get cost for no-object
#             tgt_act = torch.cat([tgt_act, torch.zeros(tgt_act.shape[0]).unsqueeze(-1).to(tgt_act.device)], dim=-1)

#             cost_hclass = -out_hprob[:, tgt_hids] # [batch_size * num_queries, detr.num_queries+1]
#             cost_oclass = -out_oprob[:, tgt_oids] # [batch_size * num_queries, detr.num_queries+1]

#             cost_pos_act = (-torch.matmul(out_act, tgt_act.t().float())) / tgt_sum
#             cost_neg_act = (torch.matmul(out_act, (~tgt_act.bool()).type(torch.int64).t().float())) / (~tgt_act.bool()).type(torch.int64).sum(dim=-1).unsqueeze(0)
#             cost_action = cost_pos_act + cost_neg_act

#             h_cost = self.cost_hbox * cost_hclass
#             o_cost = self.cost_obox * cost_oclass
#             act_cost = self.cost_action * cost_action

#             C = h_cost + o_cost + act_cost
#             if self.is_hico:
#                 tgt_tgt = torch.as_tensor(tgt_tgt, dtype=torch.int64)
#                 cost_target = -out_tgt[:, tgt_tgt]
#                 tgt_cost = self.cost_target * cost_target
#                 C += tgt_cost
#             C = C.view(num_queries, -1).cpu()

#             return_list.append(linear_sum_assignment(C))
#             targets[batch_idx]["h_labels"] = tgt_hids.to(tgt_hbox.device)
#             targets[batch_idx]["o_labels"] = tgt_oids.to(tgt_obox.device)
#             log_act_cost = torch.zeros([1]).to(tgt_act.device) if tgt_act.shape[0] == 0 else act_cost.min(dim=0)[0].mean()

#             if self.log_printer and log:
#                 log_dict['h_cost'].append(h_cost.min(dim=0)[0].mean())
#                 log_dict['o_cost'].append(o_cost.min(dim=0)[0].mean())
#                 log_dict['act_cost'].append(act_cost.min(dim=0)[0].mean())
#                 if self.is_hico: log_dict['tgt_cost'].append(tgt_cost.min(dim=0)[0].mean())
#         if self.log_printer and log:
#             log_dict['h_cost'] = torch.stack(log_dict['h_cost']).mean()
#             log_dict['o_cost'] = torch.stack(log_dict['o_cost']).mean()
#             log_dict['act_cost'] = torch.stack(log_dict['act_cost']).mean()
#             if self.is_hico: log_dict['tgt_cost'] = torch.stack(log_dict['tgt_cost']).mean()
#             if utils.get_rank() == 0: logger.info(log_dict)
#         return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in return_list], targets

# def build_hoi_matcher(args):
#     return HungarianPairMatcher(args)