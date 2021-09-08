# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import copy
import time
import datetime
import torch.nn as nn
import torch.nn.functional as F

from hotr.util.misc import NestedTensor, nested_tensor_from_tensor_list
from .feed_forward import MLP

class HOTR(nn.Module):
    def __init__(self, 
                 detr,
                 num_hoi_queries,
                 num_actions,
                 interaction_transformer,
                 freeze_detr,
                 share_enc,
                 pretrained_dec,
                 temperature,
                 hoi_aux_loss,
                 return_obj_class=None,
                 ):
        super().__init__()
        # * Instance Transformer --------------------------------------------
        self.detr = detr
        if freeze_detr:
            # if this flag is given, freeze the object detection related parameters of DETR
            for p in self.parameters():
                p.requires_grad_(False)
        hidden_dim = detr.transformer.d_model
        object_d_model = detr.transformer.object_d_model
        # -------------------------------------------------------------------
        # * Interaction Transformer -----------------------------------------
        self.num_queries = num_hoi_queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.H_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.O_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.action_embed = nn.Linear(hidden_dim, num_actions + 1)
        # TODO
        self.object_embed =  nn.Linear(object_d_model, hidden_dim)
        self.interaction_embed = MLP(2*hidden_dim, hidden_dim, 2, 3)
        # --------------------------------------------------------------------
        # * HICO-DET FFN heads -----------------------------------------------
        self.return_obj_class = (return_obj_class is not None)
        if return_obj_class: self._valid_obj_ids = return_obj_class + [return_obj_class[-1]+1]
        # if return_obj_class: self._valid_obj_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
        # 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,
        # 73,74,75,76,77,78,79,80,81]

        # * Transformer Options ---------------------------------------------
        self.interaction_transformer = interaction_transformer
        if share_enc: # share encoder
            self.interaction_transformer.encoder = detr.transformer.encoder
        if pretrained_dec: # free variables for interaction decoder
            self.interaction_transformer.decoder = copy.deepcopy(detr.transformer.decoder)
            for p in self.interaction_transformer.decoder.parameters():
                p.requires_grad_(True)
        # -------------------------------------------------------------------
        # * Loss Options -------------------
        self.tau = temperature
        self.hoi_aux_loss = hoi_aux_loss
        # ----------------------------------

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        ############################################################  BACKBONE LAYERS   ############################################################
    
        features, pos = self.detr.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        start_time = time.time()
        instance_hs, _ = self.detr.transformer(self.detr.input_proj(src), mask, self.detr.query_embed.weight, pos[-1]) # hs.shape torch.Size([6, 2, 100, 256])
        
        inst_repr = F.normalize(instance_hs[-1], p=2, dim=2) # instance representations inst_repr.shape torch.Size([2, 100, 300])        
        # prediction heads for object detection
        # self.detr.class_embed.shape torch.Size([80, 300])
        self.obj_classifier = self.object_embed(self.detr.class_embed) # obj_class_logits.shape  torch.Size([92, 256])
        # outputs_class = self.detr.class_embed(instance_hs) # outputs_class.shape torch.Size([6, 2, 100, 92])
        outputs_class = torch.stack([torch.matmul(inst_repr, self.obj_classifier.transpose(0, 1)) for inst_repr in instance_hs]) # outputs_class.shape torch.Size([6, 2, 100, 92]) -> torch.Size([2,100,92])
        outputs_coord = self.detr.bbox_embed(instance_hs).sigmoid()
        object_detection_time = time.time() - start_time

        ########################################################### HOI DETECTION LAYERS  ############################################################

        start_time = time.time()
        assert hasattr(self, 'interaction_transformer'), "Missing Interaction Transformer."
        # interaction_hs.shape | torch.Size([6, 2, 16, 256])
        interaction_hs = self.interaction_transformer(self.detr.input_proj(src), mask, self.query_embed.weight, pos[-1])[0] # interaction representations
        # [HO Pointers]
        # H_Pointer_reprs.shape torch.Size([6, 2, 16, 256])
        H_Pointer_reprs = F.normalize(self.H_Pointer_embed(interaction_hs), p=2, dim=-1)
        O_Pointer_reprs = F.normalize(self.O_Pointer_embed(interaction_hs), p=2, dim=-1)
        # len(outputs_hidx) == 6 | outputs_hidx[0].shape torch.Size([2, 16, 100]) | outputs_oidx[0].shape torch.Size([2, 16, 100])
        outputs_hidx = [(torch.bmm(H_Pointer_repr, inst_repr.transpose(1,2))) / self.tau for H_Pointer_repr in H_Pointer_reprs]
        outputs_oidx = [(torch.bmm(O_Pointer_repr, inst_repr.transpose(1,2))) / self.tau for O_Pointer_repr in O_Pointer_reprs]

        # [Interactiveness Classification]
        # relational network com-putes the score by concatenating feature of H_Pointer_repr and O_Pointer_repr.
        outputs_interaction_score = self.interaction_embed(torch.cat((H_Pointer_reprs, O_Pointer_reprs), dim=-1)) # torch.Size([6, 2, 16, 2])

        # [Action Classification]
        outputs_action = self.action_embed(interaction_hs)
        hoi_detection_time = time.time() - start_time
        hoi_recognition_time = max(hoi_detection_time - object_detection_time, 0)

        #[Target Classification]
        if self.return_obj_class:
            # detr_logits.shape | torch.Size([2, 100, 81])
            # original self._valid_obj_ids
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 
            # 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
            # 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 
            # 88, 89, 90, 91]
            detr_logits = outputs_class[-1, ..., self._valid_obj_ids]
            # len(o_indices) == 6 | o_indices[-1].shape torch.Size([2, 16])
            o_indices = [output_oidx.max(-1)[-1] for output_oidx in outputs_oidx]
            obj_logit_stack = [torch.stack([detr_logits[batch_, o_idx, :] for batch_, o_idx in enumerate(o_indice)], 0) for o_indice in o_indices]
            outputs_obj_class = obj_logit_stack

        # outputs_interaction_score[-1].shape torch.Size([2, 16, 2])        
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_hidx": outputs_hidx[-1],
            "pred_oidx": outputs_oidx[-1],
            "pred_actions": outputs_action[-1],
            "pred_interactiveness": outputs_interaction_score[-1],
            "hoi_recognition_time": hoi_recognition_time,
        }

        # outputs_action[-1].shape | torch.size([2,16,118])
        # outputs_obj_class[-1].shape | torch.Size([2, 16, 81])
        if self.return_obj_class: out["pred_obj_logits"] = outputs_obj_class[-1]

        if self.hoi_aux_loss: # auxiliary loss
            out['hoi_aux_outputs'] = \
                self._set_aux_loss_with_tgt(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_interaction_score, outputs_obj_class) \
                if self.return_obj_class else \
                self._set_aux_loss(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_interaction_score)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_interaction_score):
        return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_interactiveness':f}
                for a, b, c, d, e, f in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1],
                    outputs_interaction_score[:-1],
                    )]

    @torch.jit.unused
    def _set_aux_loss_with_tgt(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_interaction_score, outputs_tgt):
        return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_interactiveness':f, 'pred_obj_logits': g}
                for a, b, c, d, e, f, g in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1],
                    outputs_interaction_score[:-1],
                    outputs_tgt[:-1])]